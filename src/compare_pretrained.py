"""Manually train the models in models.md and accumulate held-out comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

if __package__:
    from .device import training_device, cpu_state_dict
    from .pretrained_models import (MODEL_NAMES, DISPLAY_NAMES, build_model,
                                    classification_loss, preprocessing, weight_name)
else:
    from device import training_device, cpu_state_dict
    from pretrained_models import (MODEL_NAMES, DISPLAY_NAMES, build_model,
                                   classification_loss, preprocessing, weight_name)

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace leaves the previous complete file available on interruption.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def dataset_manifest(data_dir, seed):
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError('Dataset needs at least two class directories.')
    entries = []
    for label, name in enumerate(classes):
        paths = sorted(p for p in (data_dir / name).rglob('*')
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f'No images in class: {name}')
        entries.extend({'path': p.relative_to(data_dir).as_posix(), 'label': label,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
    labels = [entry['label'] for entry in entries]
    train, rest = train_test_split(list(range(len(entries))), test_size=.3,
                                   stratify=labels, random_state=seed)
    validation, test = train_test_split(rest, test_size=.5, random_state=seed,
                                        stratify=[labels[i] for i in rest])
    manifest = {'seed': seed, 'class_names': classes, 'images': entries,
                'splits': {'train': train, 'validation': validation, 'test': test}}
    manifest['split_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def ensure_experiment(result_dir, manifest, config):
    """Different model selections may share an experiment; budgets may not."""
    for filename, expected in (('split.json', manifest), ('experiment.json', config)):
        path = result_dir / filename
        if path.exists() and json.loads(path.read_text('utf-8')) != expected:
            raise ValueError(f'{filename} differs; choose a new --run-dir for different data/settings.')
    write_json(result_dir / 'split.json', manifest)
    write_json(result_dir / 'experiment.json', config)


class LeafImages(Dataset):
    def __init__(self, data_dir, manifest, split, transform):
        self.root, self.entries = data_dir, manifest['images']
        self.indices, self.transform = manifest['splits'][split], transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        entry = self.entries[self.indices[index]]
        with Image.open(self.root / entry['path']) as source:
            image = self.transform(source.convert('RGB'))
        return image, entry['label']


def evaluate(model, loader, classes):
    model.eval()
    device = next(model.parameters()).device
    truth, predictions, probabilities = [], [], []
    total_loss = 0.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            total_loss += torch.nn.functional.cross_entropy(
                logits, labels.to(device), reduction='sum').item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
    report = classification_report(truth, predictions, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    metrics = {'loss': total_loss / len(truth),
               'accuracy': sum(a == b for a, b in zip(truth, predictions)) / len(truth),
               'macro_f1': report['macro avg']['f1-score'], 'classification_report': report,
               'confusion_matrix': confusion_matrix(truth, predictions,
                                                    labels=list(range(len(classes)))).tolist()}
    return metrics, truth, predictions, probabilities


def synchronize(device):
    if device.type == 'xpu':
        torch.xpu.synchronize()


def train_one(name, args, manifest, config):
    classes = manifest['class_names']
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'xpu':
        torch.xpu.manual_seed_all(args.seed)
    # Weight download/load time is excluded from training_seconds.
    model = build_model(name, len(classes)).to(args.device)
    train_data = LeafImages(args.data_dir, manifest, 'train', preprocessing(name, training=True))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=len(train_data) % args.batch_size == 1,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(LeafImages(args.data_dir, manifest, 'validation', preprocessing(name)),
                                   batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    checkpoint_path = args.run_dir / 'checkpoints' / f'{name}.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, history = float('inf'), None, []
    synchronize(args.device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0., 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError(f'{name}: nonfinite training loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        validation, _, _, _ = evaluate(model, validation_loader, classes)
        if not np.isfinite(validation['loss']):
            raise ValueError(f'{name}: nonfinite validation loss at epoch {epoch}')
        history.append({'epoch': epoch, 'train_loss': loss_sum / count, 'validation': validation})
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            temporary_checkpoint = checkpoint_path.with_suffix('.pth.tmp')
            torch.save({'architecture': name, 'model_state_dict': cpu_state_dict(model),
                        'class_names': classes, 'weights': weight_name(name),
                        'preprocessing': str(preprocessing(name)), 'experiment': config,
                        'split_id': manifest['split_id'], 'best_epoch': epoch,
                        'validation': validation}, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
        write_json(args.run_dir / f'{name}_history.json', history)
        print(f'{name} {epoch:02d}/{args.epochs} train={loss_sum/count:.4f} '
              f'val={validation["loss"]:.4f} F1={validation["macro_f1"]:.4f}', flush=True)
    synchronize(args.device)
    seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_data = LeafImages(args.data_dir, manifest, 'test', preprocessing(name))
    test, truth, predictions, probabilities = evaluate(
        model, DataLoader(test_data, batch_size=args.batch_size, num_workers=0), classes)
    result = {'model': name, 'display_name': DISPLAY_NAMES[name], 'weights': weight_name(name),
              'initialization': ('random' if name == 'rice_cnn' else
                                 'ImageNet backbone; random task head and any added attention'),
              'preprocessing': str(preprocessing(name)), 'split_id': manifest['split_id'],
              'experiment': config, 'best_epoch': best_epoch, 'best_validation_loss': best_loss,
              'training_seconds': seconds, 'parameters': sum(p.numel() for p in model.parameters()),
              'class_names': classes, 'test': test,
              'lar': None, 'lesion_occlusion_confidence_drop': None,
              'localization_status': '待測：需定義及病斑標註',
              'predictions': [{'path': manifest['images'][i]['path'], 'true_class': classes[y],
                               'predicted_class': classes[p], 'probabilities': dict(zip(classes, probs))}
                              for i, y, p, probs in zip(test_data.indices, truth, predictions, probabilities)]}
    write_json(args.run_dir / f'{name}_metrics.json', result)
    print(f'{name}: test Macro-F1={test["macro_f1"]:.4f}, training={seconds:.1f}s', flush=True)


def write_summary(run_dir):
    rows, split_ids = {}, set()
    configurations = set()
    experiment_path = run_dir / 'experiment.json'
    expected = json.loads(experiment_path.read_text('utf-8')) if experiment_path.exists() else None
    for name in MODEL_NAMES:
        path = run_dir / f'{name}_metrics.json'
        result = json.loads(path.read_text('utf-8')) if path.exists() else None
        if result:
            split_ids.add(result['split_id'])
            if result.get('model') != name:
                raise ValueError(f'{path.name}: model identity differs.')
            configurations.add(json.dumps(result.get('experiment'), sort_keys=True))
            if expected is not None and result.get('experiment') != expected:
                raise ValueError(f'{path.name}: training settings differ from experiment.json.')
        rows[name] = {'model': DISPLAY_NAMES[name],
                      'macro_f1': result['test']['macro_f1'] if result else None,
                      'accuracy': result['test']['accuracy'] if result else None,
                      'lar': None, 'lesion_occlusion_confidence_drop': None,
                      'training_seconds': result['training_seconds'] if result else None,
                      'weights': result['weights'] if result else weight_name(name)}
    if len(split_ids) > 1:
        raise ValueError('Cannot summarize results from different splits.')
    if len(configurations) > 1:
        raise ValueError('Cannot summarize results with different training settings.')
    write_json(run_dir / 'comparison.json', rows)
    with (run_dir / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    lines = ['| 模型 | Macro F1 ↑ | Accuracy ↑ | LAR ↑ | 病斑遮蔽後信心下降 ↑ | 訓練時間（秒）↓ |',
             '|---|---|---|---|---|---|']
    for row in rows.values():
        def number(key):
            return '待測' if row[key] is None else f'{row[key]:.4f}'
        lines.append(f'| {row["model"]} | {number("macro_f1")} | {number("accuracy")} | 待測 | 待測 | {number("training_seconds")} |')
    (run_dir / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/rice_leaf_diseases')
    parser.add_argument('--run-dir', type=Path, help='Default: results/pretrained_<device>')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true', help='Retrain selected models, replacing their files')
    parser.add_argument('--summarize-only', action='store_true', help='Aggregate saved metrics without training or GPU')
    args = parser.parse_args()
    args.run_dir = (args.run_dir or ROOT / f'results/pretrained_{args.device}').resolve()
    try:
        if args.summarize_only:
            if not any(args.run_dir.glob('*_metrics.json')):
                raise ValueError(f'No completed model results in {args.run_dir}')
            write_summary(args.run_dir)
            print(f'Summary: {args.run_dir / "comparison.md"}')
            return
        if (args.epochs < 1 or args.batch_size < 2 or args.threads < 1
                or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
                or not np.isfinite(args.weight_decay) or args.weight_decay < 0):
            raise ValueError('Require epochs/threads >= 1, batch-size >= 2, finite lr > 0, weight-decay >= 0.')
        if len(args.models) != len(set(args.models)):
            raise ValueError('Model names must be unique.')
        args.device = training_device(args.device)
        args.data_dir = args.data_dir.resolve()
        torch.set_num_threads(args.threads)
        # Prefer deterministic kernels where supported; record this policy and
        # warn if an XPU operation cannot be deterministic rather than aborting.
        torch.use_deterministic_algorithms(True, warn_only=True)
        manifest = dataset_manifest(args.data_dir, args.seed)
        if len(manifest['splits']['train']) < 2:
            raise ValueError('Need at least two training images.')
        config = {'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
                  'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                  'optimizer': 'AdamW', 'fine_tuning': 'all parameters', 'threads': args.threads,
                  'device': str(args.device),
                  'device_name': torch.xpu.get_device_name(args.device) if args.device.type == 'xpu' else 'CPU',
                  'torch': str(torch.__version__), 'torchvision': __import__('torchvision').__version__,
                  'split_id': manifest['split_id'], 'selection': 'minimum validation cross entropy',
                  'preprocessing_policy': 'weights transforms; RandomResizedCrop scale 0.8-1, flip, rotation15, jitter0.15',
                  'deterministic': 'warn_only',
                  'weights': {name: weight_name(name) for name in MODEL_NAMES}}
        for name in args.models:
            if not args.overwrite and ((args.run_dir / 'checkpoints' / f'{name}.pth').exists()
                                      or (args.run_dir / f'{name}_metrics.json').exists()):
                raise ValueError(f'{name} already exists. Select untrained models, a new --run-dir, or --overwrite.')
        ensure_experiment(args.run_dir, manifest, config)
        print(f'Device: {args.device}; data: ' + ', '.join(f'{k}={len(v)}' for k, v in manifest['splits'].items()), flush=True)
        for name in args.models:
            # Never expose a stale completed result while replacing its checkpoint.
            old_metrics = args.run_dir / f'{name}_metrics.json'
            if old_metrics.exists():
                old_metrics.unlink()
            write_summary(args.run_dir)
            train_one(name, args, manifest, config)
            write_summary(args.run_dir)
        print(f'Results: {args.run_dir / "comparison.md"}', flush=True)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
