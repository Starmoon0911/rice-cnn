"""Train architecture baselines on one saved split and compare held-out Grad-CAMs."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

if __package__:
    from .device import training_device, cpu_state_dict
    from .model import RiceCNN
    from .grad_cam import GradCAM, EXTENSIONS
else:
    from device import training_device, cpu_state_dict
    from model import RiceCNN
    from grad_cam import GradCAM, EXTENSIONS

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ('rice_cnn', 'resnet18', 'mobilenet_v3_small', 'swin_t')
FAMILIES = {
    'rice_cnn': 'Plain CNN', 'resnet18': 'Residual CNN',
    'mobilenet_v3_small': 'Depthwise separable CNN', 'swin_t': 'Windowed Transformer',
}


def channels_first(tensor):
    """Torchvision Swin uses [batch, height, width, channels]."""
    return tensor.permute(0, 3, 1, 2)


def build_model(name, num_classes):
    if name == 'rice_cnn':
        model = RiceCNN(num_classes)
        return model, model.features[12], None, 'features.12 (last convolution)'
    if name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
        return model, model.layer4[-1], None, 'layer4.1 (last residual block)'
    if name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        return model, model.features[-1], None, 'features.12 (last convolution block)'
    if name == 'swin_t':
        model = models.swin_t(weights=None, num_classes=num_classes)
        return model, model.norm, channels_first, 'norm (final spatial LayerNorm, NHWC to NCHW)'
    raise ValueError(f'Unknown model: {name}')


def make_splits(labels, seed):
    indices = list(range(len(labels)))
    train, remaining = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=seed, stratify=[labels[i] for i in remaining]
    )
    return {'train': list(train), 'validation': list(validation), 'test': list(test)}


def select_examples(test_indices, labels, per_class):
    if per_class < 0:
        raise ValueError('samples-per-class must be nonnegative')
    selected = []
    for label in sorted(set(labels[i] for i in test_indices)):
        candidates = sorted(i for i in test_indices if labels[i] == label)
        selected.extend(candidates[:per_class] if per_class else candidates)
    return selected


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_training_config(name, result_dir):
    experiment_path = result_dir / 'experiment.json'
    training_path = result_dir / f'{name}_training.json'
    if not experiment_path.is_file() or not training_path.is_file():
        raise ValueError(f'{name}: missing experiment/training metadata; train this model first.')
    experiment = json.loads(experiment_path.read_text('utf-8'))
    training = json.loads(training_path.read_text('utf-8'))
    if name not in experiment['models']:
        raise ValueError(f'{name}: model was not trained in the saved experiment; select its --models explicitly when training.')
    expected = {k: experiment[k] for k in ('epochs', 'seed', 'batch_size', 'learning_rate')}
    expected.update(optimizer='AdamW', weight_decay=1e-4)
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f'{name}: training {key} differs from experiment.json; use separate experiment directories.')
    return training


class CachedImages(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images, self.labels, self.indices = images, labels, indices
        steps = []
        if training:
            steps.extend([
                transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15),
            ])
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return self.transform(self.images[i]), self.labels[i]


def evaluate(model, loader, num_classes):
    model.eval()
    device = next(model.parameters()).device
    total_loss, truth, predictions = 0., [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
    return {
        'loss': total_loss / len(truth),
        'accuracy': accuracy_score(truth, predictions),
        'macro_f1': f1_score(truth, predictions, labels=list(range(num_classes)), average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(truth, predictions, labels=list(range(num_classes))).tolist(),
    }


def train_one(name, images, labels, classes, splits, split_id, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, _, _, _ = build_model(name, len(classes))
    model.to(args.device)
    dataset = CachedImages(images, labels, splits['train'], training=True)
    train = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        drop_last=len(dataset) % args.batch_size == 1,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation = DataLoader(CachedImages(images, labels, splits['validation']), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss, history = float('inf'), []
    path = args.model_dir / f'{name}.pth'
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0., 0
        for batch, targets in train:
            batch, targets = batch.to(args.device), targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            count += len(targets)
        metrics = evaluate(model, validation, len(classes))
        history.append({'epoch': epoch, 'train_loss': loss_sum / count, 'validation': metrics})
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'architecture': name, 'model_state_dict': cpu_state_dict(model),
                'device': next(model.parameters()).device.type,
                'class_names': classes, 'image_size': args.image_size,
                'split_id': split_id, 'initialization': 'random (no pretrained weights)',
                'best_epoch': epoch, 'validation': metrics,
            }, path)
        print(f"{name} {epoch:02d}/{args.epochs} | train={loss_sum/count:.4f} "
              f"val={metrics['loss']:.4f} acc={metrics['accuracy']:.3f}", flush=True)
        write_json(args.result_dir / f'{name}_history.json', history)
    write_json(args.result_dir / f'{name}_training.json', {
        'seconds': time.perf_counter() - started, 'epochs': args.epochs,
        'optimizer': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': 1e-4,
        'batch_size': args.batch_size, 'seed': args.seed,
        'device': args.device.type,
    })


def draw_grid(indices, images, paths, labels, classes, names, records, maps, destination):
    fig, axes = plt.subplots(len(indices), len(names) + 1,
                             figsize=(3.3 * (len(names) + 1), 2.9 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        axes[row, 0].imshow(images[index])
        axes[row, 0].set_title(f'{paths[index].name}\nTrue: {classes[labels[index]]}', fontsize=9)
        for col, name in enumerate(names, 1):
            record, heatmap = records[name][index], maps[name][index]
            ax = axes[row, col]
            ax.imshow(images[index])
            ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1, alpha=.55 * heatmap)
            ax.set_title(
                f"{name}\nPred: {record['predicted_class']} ({record['confidence']:.0%})\n"
                f"CAM: {record['target_class']} | {record['native_resolution']}"
                + ('\nNo positive evidence' if record['empty_heatmap'] else ''),
                fontsize=8, color='black' if record['correct'] else 'crimson',
            )
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Held-out Grad-CAM | same target class across models | red titles = wrong prediction\n'
                 'Random initialization; per-image relative heat; coarse localization, not lesion boundaries', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - .7 / (2.9 * len(indices))))
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def render_comparison(images, paths, labels, classes, splits, split_id, args):
    selected = select_examples(splits['test'], labels, args.samples_per_class)
    overview = select_examples(selected, labels, 3)
    records, maps, model_metrics = {}, {}, {}
    training_configs = {name: validate_training_config(name, args.result_dir) for name in args.models}
    missing = [name for name in args.models if not (args.model_dir / f'{name}.pth').is_file()]
    if missing:
        raise ValueError(f'Missing checkpoints: {missing}. Run src/compare_models.py --models '
                         + ' '.join(missing) + ' to train, using the same experiment settings.')
    test = DataLoader(CachedImages(images, labels, splits['test']), batch_size=args.batch_size)
    preprocess = CachedImages(images, labels, selected).transform
    for name in args.models:
        checkpoint_path = args.model_dir / f'{name}.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if (checkpoint['split_id'] != split_id or checkpoint['class_names'] != classes
                or checkpoint['image_size'] != args.image_size or checkpoint['architecture'] != name):
            raise ValueError(f'Checkpoint does not match this experiment: {checkpoint_path}')
        model, layer, reshape, layer_name = build_model(name, len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_metrics[name] = {
            'family': FAMILIES[name], 'checkpoint': str(checkpoint_path.resolve()),
            'initialization': checkpoint['initialization'], 'best_epoch': checkpoint['best_epoch'],
            'training': training_configs[name],
            'target_layer': layer_name, 'parameters': sum(p.numel() for p in model.parameters()),
            'test': evaluate(model, test, len(classes)),
        }
        records[name], maps[name] = {}, {}
        spatial = []

        def capture_shape(_module, _inputs, output):
            features = reshape(output) if reshape else output
            spatial.append(list(features.shape[-2:]))

        handle = layer.register_forward_hook(capture_shape)
        try:
            cam = GradCAM(model, layer, reshape_transform=reshape)
            for index in selected:
                target = labels[index] if args.target == 'true' else classes.index(args.target)
                heatmap, probabilities, target_index = cam(preprocess(images[index]).unsqueeze(0), target)
                predicted = int(probabilities.argmax())
                records[name][index] = {
                    'image': str(paths[index].relative_to(args.data_dir).as_posix()),
                    'true_class': classes[labels[index]], 'target_class': classes[target_index],
                    'predicted_class': classes[predicted], 'confidence': float(probabilities[predicted]),
                    'probabilities': dict(zip(classes, map(float, probabilities))),
                    'correct': predicted == labels[index], 'empty_heatmap': bool(heatmap.max() == 0),
                    'native_resolution': 'x'.join(map(str, spatial[-1])),
                }
                maps[name][index] = heatmap
        finally:
            handle.remove()
        print(f'Grad-CAM: {name}, {len(selected)} test images', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_grid(overview, images, paths, labels, classes, args.models, records, maps, args.output_dir / 'overview.png')
    figures = []
    for number, index in enumerate(selected, 1):
        filename = f'{number:03d}_{paths[index].stem}.png'
        draw_grid([index], images, paths, labels, classes, args.models, records, maps, args.output_dir / filename)
        figures.append({'image': records[args.models[0]][index]['image'], 'figure': filename})
    for label, class_name in enumerate(classes):
        indices = [i for i in selected if labels[i] == label]
        if indices:
            filename = f'class_{label}.png'
            draw_grid(indices, images, paths, labels, classes, args.models, records, maps, args.output_dir / filename)
    np.savez_compressed(args.output_dir / 'heatmaps.npz', **{
        f'{name}_{number:03d}': maps[name][index]
        for name in args.models for number, index in enumerate(selected, 1)
    })
    summary = {
        'selection': 'held_out_test_only', 'split_id': split_id, 'seed': args.seed,
        'image_size': args.image_size, 'target': args.target,
        'interpretation': 'Each CAM normalized independently; no lesion annotations or localization ranking. '
                          'Architecture and training effects are confounded in this small single-seed experiment.',
        'models': model_metrics,
        'images': [dict(figures[n], models={name: records[name][i] for name in args.models})
                   for n, i in enumerate(selected)],
    }
    write_json(args.output_dir / 'summary.json', summary)
    write_json(args.result_dir / 'metrics.json', model_metrics)
    table = ''.join(
        f'<tr><td>{html.escape(name)}</td><td>{html.escape(m["family"])}</td>'
        f'<td>{m["best_epoch"]}</td><td>{m["test"]["accuracy"]:.1%}</td>'
        f'<td>{m["test"]["macro_f1"]:.3f}</td></tr>' for name, m in model_metrics.items()
    )
    gallery = ''.join(
        f'<figure><a href="{item["figure"]}"><img loading="lazy" src="{item["figure"]}"></a>'
        f'<figcaption>{html.escape(item["image"])}</figcaption></figure>' for item in figures
    )
    (args.output_dir / 'index.html').write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>病斑 Grad-CAM 比較</title>'
        '<style>body{font:16px system-ui;margin:32px;background:#f5f5f1;color:#17231e}'
        'img{width:100%;height:auto}figure{margin:24px 0;background:white;padding:12px}'
        'table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}'
        'p{max-width:1000px;line-height:1.7}</style><h1>水稻病斑 Grad-CAM 視覺比較</h1>'
        '<p>所有模型從隨機權重訓練，使用相同的訓練／驗證／測試切分。'
        '最佳 epoch 由驗證 loss 決定；下方只顯示測試圖片。'
        f'本次解釋目標：{html.escape(args.target)}（true 代表真實類別）。</p>'
        '<p>左欄為輸入原圖，其餘為各模型的熱區疊圖。紅字表示分類錯誤；'
        'No positive evidence 表示沒有正向熱區。每個熱區各自正規化，顏色不能比較絕對強度。'
        '原生熱區解析度標示於各欄；放大後不是病斑邊界。'
        f'目前沒有病斑位置標註，無法據此量化排名。測試集共 {len(splits["test"])} 張，結果為探索性比較。</p>'
        '<table><tr><th>模型</th><th>架構</th><th>最佳 epoch</th><th>Test accuracy</th>'
        f'<th>Test Macro-F1</th></tr>{table}</table><h2>總覽（每類最多 3 張）</h2>'
        f'<a href="overview.png"><img src="overview.png"></a><h2>逐張比較</h2>{gallery}</html>',
        encoding='utf-8',
    )
    print(f'Output: {args.output_dir.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/rice_leaf_diseases')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--samples-per-class', type=int, default=0, help='0: all held-out test images')
    parser.add_argument('--target', default='true', help='true or one exact class name (same across all models)')
    parser.add_argument('--render-only', action='store_true', help='Load this experiment checkpoints without training')
    args = parser.parse_args()
    try:
        args.device = training_device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    suffix = '_xpu' if args.device.type == 'xpu' else ''
    args.model_dir = args.model_dir or ROOT / f'models/comparison{suffix}'
    args.result_dir = args.result_dir or ROOT / f'results/comparison{suffix}'
    args.output_dir = args.output_dir or ROOT / f'figures/model_comparison{suffix}'
    print(f'Training device: {args.device}', flush=True)
    if args.device.type == 'xpu':
        print(f'GPU: {torch.xpu.get_device_name(args.device)}', flush=True)
    if (args.epochs < 1 or args.image_size < 32 or args.batch_size < 2 or args.threads < 1
            or args.learning_rate <= 0 or args.samples_per_class < 0):
        parser.error('Require epochs/threads >= 1, image-size >= 32, batch-size >= 2, lr > 0, samples >= 0.')
    if len(set(args.models)) != len(args.models):
        parser.error('Model names must be unique.')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.data_dir = args.data_dir.resolve()
    if not args.data_dir.is_dir():
        parser.error(f'Dataset directory not found: {args.data_dir}')
    classes = sorted(p.name for p in args.data_dir.iterdir() if p.is_dir())
    paths, labels = [], []
    for label, name in enumerate(classes):
        candidates = sorted(p for p in (args.data_dir / name).rglob('*')
                            if p.is_file() and p.suffix.lower() in EXTENSIONS)
        paths.extend(candidates)
        labels.extend([label] * len(candidates))
    if args.target != 'true' and args.target not in classes:
        parser.error(f'--target must be true or one of {classes}')
    try:
        splits = make_splits(labels, args.seed)
    except ValueError as error:
        parser.error(f'Cannot stratify dataset: {error}')
    manifest = {
        'seed': args.seed, 'class_names': classes, 'image_size': args.image_size,
        'images': [{'path': p.relative_to(args.data_dir).as_posix(), 'label': label,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p, label in zip(paths, labels)],
        'splits': splits,
    }
    split_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['split_id'] = split_id
    manifest_path = args.result_dir / 'split.json'
    if args.render_only:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text('utf-8')) != manifest:
            parser.error('Saved split differs from dataset/seed/image-size; use the original experiment settings.')
    else:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        write_json(args.result_dir / 'experiment.json', {
            'models': args.models, 'epochs': args.epochs, 'seed': args.seed, 'image_size': args.image_size,
            'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'threads': args.threads,
            'device': args.device.type,
            'initialization': 'random (no pretrained weights)',
            'torch': str(torch.__version__), 'torchvision': __import__('torchvision').__version__,
            'selection': 'minimum validation cross entropy', 'split_id': split_id,
        })
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert('RGB').resize((args.image_size, args.image_size), Image.Resampling.BILINEAR))
    print(f'Dataset: {len(paths)} | ' + ' | '.join(f'{k}: {len(v)}' for k, v in splits.items()), flush=True)
    if not args.render_only:
        for name in args.models:
            train_one(name, images, labels, classes, splits, split_id, args)
    try:
        render_comparison(images, paths, labels, classes, splits, split_id, args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
