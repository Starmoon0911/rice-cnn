"""Class-specific Grad-CAM for the saved RiceCNN, without retraining."""

import argparse
import json
import math
from pathlib import Path
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms

if __package__:
    from .model import RiceCNN
else:
    from model import RiceCNN


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


class GradCAM:
    """Explain one image's class logit; call with a model in eval mode.

    Returns (normalized input-sized heatmap, class probabilities, target index).
    Hooks are removed even on failure; parameter gradients are left untouched.
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

    def __call__(self, image, target_class=None):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('GradCAM requires one image with shape [1, C, H, W].')
        activations = []

        def capture(_module, _inputs, output):
            activations.append(output)

        handle = self.target_layer.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                logits = self.model(image.detach().clone().requires_grad_(True))
                predicted = int(logits.argmax(dim=1).item())
                target = predicted if target_class is None else target_class
                if not 0 <= target < logits.shape[1]:
                    raise ValueError(f'Target class must be in [0, {logits.shape[1] - 1}].')
                if len(activations) != 1:
                    raise ValueError('Target layer must run exactly once per forward pass.')
                features = activations[0]
                gradients = torch.autograd.grad(logits[0, target], features)[0]
                if self.reshape_transform is not None:
                    features = self.reshape_transform(features)
                    gradients = self.reshape_transform(gradients)
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                heatmap = (weights * features).sum(dim=1, keepdim=True).relu()
                heatmap = F.interpolate(
                    heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False
                )[0, 0]
                maximum = heatmap.max()
                if maximum > 0:
                    heatmap = heatmap / maximum
                probabilities = logits.softmax(dim=1)[0].detach().cpu().numpy()
                return heatmap.detach().cpu().numpy(), probabilities, int(target)
        finally:
            handle.remove()


def save_figure(image, heatmap, record, destination):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(image)
    axes[0].set_title(f'Model input ({image.width} x {image.height})')
    color = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM: {record['target_class']}")
    axes[2].imshow(image)
    # A zero map contributes no color, rather than tinting the whole image blue.
    axes[2].imshow(heatmap, cmap='jet', vmin=0, vmax=1, alpha=0.55 * heatmap)
    axes[2].set_title('Overlay')
    for ax in axes:
        ax.axis('off')
    fig.colorbar(color, ax=axes[1], fraction=0.046, pad=0.04, label='Relative evidence')
    fig.suptitle(
        f"{Path(record['image']).name} | True: {record['true_class'] or 'unknown'}\n"
        f"Predicted: {record['predicted_class']} ({record['confidence']:.1%})"
    )
    if record['empty_heatmap']:
        fig.text(0.5, 0.02, 'No positive Grad-CAM evidence for this class.', ha='center')
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_overview(previews, destination):
    columns = min(3, len(previews))
    rows = math.ceil(len(previews) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (image, heatmap, record) in zip(axes.flat, previews):
        ax.imshow(image)
        ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1, alpha=0.55 * heatmap)
        mismatch = record['true_class'] and record['true_class'] != record['predicted_class']
        ax.set_title(
            f"{Path(record['image']).name}\nTrue: {record['true_class'] or 'unknown'}\n"
            f"Pred: {record['predicted_class']} ({record['confidence']:.1%})\n"
            f"CAM: {record['target_class']}"
            + ('\nNo positive CAM evidence' if record['empty_heatmap'] else ''),
            fontsize=9, color='crimson' if mismatch else 'black',
        )
    fig.suptitle('Grad-CAM review (red titles = incorrect predictions)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/best_rice_cnn_cpu.pth')
    parser.add_argument('--image', type=Path, help='Explain a single image instead of dataset samples.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/rice_leaf_diseases')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures/grad_cam')
    parser.add_argument('--samples-per-class', type=int, default=3, help='Default: 3; 0 selects all images.')
    parser.add_argument('--target-class', help='Exact class name; default: explain the predicted class.')
    args = parser.parse_args()
    if args.samples_per_class < 0:
        parser.error('--samples-per-class must be nonnegative.')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')

    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    class_names = checkpoint['class_names']
    image_size = checkpoint['image_size']
    if args.target_class is not None and args.target_class not in class_names:
        parser.error(f'--target-class must be one of: {class_names}')
    target = class_names.index(args.target_class) if args.target_class is not None else None
    model = RiceCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # The last Conv2d, before BatchNorm/ReLU and global average pooling.
    cam = GradCAM(model, model.features[12])
    resize = transforms.Resize((image_size, image_size))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image:
        if not args.image.is_file():
            parser.error(f'Image not found: {args.image}')
        paths = [args.image]
    else:
        paths = []
        rng = random.Random(42)
        for name in class_names:
            candidates = sorted(
                p for p in (args.data_dir / name).rglob('*')
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )
            if args.samples_per_class and len(candidates) > args.samples_per_class:
                candidates = sorted(rng.sample(candidates, args.samples_per_class))
            paths.extend(candidates)
    if not paths:
        parser.error(f'No images found in {args.data_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, previews = [], []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            image = resize(source.convert('RGB'))
        heatmap, probabilities, target_index = cam(preprocess(image).unsqueeze(0), target)
        predicted = int(probabilities.argmax())
        true_class = path.parent.name if path.parent.name in class_names else None
        filename = f'{index:03d}_{path.parent.name}_{path.stem}.png'
        record = {
            'image': str(path.resolve()),
            'true_class': true_class,
            'true_class_source': 'parent_folder' if true_class else None,
            'predicted_class': class_names[predicted],
            'confidence': float(probabilities[predicted]),
            'target_class': class_names[target_index],
            'probabilities': dict(zip(class_names, map(float, probabilities))),
            'empty_heatmap': bool(heatmap.max() == 0),
            'figure': filename,
        }
        save_figure(image, heatmap, record, args.output_dir / filename)
        records.append(record)
        previews.append((image, heatmap, record))
        print(f"[{index}/{len(paths)}] {path.name}: {record['predicted_class']} ({record['confidence']:.1%})")
    save_overview(previews, args.output_dir / 'overview.png')
    summary = {
        'checkpoint': str(args.checkpoint.resolve()),
        'image_size': image_size,
        'target_layer': 'features.12 (last Conv2d)',
        'selection': 'single_image' if args.image else 'dataset_samples_not_held_out',
        'seed': 42,
        'samples_per_class': args.samples_per_class if not args.image else None,
        'interpretation': 'Relative positive evidence per image, not lesion segmentation or proof of correctness.',
        'images': records,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Output: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
