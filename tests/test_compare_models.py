import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import compare_models
from src.compare_models import build_model, make_splits, select_examples, channels_first
from src.grad_cam import GradCAM


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_is_reproducible_disjoint_and_covers_dataset(self):
        labels = [label for label in range(3) for _ in range(40)]
        first = make_splits(labels, 42)
        self.assertEqual(first, make_splits(labels, 42))
        self.assertEqual([len(first[k]) for k in ('train', 'validation', 'test')], [84, 18, 18])
        flattened = [i for values in first.values() for i in values]
        self.assertEqual(sorted(flattened), list(range(120)))
        for values in first.values():
            self.assertEqual(set(labels[i] for i in values), {0, 1, 2})

    def test_examples_only_come_from_test_set_and_zero_selects_all(self):
        labels = [i // 40 for i in range(120)]
        test = make_splits(labels, 42)['test']
        selected = select_examples(test, labels, 2)
        self.assertEqual(len(selected), 6)
        self.assertTrue(set(selected) <= set(test))
        self.assertEqual(set(select_examples(test, labels, 0)), set(test))
        with self.assertRaises(ValueError):
            select_examples(test, labels, -1)

    def test_real_architectures_produce_class_specific_finite_maps(self):
        for name in ('rice_cnn', 'resnet18', 'mobilenet_v3_small', 'swin_t'):
            with self.subTest(name=name):
                model, layer, reshape, layer_name = build_model(name, 3)
                model.eval()
                image = torch.randn(1, 3, 128, 128)
                before = len(layer._forward_hooks)
                heatmap, probabilities, target = GradCAM(
                    model, layer, reshape_transform=reshape
                )(image, target_class=1)
                self.assertEqual(heatmap.shape, (128, 128))
                self.assertTrue(np.isfinite(heatmap).all())
                self.assertGreaterEqual(float(heatmap.min()), 0)
                self.assertLessEqual(float(heatmap.max()), 1.000001)
                self.assertAlmostEqual(float(probabilities.sum()), 1, places=5)
                self.assertEqual(target, 1)
                self.assertEqual(len(layer._forward_hooks), before)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(layer_name)

    def test_mixed_training_budgets_are_rejected(self):
        config = {'models': ['rice_cnn'], 'epochs': 30, 'seed': 42,
                  'batch_size': 12, 'learning_rate': .0003}
        training = {k: v for k, v in config.items() if k != 'models'}
        training.update(optimizer='AdamW', weight_decay=.0001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experiment.json').write_text(json.dumps(config))
            (root / 'rice_cnn_training.json').write_text(json.dumps(training))
            self.assertTrue(hasattr(compare_models, 'validate_training_config'),
                            'Training configuration must be validated before comparing checkpoints')
            self.assertEqual(compare_models.validate_training_config('rice_cnn', root), training)
            config['epochs'] = 2
            (root / 'experiment.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'epochs'):
                compare_models.validate_training_config('rice_cnn', root)
            with self.assertRaisesRegex(ValueError, 'resnet18'):
                compare_models.validate_training_config('resnet18', root)

    def test_nhwc_heatmaps_preserve_known_class_specific_regions(self):
        class ToNHWC(torch.nn.Module):
            def forward(self, tensor):
                return tensor.permute(0, 2, 3, 1)

        class PoolNHWC(torch.nn.Module):
            def forward(self, tensor):
                return tensor.mean((1, 2))

        conv = torch.nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0, 0, 0] = 1
            conv.weight[1, 1, 0, 0] = 1
        layer = ToNHWC()
        model = torch.nn.Sequential(conv, layer, PoolNHWC()).eval()
        image = torch.zeros(1, 3, 4, 6)
        image[0, 0, :2, :3] = 2
        image[0, 1, 2:, 3:] = 3
        red, _, _ = GradCAM(model, layer, channels_first)(image, 0)
        green, _, _ = GradCAM(model, layer, channels_first)(image, 1)
        expected_red = np.zeros((4, 6), dtype=np.float32)
        expected_red[:2, :3] = 1
        expected_green = np.zeros((4, 6), dtype=np.float32)
        expected_green[2:, 3:] = 1
        np.testing.assert_allclose(red, expected_red, atol=1e-6)
        np.testing.assert_allclose(green, expected_green, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
