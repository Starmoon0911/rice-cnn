import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.grad_cam import GradCAM


ROOT = Path(__file__).resolve().parents[1]


class GradCAMTests(unittest.TestCase):
    def setUp(self):
        # Each class uses a different color channel, with known spatial evidence.
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0, 0] = 1
            self.conv.weight[1, 1, 0, 0] = 1
            head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        self.model = nn.Sequential(
            self.conv, nn.AdaptiveAvgPool2d(1), nn.Flatten(), head
        ).eval()
        self.image = torch.zeros(1, 3, 4, 4)
        self.image[0, 0, :2, :2] = 2
        self.image[0, 1, 2:, 2:] = 1

    def test_target_class_changes_highlighted_region(self):
        cam = GradCAM(self.model, self.conv)
        red, probabilities, target = cam(self.image)
        green, _, _ = cam(self.image, target_class=1)
        expected_red = np.zeros((4, 4), dtype=np.float32)
        expected_red[:2, :2] = 1
        expected_green = np.zeros((4, 4), dtype=np.float32)
        expected_green[2:, 2:] = 1
        np.testing.assert_allclose(red, expected_red, atol=1e-6)
        np.testing.assert_allclose(green, expected_green, atol=1e-6)
        self.assertEqual(target, 0)
        self.assertAlmostEqual(float(probabilities.sum()), 1., places=6)

    def test_negative_evidence_produces_finite_zero_map(self):
        heatmap, _, _ = GradCAM(self.model, self.conv)(self.image, target_class=2)
        np.testing.assert_array_equal(heatmap, np.zeros((4, 4)))
        self.assertTrue(np.isfinite(heatmap).all())

    def test_repeated_calls_do_not_leave_hooks_or_parameter_gradients(self):
        cam = GradCAM(self.model, self.conv)
        initial_hooks = len(self.conv._forward_hooks)
        for _ in range(3):
            cam(self.image)
        self.assertEqual(len(self.conv._forward_hooks), initial_hooks)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_invalid_target_cleans_up_hook(self):
        initial_hooks = len(self.conv._forward_hooks)
        for target in (-1, 3):
            with self.assertRaises(ValueError):
                GradCAM(self.model, self.conv)(self.image, target_class=target)
        self.assertEqual(len(self.conv._forward_hooks), initial_hooks)

    def test_cli_loads_existing_weights_and_writes_inspectable_outputs(self):
        image = ROOT / 'dataset/rice_leaf_diseases/Leaf smut/DSC_0500.jpg'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'src/grad_cam.py'),
                 '--image', str(image), '--target-class', 'Brown spot',
                 '--output-dir', directory],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((Path(directory) / 'summary.json').read_text('utf-8'))
            self.assertEqual(len(summary['images']), 1)
            record = summary['images'][0]
            self.assertEqual(record['true_class'], 'Leaf smut')
            self.assertEqual(record['target_class'], 'Brown spot')
            self.assertAlmostEqual(sum(record['probabilities'].values()), 1., places=5)
            for filename in (record['figure'], 'overview.png'):
                with Image.open(Path(directory) / filename) as output:
                    output.verify()


if __name__ == '__main__':
    unittest.main()
