import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from src.compare_models import train_one


class XPUTrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.xpu.is_available(), 'Intel XPU required')
    def test_training_on_xpu_saves_portable_checkpoint(self):
        images = [Image.fromarray(np.full((32, 32, 3), i * 20, dtype=np.uint8))
                  for i in range(8)]
        labels = [0, 1] * 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(device=torch.device('xpu'), seed=42, batch_size=2,
                                   learning_rate=.001, epochs=1, image_size=32,
                                   model_dir=root, result_dir=root)
            train_one('rice_cnn', images, labels, ['a', 'b'],
                      {'train': list(range(6)), 'validation': [6, 7]}, 'test', args)
            checkpoint = torch.load(root / 'rice_cnn.pth', weights_only=True)
            self.assertEqual(checkpoint['device'], 'xpu')
            self.assertTrue(all(t.device.type == 'cpu'
                                for t in checkpoint['model_state_dict'].values()))
            self.assertTrue(np.isfinite(checkpoint['validation']['loss']))


if __name__ == '__main__':
    unittest.main()
