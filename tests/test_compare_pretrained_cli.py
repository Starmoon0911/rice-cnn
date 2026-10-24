import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.compare_pretrained import main


class PretrainedCLITests(unittest.TestCase):
    def run_cli(self, root, *options):
        data = root / 'data'
        if not data.exists():
            for name in ('a', 'b'):
                (data / name).mkdir(parents=True)
                for index in range(20):
                    Image.new('RGB', (32, 32), (index * 10, 100, 60)).save(
                        data / name / f'{index}.png')
        argv = ['compare_pretrained.py', '--device', 'cpu', '--data-dir', str(data),
                '--run-dir', str(root / 'run'), '--epochs', '1', '--threads', '2',
                *options]
        output = io.StringIO()
        with patch('sys.argv', argv), contextlib.redirect_stdout(output):
            main()
        return output.getvalue()

    def test_existing_checkpoint_is_preserved_while_new_model_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'run/checkpoints/resnet18.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'existing checkpoint')
            output = self.run_cli(root, '--models', 'resnet18', 'rice_cnn')
            self.assertIn('Skipping resnet18', output)
            self.assertEqual(checkpoint.read_bytes(), b'existing checkpoint')
            rows = json.loads((root / 'run/comparison.json').read_text('utf-8'))
            self.assertIsNone(rows['resnet18']['macro_f1'])
            self.assertIsNotNone(rows['rice_cnn']['macro_f1'])

    def test_completed_models_are_skipped_and_summary_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_cli(root, '--models', 'rice_cnn')
            paths = [root / 'run/checkpoints/rice_cnn.pth',
                     root / 'run/rice_cnn_metrics.json', root / 'run/rice_cnn_history.json']
            before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
            (root / 'run/comparison.json').unlink()
            output = self.run_cli(root, '--models', 'rice_cnn')
            self.assertIn('Skipping rice_cnn', output)
            self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns)
                                      for path in paths])
            self.assertTrue((root / 'run/comparison.json').exists())

    def test_metrics_only_are_preserved_unless_overwrite_is_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_cli(root, '--models', 'rice_cnn')
            checkpoint = root / 'run/checkpoints/rice_cnn.pth'
            checkpoint.unlink()
            metrics = root / 'run/rice_cnn_metrics.json'
            before = metrics.read_bytes()
            output = self.run_cli(root, '--models', 'rice_cnn')
            self.assertIn('Skipping rice_cnn', output)
            self.assertFalse(checkpoint.exists())
            self.assertEqual(metrics.read_bytes(), before)
            output = self.run_cli(root, '--models', 'rice_cnn', '--overwrite')
            self.assertNotIn('Skipping rice_cnn', output)
            self.assertTrue(checkpoint.exists())


if __name__ == '__main__':
    unittest.main()
