import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import models


class PretrainedModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_nine_models_produce_three_logits(self):
        from src.pretrained_models import MODEL_NAMES, build_model, preprocessing
        self.assertEqual(len(MODEL_NAMES), 9)
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, 3, pretrained=False).eval()
                transform = preprocessing(name)
                batch = transform(Image.new('RGB', (360, 320))).unsqueeze(0)
                self.assertEqual(tuple(batch.shape[-2:]),
                                 (299, 299) if name == 'inception_v3' else (224, 224))
                with torch.no_grad():
                    output = model(batch)
                self.assertEqual(tuple(output.shape), (1, 3))
                self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.xpu.is_available(), 'Intel XPU required')
    def test_efficientnet_pooled_convolution_gradients_match_cpu_on_xpu(self):
        from src.pretrained_models import build_model
        # The final partial batch of four triggered NaN bias gradients in the
        # XPU Conv2d backward kernel for this real EfficientNet SE layer.
        import copy
        torch.manual_seed(42)
        model = build_model('efficientnet_b0', 3, pretrained=False)
        reference = model.features[6][1].block[2].fc1
        accelerated = copy.deepcopy(reference).to('xpu')
        for batch_size in (2, 4, 8):
            with self.subTest(batch_size=batch_size):
                reference.zero_grad(set_to_none=True)
                accelerated.zero_grad(set_to_none=True)
                inputs = torch.randn(batch_size, reference.in_channels, 1, 1,
                                     requires_grad=True)
                xpu_inputs = inputs.detach().to('xpu').requires_grad_()
                output = reference(inputs)
                xpu_output = accelerated(xpu_inputs)
                gradient = torch.randn_like(output)
                output.backward(gradient)
                xpu_output.backward(gradient.to('xpu'))
                torch.testing.assert_close(xpu_output.cpu(), output, atol=1e-5, rtol=1e-4)
                for expected, actual in ((inputs.grad, xpu_inputs.grad),
                                         (reference.weight.grad, accelerated.weight.grad),
                                         (reference.bias.grad, accelerated.bias.grad)):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)

    def test_attention_preserves_pretrained_backbone_and_receives_gradients(self):
        from src.pretrained_models import build_model
        reference = models.resnet18(weights=None)
        for name in ('resnet18', 'se_resnet18', 'cbam_resnet18'):
            with self.subTest(name=name):
                # Replace only the network download; load the real state_dict.
                with patch.object(models.ResNet18_Weights, 'get_state_dict',
                                  return_value=reference.state_dict()):
                    model = build_model(name, 3)
                torch.testing.assert_close(model.conv1.weight, reference.conv1.weight)
                torch.testing.assert_close(model.layer4[1].conv2.weight,
                                           reference.layer4[1].conv2.weight)
                loss = model(torch.randn(2, 3, 64, 64)).sum()
                loss.backward()
                attention = [p for n, p in model.named_parameters() if 'attention' in n]
                if name != 'resnet18':
                    self.assertTrue(attention)
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in attention))

    def test_inception_auxiliary_head_participates_in_loss(self):
        from src.pretrained_models import build_model, classification_loss
        model = build_model('inception_v3', 3, pretrained=False).train()
        output = model(torch.randn(2, 3, 299, 299))
        self.assertEqual(tuple(output.aux_logits.shape), (2, 3))
        loss = classification_loss(output, torch.tensor([0, 1]))
        expected = (torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1]))
                    + .4 * torch.nn.functional.cross_entropy(output.aux_logits, torch.tensor([0, 1])))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(model.AuxLogits.fc.weight.grad)

    def test_summary_keeps_unmeasured_localization_pending(self):
        from src.compare_pretrained import write_summary
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {'model': 'resnet18', 'split_id': 'same', 'test': {'macro_f1': .75,
                      'accuracy': .8}, 'training_seconds': 12., 'weights': 'ResNet18_Weights.IMAGENET1K_V1'}
            import json
            (root / 'resnet18_metrics.json').write_text(json.dumps(result), encoding='utf-8')
            rows = write_summary(root)
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows['resnet18']['macro_f1'], .75)
            self.assertIsNone(rows['resnet18']['lar'])
            self.assertIsNone(rows['rice_cnn']['macro_f1'])
            self.assertIn('待測', (root / 'comparison.md').read_text('utf-8'))

    def test_experiment_rejects_changed_budget_before_writing(self):
        from src.compare_pretrained import ensure_experiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_experiment(root, {'split_id': 'same'}, {'epochs': 30})
            with self.assertRaisesRegex(ValueError, 'experiment.json differs'):
                ensure_experiment(root, {'split_id': 'same'}, {'epochs': 1})
            with self.assertRaisesRegex(ValueError, 'split.json differs'):
                ensure_experiment(root, {'split_id': 'changed'}, {'epochs': 30})

    def test_summary_rejects_mixed_training_budgets(self):
        import json
        from src.compare_pretrained import write_summary
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, epochs in [('resnet18', 30), ('rice_cnn', 1)]:
                result = {'model': name, 'split_id': 'same', 'experiment': {'epochs': epochs},
                          'test': {'macro_f1': .5, 'accuracy': .5}, 'weights': 'test',
                          'training_seconds': 1.}
                (root / f'{name}_metrics.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'settings'):
                write_summary(root)

    def test_synthetic_training_saves_reloadable_checkpoint_and_test_predictions(self):
        import json
        from types import SimpleNamespace
        from src.compare_pretrained import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for i in range(8):
                path = root / f'{i}.png'
                Image.new('RGB', (40, 40), color=(i * 30, 100, 60)).save(path)
                entries.append({'path': path.name, 'label': i % 2})
            manifest = {'class_names': ['a', 'b'], 'images': entries, 'split_id': 'test',
                        'splits': {'train': [0, 1, 2, 3], 'validation': [4, 5], 'test': [6, 7]}}
            args = SimpleNamespace(seed=42, device=torch.device('cpu'), data_dir=root,
                                   batch_size=2, learning_rate=.001, weight_decay=.0001,
                                   epochs=1, run_dir=root)
            train_one('rice_cnn', args, manifest, {'epochs': 1})
            result = json.loads((root / 'rice_cnn_metrics.json').read_text('utf-8'))
            self.assertEqual([p['path'] for p in result['predictions']], ['6.png', '7.png'])
            self.assertEqual(sum(map(sum, result['test']['confusion_matrix'])), 2)
            self.assertGreater(result['training_seconds'], 0)
            checkpoint = torch.load(root / 'checkpoints/rice_cnn.pth', weights_only=True)
            from src.pretrained_models import build_model
            restored = build_model('rice_cnn', 2, pretrained=False)
            restored.load_state_dict(checkpoint['model_state_dict'])
            self.assertTrue(all(p.device.type == 'cpu' for p in checkpoint['model_state_dict'].values()))


if __name__ == '__main__':
    unittest.main()
