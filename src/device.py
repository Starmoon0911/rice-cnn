"""Device selection shared by the training entry points."""

import torch


def training_device(name):
    device = torch.device(name)
    if device.type == 'xpu' and not torch.xpu.is_available():
        raise RuntimeError(
            'XPU is unavailable. Install an Intel GPU driver and XPU PyTorch wheels '
            '(https://download.pytorch.org/whl/xpu), or use --device cpu.'
        )
    return device


def cpu_state_dict(model):
    """Keep checkpoints loadable on hosts without an Intel GPU."""
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
