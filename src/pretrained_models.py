"""The nine models in models.md; ImageNet backbones with new task heads."""

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock

if __package__:
    from .model import RiceCNN
else:
    from model import RiceCNN


MODEL_NAMES = ('rice_cnn', 'resnet18', 'densenet121', 'efficientnet_b0',
               'mobilenet_v3_small', 'inception_v3', 'se_resnet18',
               'cbam_resnet18', 'convnext_tiny')
DISPLAY_NAMES = dict(zip(MODEL_NAMES, (
    'Simple CNN', 'ResNet18', 'DenseNet121', 'EfficientNet-B0', 'MobileNetV3-Small',
    'InceptionV3', 'SE-ResNet18', 'CBAM-ResNet18', 'ConvNeXt-Tiny')))


def weights_for(name):
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}')
    if name == 'rice_cnn':
        return None
    backbone = 'resnet18' if name in ('se_resnet18', 'cbam_resnet18') else name
    return models.get_model_weights(backbone).DEFAULT


def weight_name(name):
    weights = weights_for(name)
    return str(weights) if weights is not None else 'random (no pretrained weights)'


class ChannelAttention(nn.Module):
    """SE uses average pooling; CBAM shares its MLP across average/max pools."""

    def __init__(self, channels, use_max=False, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.use_max = use_max
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                 nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        gate = self.mlp(x.mean((2, 3), keepdim=True))
        if self.use_max:
            gate = gate + self.mlp(x.amax((2, 3), keepdim=True))
        return x * gate.sigmoid()


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels, use_max=True)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        x = self.channel(x)
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * self.spatial(pooled).sigmoid()


class AttentionBasicBlock(BasicBlock):
    """ResNet18 block with attention before residual addition, retaining weight keys."""

    def __init__(self, block, attention):
        nn.Module.__init__(self)
        self.conv1, self.bn1, self.relu = block.conv1, block.bn1, block.relu
        self.conv2, self.bn2 = block.conv2, block.bn2
        self.downsample, self.stride = block.downsample, block.stride
        channels = self.conv2.out_channels
        self.attention = ChannelAttention(channels) if attention == 'se' else CBAM(channels)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        return self.relu(out + identity)


def build_model(name, num_classes, pretrained=True):
    """Load the full 1000-class backbone first, then replace classification heads.

    pretrained=False is for offline checks/checkpoint reconstruction, not a fallback
    for failed downloads. Download errors intentionally propagate to the caller.
    """
    weights = weights_for(name) if pretrained else None
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}')
    if name == 'rice_cnn':
        return RiceCNN(num_classes)
    backbone = 'resnet18' if name in ('se_resnet18', 'cbam_resnet18') else name
    kwargs = {'init_weights': False} if backbone == 'inception_v3' else {}
    model = models.get_model(backbone, weights=weights, **kwargs)
    if backbone == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if name in ('se_resnet18', 'cbam_resnet18'):
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
                for i, block in enumerate(layer):
                    layer[i] = AttentionBasicBlock(block, 'se' if name == 'se_resnet18' else 'cbam')
    elif name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'inception_v3':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        # torchvision turns this on when pretrained weights are supplied; retain it
        # when rebuilding the architecture without downloading weights as well.
        model.transform_input = True
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def preprocessing(name, training=False):
    weights = weights_for(name)
    if weights is None:
        weights = models.ResNet18_Weights.DEFAULT
    preset = weights.transforms()
    if not training:
        return preset
    return transforms.Compose([
        transforms.RandomResizedCrop(preset.crop_size[0], scale=(.8, 1.),
                                     interpolation=preset.interpolation, antialias=True),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15),
        transforms.ToTensor(), transforms.Normalize(preset.mean, preset.std),
    ])


def classification_loss(output, labels):
    if hasattr(output, 'logits'):
        loss = nn.functional.cross_entropy(output.logits, labels)
        if output.aux_logits is not None:
            loss = loss + .4 * nn.functional.cross_entropy(output.aux_logits, labels)
        return loss
    return nn.functional.cross_entropy(output, labels)
