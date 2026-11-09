"""Multi-task networks: a shared pretrained encoder with a classification head
and a segmentation head (simple 1x1 head or a U-Net style decoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

BACKBONES = ("vgg16", "mobilenet_v2", "efficientnet_b0")
ARCHS = ("multitask", "unet")

# Slices of ``model.features`` ending at strides 2, 4, 8, 16, 32.
_STAGE_SLICES = {
    "vgg16": [(0, 5), (5, 10), (10, 17), (17, 24), (24, None)],
    "mobilenet_v2": [(0, 2), (2, 4), (4, 7), (7, 14), (14, None)],
    "efficientnet_b0": [(0, 2), (2, 3), (3, 4), (4, 6), (6, None)],
}


def _features(backbone: str, pretrained: bool) -> nn.Sequential:
    if backbone == "vgg16":
        return tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features
    if backbone == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return tvm.mobilenet_v2(weights=weights).features
    if backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return tvm.efficientnet_b0(weights=weights).features
    raise ValueError(f"Unsupported backbone {backbone!r}; choose from {BACKBONES}")


class Encoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        feats = _features(backbone, pretrained)
        self.stages = nn.ModuleList(feats[a:b] for a, b in _STAGE_SLICES[backbone])
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outs = self(torch.zeros(1, 3, 64, 64))
        self.train(was_training)
        self.channels = [o.shape[1] for o in outs]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


class _ClassifierHead(nn.Sequential):
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.2):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )


class MultiTaskNet(nn.Module):
    """Classification head + 1x1 segmentation head on the deepest feature map."""

    def __init__(self, num_cls_classes: int, num_seg_classes: int = 2,
                 backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.encoder = Encoder(backbone, pretrained)
        deep = self.encoder.channels[-1]
        self.classifier = _ClassifierHead(deep, num_cls_classes)
        self.segmentation = nn.Conv2d(deep, num_seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        deep = self.encoder(x)[-1]
        seg = F.interpolate(self.segmentation(deep), size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg, self.classifier(deep)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiTaskUNet(nn.Module):
    """Classification head + U-Net style decoder using encoder skip connections."""

    def __init__(self, num_cls_classes: int, num_seg_classes: int = 2,
                 backbone: str = "efficientnet_b0", pretrained: bool = True,
                 decoder_channels: tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()
        self.encoder = Encoder(backbone, pretrained)
        ch = self.encoder.channels  # shallow -> deep
        self.classifier = _ClassifierHead(ch[-1], num_cls_classes)
        blocks, in_ch = [], ch[-1]
        for skip_ch, out_ch in zip(reversed(ch[:-1]), decoder_channels):
            blocks.append(_UpBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.seg_head = nn.Conv2d(in_ch, num_seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.decoder, reversed(feats[:-1])):
            y = block(y, skip)
        seg = F.interpolate(self.seg_head(y), size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg, self.classifier(feats[-1])


def build_model(arch: str, backbone: str, num_cls_classes: int,
                num_seg_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if arch == "multitask":
        return MultiTaskNet(num_cls_classes, num_seg_classes, backbone, pretrained)
    if arch == "unet":
        return MultiTaskUNet(num_cls_classes, num_seg_classes, backbone, pretrained)
    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHS}")
