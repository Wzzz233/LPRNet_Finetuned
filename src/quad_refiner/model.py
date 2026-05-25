from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# Compat: older torchvision (rknn_env) doesn't have ResNet18_Weights
try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    ResNet18_Weights = None
    from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class QuadHeatmapRefiner(nn.Module):
    """4-corner heatmap refiner with optional local offset refinement.

    V1: heatmap only (4-channel output)
    V2b: heatmap (4-ch) + offset (8-ch) for sub-pixel refinement
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = 64, enable_offset: bool = False):
        super().__init__()
        self.enable_offset = bool(enable_offset)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            base = resnet18(weights=weights)
        except TypeError:
            # Older torchvision (<0.13) uses pretrained= instead of weights=
            base = resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.lat1 = nn.Conv2d(64, feature_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(128, feature_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(256, feature_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(512, feature_dim, kernel_size=1)

        self.smooth3 = ConvBNReLU(feature_dim, feature_dim)
        self.smooth2 = ConvBNReLU(feature_dim, feature_dim)
        self.smooth1 = ConvBNReLU(feature_dim, feature_dim)

        self.heatmap_head = nn.Sequential(
            ConvBNReLU(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, 4, kernel_size=1),
        )
        self.mask_head = nn.Sequential(
            ConvBNReLU(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )
        if self.enable_offset:
            self.offset_head = nn.Sequential(
                ConvBNReLU(feature_dim, feature_dim),
                nn.Conv2d(feature_dim, 8, kernel_size=1),  # dx,dy for each of 4 corners
            )
        else:
            self.offset_head = None

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))

        heatmaps = self.heatmap_head(p1)
        mask = self.mask_head(p1)
        out = {
            'heatmaps': heatmaps,
            'mask': mask,
        }
        if self.offset_head is not None:
            out['offsets'] = self.offset_head(p1)
        return out
