#!/usr/bin/env python3
"""Export stage1 best quad refiner checkpoint to ONNX."""
import sys
sys.path.insert(0, '/home/wzzz/LPRNet/src')

import torch
from torch import nn
from pathlib import Path

# Minimal model definition to avoid loading full quad_refiner package
from collections import OrderedDict
import torchvision

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
    def __init__(self, pretrained=False, feature_dim=64):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
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

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        from torch.nn import functional as F
        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))
        heatmaps = self.heatmap_head(p1)
        mask = self.mask_head(p1)
        return heatmaps, mask

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out[0], out[1]

ckpt_path = '/home/wzzz/LPRNet/runs/quad_refiner/stage1_r18_gt_20260408_102202/exp/best.pt'
out_path = '/home/wzzz/LPRNet/models/quad_refiner/stage1_r18_gt_best.onnx'

model = QuadHeatmapRefiner(pretrained=False, feature_dim=64)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)
# Strip possible 'model.' prefix
new_sd = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace('model.', '', 1) if k.startswith('model.') else k
    new_sd[new_key] = v
model.load_state_dict(new_sd, strict=True)
model.eval()

wrapper = ExportWrapper(model)
dummy = torch.randn(1, 3, 128, 256)

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    wrapper,
    dummy,
    out_path,
    input_names=['image'],
    output_names=['heatmaps', 'mask'],
    dynamic_axes={'image': {0: 'batch'}, 'heatmaps': {0: 'batch'}, 'mask': {0: 'batch'}},
    opset_version=17,
)
print(f'Exported ONNX to {out_path}')