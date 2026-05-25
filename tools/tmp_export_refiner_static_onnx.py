#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/wzzz/LPRNet/src')
import torch
from torch import nn
from collections import OrderedDict

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
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.lat1 = nn.Conv2d(64, feature_dim, 1)
        self.lat2 = nn.Conv2d(128, feature_dim, 1)
        self.lat3 = nn.Conv2d(256, feature_dim, 1)
        self.lat4 = nn.Conv2d(512, feature_dim, 1)
        self.smooth3 = ConvBNReLU(feature_dim, feature_dim)
        self.smooth2 = ConvBNReLU(feature_dim, feature_dim)
        self.smooth1 = ConvBNReLU(feature_dim, feature_dim)
        self.heatmap_head = nn.Sequential(ConvBNReLU(feature_dim, feature_dim), nn.Conv2d(feature_dim, 4, 1))
        self.mask_head = nn.Sequential(ConvBNReLU(feature_dim, feature_dim), nn.Conv2d(feature_dim, 1, 1))
    def forward(self, x):
        import torch.nn.functional as F
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))
        return self.heatmap_head(p1), self.mask_head(p1)

ckpt='/home/wzzz/LPRNet/runs/quad_refiner/stage1_r18_gt_20260408_102202/exp/best.pt'
out='/home/wzzz/VHDL_Project/ARM/stage1_r18_gt_best_static_op13.onnx'
model=QuadHeatmapRefiner(False,64)
sd=torch.load(ckpt,map_location='cpu',weights_only=False)
sd=sd.get('state_dict',sd)
nsd=OrderedDict((k.replace('model.','',1) if k.startswith('model.') else k,v) for k,v in sd.items())
model.load_state_dict(nsd, strict=True)
model.eval()
dummy=torch.randn(1,3,128,256)
with torch.no_grad():
    torch.onnx.export(model, dummy, out, input_names=['image'], output_names=['heatmaps','mask'], opset_version=13, do_constant_folding=True)
print(out)
