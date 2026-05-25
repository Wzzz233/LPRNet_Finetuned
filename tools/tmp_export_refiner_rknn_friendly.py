#!/usr/bin/env python3
import torch
from torch import nn
from collections import OrderedDict

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class RefinerRKNNFriendly(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        fd=64
        self.lat1 = nn.Conv2d(64, fd, 1)
        self.lat2 = nn.Conv2d(128, fd, 1)
        self.lat3 = nn.Conv2d(256, fd, 1)
        self.lat4 = nn.Conv2d(512, fd, 1)
        self.smooth3 = ConvBNReLU(fd, fd)
        self.smooth2 = ConvBNReLU(fd, fd)
        self.smooth1 = ConvBNReLU(fd, fd)
        self.heatmap_head = nn.Sequential(ConvBNReLU(fd, fd), nn.Conv2d(fd, 4, 1))
        self.mask_head = nn.Sequential(ConvBNReLU(fd, fd), nn.Conv2d(fd, 1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x):
        c1 = self.layer1(self.stem(x))     # 32x64
        c2 = self.layer2(c1)               # 16x32
        c3 = self.layer3(c2)               # 8x16
        c4 = self.layer4(c3)               # 4x8
        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + self.up(p4))
        p2 = self.smooth2(self.lat2(c2) + self.up(p3))
        p1 = self.smooth1(self.lat1(c1) + self.up(p2))
        return self.heatmap_head(p1), self.mask_head(p1)

ckpt='/home/wzzz/LPRNet/runs/quad_refiner/stage1_r18_gt_20260408_102202/exp/best.pt'
out='/home/wzzz/VHDL_Project/ARM/stage1_r18_gt_best_rknnfriendly_op12.onnx'
model=RefinerRKNNFriendly()
sd=torch.load(ckpt,map_location='cpu',weights_only=False)
sd=sd.get('state_dict',sd)
nsd=OrderedDict((k.replace('model.','',1) if k.startswith('model.') else k,v) for k,v in sd.items())
model.load_state_dict(nsd, strict=True)
model.eval()
dummy=torch.randn(1,3,128,256)
with torch.no_grad():
    torch.onnx.export(model, dummy, out, input_names=['image'], output_names=['heatmaps','mask'], opset_version=12, do_constant_folding=True)
print(out)
