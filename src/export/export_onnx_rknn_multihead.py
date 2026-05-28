#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import CHARS
from LPRNet_multihead import FAMILY_HEADS, small_basic_block


class maxpool_3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        assert len(kernel_size) == 3 and len(stride) == 3
        kernel_size2d_1 = kernel_size[-2:]
        stride2d_1 = stride[-2:]
        kernel_size2d_2 = (kernel_size[0], kernel_size[0])
        stride2d_2 = (kernel_size[0], stride[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size2d_1, stride=stride2d_1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size2d_2, stride=stride2d_2)

    def forward(self, x):
        x = self.maxpool1(x)
        x = x.transpose(1, 3)
        x = self.maxpool2(x)
        x = x.transpose(1, 3)
        return x


class SharedBackboneExport(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            maxpool_3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            maxpool_3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            maxpool_3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )

    def extract_context(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = []
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow.view(f_pow.size(0), -1), dim=1, keepdim=True).view(f_pow.size(0), 1, 1, 1)
            f = torch.div(f, f_mean.clamp_min(1e-12))
            global_context.append(f)
        return torch.cat(global_context, 1)


class LPRNetMultiHeadExport(SharedBackboneExport):
    def __init__(self, class_num, dropout_rate, enhanced_green_head='', pos0_head_cols=0, pos0_num_classes=31):
        super().__init__(class_num=class_num, dropout_rate=dropout_rate)
        self.containers = nn.ModuleDict()
        self.pos0_head_cols = pos0_head_cols
        for family in FAMILY_HEADS:
            if family == 'green8' and enhanced_green_head == 'expE':
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Conv2d(in_channels=256, out_channels=self.class_num, kernel_size=(1, 1)),
                )
            elif family == 'green8' and enhanced_green_head:
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(in_channels=256, out_channels=self.class_num, kernel_size=(1, 1)),
                )
            else:
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
                )

        if pos0_head_cols > 0:
            in_ch = 448 + self.class_num
            self.pos0_head = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, pos0_num_classes),
            )
        else:
            self.pos0_head = None

    def forward(self, x):
        context = self.extract_context(x)
        outs = []
        for family in FAMILY_HEADS:
            head = self.containers[family](context)
            outs.append(torch.mean(head, dim=2))
        if self.pos0_head is not None:
            outs.append(self.pos0_head(context[:, :, :, :self.pos0_head_cols]))
        return tuple(outs)


def detect_green_head_variant(state_dict):
    if any(k.startswith('containers.green8.6.') for k in state_dict.keys()):
        return 'expE'
    if any(k.startswith('containers.green8.3.') for k in state_dict.keys()):
        return 'expD'
    return ''


def detect_pos0_head(state_dict):
    """从 checkpoint 检测 pos0_head 是否存在及其 num_classes。"""
    for k, v in state_dict.items():
        if k == 'pos0_head.4.weight':  # Linear layer weight shape: (num_classes, 128)
            return {'cols': 4, 'num_classes': int(v.shape[0])}
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Export multihead LPRNet weights to RKNN-friendly ONNX.')
    parser.add_argument('--weights', required=True, help='Path to .pth weights.')
    parser.add_argument('--output', required=True, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=12)
    parser.add_argument('--dropout-rate', type=float, default=0.0)
    parser.add_argument('--enhanced-green-head', default='auto', choices=['auto', '', 'expD', 'expE'])
    parser.add_argument('--pos0-head-cols', default='auto', help='pos0 head spatial columns (auto detects from weights; 0 disables)')
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    device = torch.device('cpu')
    state = torch.load(str(weights_path), map_location=device)
    variant = detect_green_head_variant(state) if args.enhanced_green_head == 'auto' else args.enhanced_green_head
    print(f'[Info] detected/using green head variant: {variant or "baseline"}')

    pos0_info = detect_pos0_head(state)
    if args.pos0_head_cols == 'auto':
        pos0_cols = pos0_info['cols'] if pos0_info else 0
        pos0_num_classes = pos0_info['num_classes'] if pos0_info else 31
    else:
        pos0_cols = int(args.pos0_head_cols)
        pos0_num_classes = pos0_info['num_classes'] if pos0_info else 31
    print(f'[Info] pos0_head: cols={pos0_cols} num_classes={pos0_num_classes} ({"detected" if pos0_info else "not found in weights"})')

    net = LPRNetMultiHeadExport(class_num=len(CHARS), dropout_rate=args.dropout_rate, enhanced_green_head=variant,
                                pos0_head_cols=pos0_cols, pos0_num_classes=pos0_num_classes)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[Warn] missing keys:', missing)
        print('[Warn] unexpected keys:', unexpected)
    net.eval().to(device)

    # Wrap model with normalization: uint8 [0,255] → (x-127.5)/128 → [-1,1]
    class NormalizedWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            x = x - 127.5
            x = x / 128.0
            return self.base(x)
    
    net = NormalizedWrapper(net)
    net.eval().to(device)

    output_names = ['normal7', 'green8', 'special']
    if pos0_cols > 0:
        output_names.append('pos0')

    dummy_input = torch.randn(1, 3, 24, 94, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
    )
    print(f'[OK] Multihead ONNX exported: {output_path.resolve()} outputs={output_names}')


if __name__ == '__main__':
    main()
