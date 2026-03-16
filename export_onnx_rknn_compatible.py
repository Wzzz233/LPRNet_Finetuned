#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data.load_data import CHARS
from model.LPRNet import small_basic_block


class maxpool_3d(nn.Module):
    # Export-friendly replacement of MaxPool3d used by original LPRNet.
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


class LPRNetExport(nn.Module):
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
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
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
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RKNN-compatible ONNX from LPRNet PyTorch weights.")
    parser.add_argument("--weights", default="./weights_red_stage3/Final_LPRNet_model.pth", help="Path to .pth weights.")
    parser.add_argument("--output", default="./LPRNet_stage3_rknn_compatible.onnx", help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device("cpu")
    model = LPRNetExport(class_num=len(CHARS), dropout_rate=0).to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 24, 94, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"[OK] ONNX exported: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
