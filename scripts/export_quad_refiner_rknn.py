#!/usr/bin/env python3
"""
Export V2b quad refiner (with offset head) to ONNX, then convert to RKNN.
Uses RKNN-friendly operators (nn.Upsample instead of F.interpolate).
"""
import argparse, sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


class QuadRefinerRKNNFriendly(nn.Module):
    """ResNet18 + FPN + 4-ch heatmap + 1-ch mask + optional 8-ch offset.
    Uses nn.Upsample for RKNN compatibility."""
    def __init__(self, enable_offset=False):
        super().__init__()
        self.enable_offset = enable_offset
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        fd = 64
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
        if self.enable_offset:
            self.offset_head = nn.Sequential(ConvBNReLU(fd, fd), nn.Conv2d(fd, 8, 1))

    def forward(self, x):
        c1 = self.layer1(self.stem(x))     # 32x64
        c2 = self.layer2(c1)               # 16x32
        c3 = self.layer3(c2)               # 8x16
        c4 = self.layer4(c3)               # 4x8
        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + self.up(p4))
        p2 = self.smooth2(self.lat2(c2) + self.up(p3))
        p1 = self.smooth1(self.lat1(c1) + self.up(p2))
        heatmaps = self.heatmap_head(p1)
        mask = self.mask_head(p1)
        if self.enable_offset:
            offsets = self.offset_head(p1)
            return heatmaps, mask, offsets
        return heatmaps, mask


def main():
    ap = argparse.ArgumentParser(description='Export V2b refiner to ONNX + RKNN.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-dir', default='models/quad_refiner')
    ap.add_argument('--opset', type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load state dict and detect offset
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    has_offset = any('offset_head' in k for k in state_dict.keys())

    # Strip 'model.' prefix if present (from training wrapper checkpoints)
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        key = k.replace('model.', '', 1) if k.startswith('model.') else k
        cleaned[key] = v

    # Build RKNN-friendly model
    model = QuadRefinerRKNNFriendly(enable_offset=has_offset)
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # Export ONNX
    onnx_path = out_dir / 'quad_refiner_v2b_rknn_friendly.onnx'
    dummy = torch.randn(1, 3, 128, 256)
    output_names = ['heatmaps', 'mask']
    if has_offset:
        output_names.append('offsets')

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['image'],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f'ONNX exported: {onnx_path}')
    print(f'  outputs: {output_names}')
    print(f'  offset: {has_offset}')

    # Verify ONNX vs PyTorch
    import onnx
    import onnxruntime
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_sess = onnxruntime.InferenceSession(str(onnx_path))
    x_np = dummy.numpy()
    with torch.no_grad():
        pt_out = model(dummy)
    ort_out = ort_sess.run(output_names, {'image': x_np.astype('float32')})

    for i, name in enumerate(output_names):
        pt_arr = torch.sigmoid(pt_out[i]).numpy() if name == 'heatmaps' else \
                 (pt_out[i] if name == 'offsets' else torch.sigmoid(pt_out[i]).numpy())
        max_diff = float(abs(pt_arr - ort_out[i]).max())
        print(f'  {name}: max_diff={max_diff:.6f} {"PASS" if max_diff < 1e-4 else "FAIL"}')

    # Convert to RKNN
    try:
        from rknn.api import RKNN
        rknn_path = out_dir / 'quad_refiner_v2b_rknn_friendly.rknn'
        rknn = RKNN(verbose=False)
        rknn.config(target_platform='rk3568')
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            print('RKNN load_onnx failed')
            return 1
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print('RKNN build failed')
            return 1
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            print('RKNN export failed')
            return 1
        rknn.release()
        print(f'RKNN exported: {rknn_path}')

        # Verify RKNN vs ONNX
        rknn2 = RKNN(verbose=False)
        rknn2.load_rknn(str(rknn_path))
        rknn2.init_runtime()
        rknn_out = rknn2.inference(inputs=[x_np.astype('float32')])
        rknn2.release()

        for i, name in enumerate(output_names):
            onnx_arr = ort_out[i]
            rknn_arr = rknn_out[i]
            max_diff = float(abs(onnx_arr - rknn_arr).max())
            print(f'  RKNN {name}: max_diff={max_diff:.6f} {"PASS" if max_diff < 1e-3 else "FAIL"}')
    except ImportError:
        print('RKNN toolkit not available in this env, skipping RKNN conversion')
        print('Run with rknn_env: conda run -p /root/miniconda3/envs/rknn_env python ...')

    print('Done')


if __name__ == '__main__':
    main()
