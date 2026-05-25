from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .model import QuadHeatmapRefiner


class ExportWrapper(nn.Module):
    """Export wrapper that auto-detects offset head."""
    def __init__(self, model, has_offset: bool = False):
        super().__init__()
        self.model = model
        self.has_offset = has_offset

    def forward(self, x):
        out = self.model(x)
        if self.has_offset and 'offsets' in out:
            return out['heatmaps'], out['mask'], out['offsets']
        return out['heatmaps'], out['mask']


def main(argv=None):
    ap = argparse.ArgumentParser(description='Export quad refiner to ONNX.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--input-width', type=int, default=256)
    ap.add_argument('--input-height', type=int, default=128)
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    has_offset = any('offset_head' in k for k in state_dict.keys())

    model = QuadHeatmapRefiner(pretrained=False, enable_offset=has_offset)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ExportWrapper(model, has_offset=has_offset)
    dummy = torch.randn(1, 3, args.input_height, args.input_width)

    output_names = ['heatmaps', 'mask']
    if has_offset:
        output_names.append('offsets')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=['image'],
        output_names=output_names,
        dynamic_axes={'image': {0: 'batch'}, 'heatmaps': {0: 'batch'}, 'mask': {0: 'batch'}},
        opset_version=args.opset,
    )
    print(f'Exported to {output_path} (offset={has_offset}, outputs={output_names})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
