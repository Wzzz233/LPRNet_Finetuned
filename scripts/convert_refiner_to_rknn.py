#!/usr/bin/env python3
"""Convert pre-exported ONNX refiner model to RKNN. Run in rknn_env."""
import argparse
from pathlib import Path
from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser(description='Convert ONNX refiner to RKNN.')
    ap.add_argument('--onnx', required=True, help='Path to ONNX model')
    ap.add_argument('--output', required=True, help='Output RKNN path')
    ap.add_argument('--target', default='rk3568', help='Target platform')
    ap.add_argument('--dtype', default='fp', choices=['fp', 'int8'], help='Quantization dtype')
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)
    print(f'Configuring for {args.target}...')
    ret = rknn.config(target_platform=args.target)
    if ret != 0:
        print(f'config failed: {ret}')
        return 1

    print(f'Loading ONNX: {onnx_path}...')
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print(f'load_onnx failed: {ret}')
        return 1

    print('Building RKNN...')
    ret = rknn.build(do_quantization=(args.dtype == 'int8'), dataset='')
    if ret != 0:
        print(f'build failed: {ret}')
        return 1

    print(f'Exporting RKNN: {output_path}...')
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        print(f'export_rknn failed: {ret}')
        return 1

    rknn.release()
    print(f'Done: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
