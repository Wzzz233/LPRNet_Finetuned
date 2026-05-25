#!/usr/bin/env python3
"""
端到端一致性测试：用同一张真实图片跑 PyTorch / ONNX / RKNN 三路，
对比输出的 heatmaps + offsets 是否一致。

用法：
  # 在主环境跑 PT vs ONNX
  .conda/bin/python scripts/test_refiner_consistency.py \
    --checkpoint runs/quad_refiner/true_quad_refiner_v2b_offset/last.pt \
    --onnx models/quad_refiner/quad_refiner_v2b_last_rknn_friendly.onnx \
    --image <任意含车牌的图片> \
    --output /tmp/refiner_consistency

  # 在 rknn_env 补跑 RKNN 对比
  conda run -p /root/miniconda3/envs/rknn_env python scripts/test_refiner_consistency.py \
    --rknn models/quad_refiner/quad_refiner_v2b_last_rknn_friendly.rknn \
    --image <同一张图片> \
    --output /tmp/refiner_consistency \
    --skip-pt-onnx
"""
import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quad_refiner.model import QuadHeatmapRefiner
from load_data import parse_ccpd_quad_from_name

# 内联需要的几何函数，避免走 __init__.py 的 import chain
def _order_pts(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[s.argmin()]
    br = pts[s.argmax()]
    tr = pts[diff.argmin()]
    bl = pts[diff.argmax()]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_refiner_input(img, coarse_quad, pad_x=0.20, pad_y=0.25, ref_w=256, ref_h=128):
    """从原图裁切 refiner 输入 patch（与板端/训练一致）。"""
    h, w = img.shape[:2]
    q = _order_pts(np.array(coarse_quad))
    x1, y1 = int(q[:, 0].min()), int(q[:, 1].min())
    x2, y2 = int(q[:, 0].max()), int(q[:, 1].max())
    bw, bh = x2 - x1, y2 - y1
    ex = int(round(bw * pad_x))
    ey = int(round(bh * pad_y))
    px1 = max(0, x1 - ex)
    py1 = max(0, y1 - ey)
    px2 = min(w - 1, x2 + ex)
    py2 = min(h - 1, y2 + ey)
    patch = img[py1:py2 + 1, px1:px2 + 1]
    patch = cv2.resize(patch, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    patch_f = patch.astype(np.float32).transpose(2, 0, 1) / 255.0
    return patch_f, (px1, py1, px2, py2)


def main():
    ap = argparse.ArgumentParser(description='Refiner consistency test: PT vs ONNX vs RKNN')
    ap.add_argument('--checkpoint', help='PyTorch checkpoint (.pt)')
    ap.add_argument('--onnx', help='ONNX model path')
    ap.add_argument('--rknn', help='RKNN model path')
    ap.add_argument('--image', required=True, help='Test image path (must contain a visible plate)')
    ap.add_argument('--output', default='/tmp/refiner_consistency', help='Output directory')
    ap.add_argument('--skip-pt-onnx', action='store_true', help='Skip PT/ONNX tests (run RKNN only)')
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test image
    img = cv2.imread(args.image)
    if img is None:
        print(f'FAIL: cannot read image {args.image}')
        return 1
    print(f'Image: {args.image} shape={img.shape}')
    h, w = img.shape[:2]

    # 解析 CCPD 文件名中的真实 quad 作为 coarse quad
    gt_quad = parse_ccpd_quad_from_name(os.path.basename(args.image))
    if gt_quad is not None:
        # 用 GT quad 代替 coarse quad（贴近实际 OBB 检测的场景）
        test_quad = _order_pts(gt_quad).tolist()
        print(f'Using CCPD GT quad as coarse quad')
    else:
        # 非 CCPD 图片：取图像中心区域
        cx, cy = w // 2, h // 2
        pw, ph = 100, 40
        test_quad = [[cx - pw, cy - ph], [cx + pw, cy - ph],
                     [cx + pw, cy + ph], [cx - pw, cy + ph]]
    print(f'Coarse quad: {test_quad}')

    # 生成 refiner 输入
    patch_f, patch_box = compute_refiner_input(img, test_quad)
    print(f'Patch box: {patch_box}, shape={patch_f.shape}')

    # ---------- PyTorch ----------
    pt_results = {}
    if args.checkpoint and not args.skip_pt_onnx:
        import torch
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            # Older torch (rknn_env) doesn't support weights_only
            ckpt = torch.load(args.checkpoint, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        has_offset = any('offset_head' in k for k in sd.keys())
        # Strip model. prefix
        from collections import OrderedDict
        cleaned = OrderedDict()
        for k, v in sd.items():
            key = k.replace('model.', '', 1) if k.startswith('model.') else k
            cleaned[key] = v
        model = QuadHeatmapRefiner(pretrained=False, enable_offset=has_offset)
        model.load_state_dict(cleaned)
        model.eval()
        x = torch.from_numpy(patch_f).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        pt_results['heatmaps'] = torch.sigmoid(out['heatmaps']).numpy()
        pt_results['mask'] = torch.sigmoid(out['mask']).numpy()
        if has_offset and 'offsets' in out:
            pt_results['offsets'] = out['offsets'].numpy()
        print(f'[PT] heatmaps={pt_results["heatmaps"].shape} offsets={pt_results.get("offsets", "N/A")} stats: '
              f'hm=[{pt_results["heatmaps"].min():.4f},{pt_results["heatmaps"].max():.4f}] '
              f'off=[{pt_results.get("offsets", np.zeros(1)).min():.4f},{pt_results.get("offsets", np.zeros(1)).max():.4f}]')
        np.save(str(out_dir / 'pt_heatmaps.npy'), pt_results['heatmaps'])
        np.save(str(out_dir / 'pt_mask.npy'), pt_results['mask'])
        if 'offsets' in pt_results:
            np.save(str(out_dir / 'pt_offsets.npy'), pt_results['offsets'])
        print('[PT] saved')

    # ---------- ONNX ----------
    onnx_results = {}
    if args.onnx and not args.skip_pt_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx)
        x_np = patch_f.astype(np.float32).reshape(1, 3, 128, 256)
        ort_out = sess.run(['heatmaps', 'mask', 'offsets'], {'image': x_np})
        onnx_results['heatmaps'] = 1.0 / (1.0 + np.exp(-ort_out[0]))  # sigmoid
        onnx_results['mask'] = 1.0 / (1.0 + np.exp(-ort_out[1]))
        onnx_results['offsets'] = ort_out[2]
        print(f'[ONNX] heatmaps={onnx_results["heatmaps"].shape} offsets={onnx_results["offsets"].shape} '
              f'hm=[{onnx_results["heatmaps"].min():.4f},{onnx_results["heatmaps"].max():.4f}] '
              f'off=[{onnx_results["offsets"].min():.4f},{onnx_results["offsets"].max():.4f}]')
        np.save(str(out_dir / 'onnx_heatmaps.npy'), onnx_results['heatmaps'])
        np.save(str(out_dir / 'onnx_mask.npy'), onnx_results['mask'])
        np.save(str(out_dir / 'onnx_offsets.npy'), onnx_results['offsets'])
        print('[ONNX] saved')

    # ---------- PT vs ONNX ----------
    if pt_results and onnx_results:
        print('\n=== PT vs ONNX ===')
        for name in ['heatmaps', 'mask', 'offsets']:
            if name in pt_results and name in onnx_results:
                diff = np.abs(pt_results[name] - onnx_results[name])
                print(f'  {name}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}',
                      'PASS' if diff.max() < 1e-4 else 'FAIL')
                # Save diff visualization for heatmaps
                if name == 'heatmaps':
                    diff_img = (diff[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                    cv2.imwrite(str(out_dir / 'diff_heatmaps.png'),
                                np.concatenate([diff_img] * 3, axis=-1) if diff_img.shape[-1] == 1 else diff_img)

    # ---------- RKNN ----------
    rknn_results = {}
    if args.rknn:
        try:
            from rknn.api import RKNN
            rknn = RKNN(verbose=False)
            ret = rknn.load_rknn(args.rknn)
            if ret != 0:
                print(f'[RKNN] load_rknn failed: {ret}')
            else:
                ret = rknn.init_runtime()
                if ret != 0:
                    print(f'[RKNN] init_runtime failed (simulator): {ret}')
                    print('  -> RKNN can only run on RK3568 hardware. Skipping.')
                else:
                    x_np = patch_f.astype(np.float32).reshape(1, 3, 128, 256)
                    rknn_out = rknn.inference(inputs=[x_np])
                    rknn.release()
                    if len(rknn_out) >= 3:
                        rknn_results['heatmaps'] = 1.0 / (1.0 + np.exp(-rknn_out[0]))
                        rknn_results['mask'] = 1.0 / (1.0 + np.exp(-rknn_out[1]))
                        rknn_results['offsets'] = rknn_out[2]
                        print(f'[RKNN] heatmaps={rknn_results["heatmaps"].shape} offsets={rknn_results["offsets"].shape}')
                        np.save(str(out_dir / 'rknn_heatmaps.npy'), rknn_results['heatmaps'])
                        np.save(str(out_dir / 'rknn_mask.npy'), rknn_results['mask'])
                        np.save(str(out_dir / 'rknn_offsets.npy'), rknn_results['offsets'])
                        print('[RKNN] saved')
        except ImportError:
            print('[RKNN] rknn toolkit not available, skipping')

    # Compare RKNN vs ONNX (if both available)
    if rknn_results and onnx_results:
        print('\n=== ONNX vs RKNN ===')
        for name in ['heatmaps', 'mask', 'offsets']:
            if name in rknn_results and name in onnx_results:
                diff = np.abs(rknn_results[name] - onnx_results[name])
                print(f'  {name}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}',
                      'PASS' if diff.max() < 1e-3 else 'FAIL')

    # ---------- Save input patch for board-side replay ----------
    cv2.imwrite(str(out_dir / 'refiner_input_patch.png'),
                (patch_f.transpose(1, 2, 0) * 255).astype(np.uint8))
    print(f'\nAll outputs saved to {out_dir}')
    print('  refiner_input_patch.png - 输入 patch (用来在板端复现)')
    print('  pt_*.npy - PyTorch 输出')
    print('  onnx_*.npy - ONNX 输出')
    print('  rknn_*.npy - RKNN 输出（如可用）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
