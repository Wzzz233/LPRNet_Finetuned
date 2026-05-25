#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from rknn.api import RKNN

from model.LPRNet import small_basic_block

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
MEAN_VALUES = [[127.5, 127.5, 127.5]]
STD_VALUES = [[128.0, 128.0, 128.0]]


class Box:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def w(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def h(self) -> int:
        return self.y2 - self.y1 + 1


def parse_ccpd_bbox_from_name(image_name: str):
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    bbox = parts[2]
    if '_' not in bbox:
        return None
    p1, p2 = bbox.split('_', 1)
    if '&' not in p1 or '&' not in p2:
        return None
    x1s, y1s = p1.split('&', 1)
    x2s, y2s = p2.split('&', 1)
    try:
        return Box(int(x1s), int(y1s), int(x2s), int(y2s))
    except ValueError:
        return None


def clamp_box(box: Box, img_w: int, img_h: int) -> Box:
    x1 = max(0, min(img_w - 1, box.x1))
    y1 = max(0, min(img_h - 1, box.y1))
    x2 = max(0, min(img_w - 1, box.x2))
    y2 = max(0, min(img_h - 1, box.y2))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return Box(x1, y1, x2, y2)


def compute_expand_crop_box(src: Box, img_w: int, img_h: int, pad_x: float, pad_y: float) -> Box:
    ex = int(src.w * pad_x + 0.5)
    ey = int(src.h * pad_y + 0.5)
    return clamp_box(Box(src.x1 - ex, src.y1 - ey, src.x2 + ex, src.y2 + ey), img_w, img_h)


def compute_ocr_crop_box(src: Box, img_w: int, img_h: int, crop_mode: str) -> Box:
    if crop_mode == 'match':
        return clamp_box(src, img_w, img_h)
    if crop_mode == 'box':
        return compute_expand_crop_box(src, img_w, img_h, 0.06, 0.12)
    if crop_mode == 'tight':
        return compute_expand_crop_box(src, img_w, img_h, 0.08, 0.16)
    if crop_mode == 'box-pad':
        return compute_expand_crop_box(src, img_w, img_h, 0.15, 0.28)
    return clamp_box(src, img_w, img_h)


def estimate_ocr_occ_ratio(crop_w: int, crop_h: int, in_w: int, in_h: int, resize_mode: str) -> float:
    if crop_w <= 0 or crop_h <= 0 or in_w <= 0 or in_h <= 0:
        return 0.0
    if resize_mode != 'letterbox':
        return 1.0
    sx = in_w / float(crop_w)
    sy = in_h / float(crop_h)
    scale = min(sx, sy)
    if scale <= 0.0:
        return 0.0
    scaled_w = int(crop_w * scale + 0.5)
    scaled_w = max(1, min(in_w, scaled_w))
    return scaled_w / float(in_w)


def compute_match_ytrim_crop(src: Box, in_w: int, in_h: int, min_occ_ratio: float):
    if min_occ_ratio <= 0.0:
        return None
    bw = src.w
    bh = src.h
    if bw <= 0 or bh <= 0 or in_w <= 0 or in_h <= 0:
        return None
    min_occ_ratio = min(1.0, min_occ_ratio)
    model_aspect = in_w / float(in_h)
    target_aspect = model_aspect * min_occ_ratio
    if target_aspect <= 0.0:
        return None
    cur_aspect = bw / float(bh)
    if cur_aspect >= target_aspect:
        return None
    new_h = int(bw / target_aspect + 0.5)
    if new_h < 1:
        new_h = 1
    if new_h >= bh:
        return None
    trim = (bh - new_h) // 2
    y1 = src.y1 + trim
    y2 = y1 + new_h - 1
    if y2 > src.y2:
        y2 = src.y2
        y1 = y2 - new_h + 1
    if y1 < src.y1:
        y1 = src.y1
    if y2 <= y1:
        return None
    return Box(src.x1, y1, src.x2, y2)


def resize_bgr_with_kernel(src: np.ndarray, dst_w: int, dst_h: int, kernel: str) -> np.ndarray:
    interp = cv2.INTER_LINEAR if kernel == 'bilinear' else cv2.INTER_NEAREST
    return cv2.resize(src, (dst_w, dst_h), interpolation=interp)


def resize_bgr_letterbox(src: np.ndarray, dst_w: int, dst_h: int, kernel: str, pad_value: int = 0):
    src_h, src_w = src.shape[:2]
    out = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    scale = min(sx, sy)
    if scale <= 0.0:
        return out
    scaled_w = int(src_w * scale + 0.5)
    scaled_h = int(src_h * scale + 0.5)
    scaled_w = max(1, min(dst_w, scaled_w))
    scaled_h = max(1, min(dst_h, scaled_h))
    off_x = (dst_w - scaled_w) // 2
    off_y = (dst_h - scaled_h) // 2
    resized = resize_bgr_with_kernel(src, scaled_w, scaled_h, kernel)
    out[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized
    return out


def ocr_preprocess_bgr888(img: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or 'none').strip().lower()
    if mode == 'raw':
        mode = 'none'
    if mode == 'gray3':
        mode = 'gray'
    if mode == 'none':
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'gray':
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == 'bin':
        tmp = cv2.blur(gray, (3, 3))
        thr = cv2.blur(tmp, (5, 5))
        out = np.where(tmp > (thr - 8), 255, 0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported ocr preproc mode: {mode}')


def prepare_board_ocr_input_bgr888(crop_bgr: np.ndarray, in_w: int, in_h: int, resize_mode: str, resize_kernel: str, preproc_mode: str, channel_order: str) -> np.ndarray:
    work = ocr_preprocess_bgr888(crop_bgr.copy(), preproc_mode)
    if resize_mode == 'letterbox':
        out = resize_bgr_letterbox(work, in_w, in_h, resize_kernel, pad_value=0)
    else:
        out = resize_bgr_with_kernel(work, in_w, in_h, resize_kernel)
    if channel_order == 'rgb':
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


class maxpool_3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
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


def decode_logits(logits: np.ndarray) -> str:
    if logits.ndim == 3:
        logits = logits[0]
    pred = np.argmax(logits, axis=0).tolist()
    out = []
    prev = None
    blank = len(CHARS) - 1
    for idx in pred:
        if idx == blank or idx == prev:
            prev = idx
            continue
        out.append(CHARS[idx])
        prev = idx
    return "".join(out)


def normalize_for_torch(img_bgr: np.ndarray) -> np.ndarray:
    x = img_bgr.astype("float32")
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0)


def build_board_input(
    image_path: Path,
    img_w: int,
    img_h: int,
    crop_mode: str,
    resize_mode: str,
    resize_kernel: str,
    preproc_mode: str,
    channel_order: str,
    min_occ_ratio: float,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    ih, iw = image.shape[:2]

    bbox = parse_ccpd_bbox_from_name(str(image_path))
    if bbox is None:
        raise RuntimeError(f"cannot parse CCPD bbox from: {image_path}")
    bbox = clamp_box(bbox, iw, ih)
    crop_box = compute_ocr_crop_box(bbox, iw, ih, crop_mode)
    occ_ratio = estimate_ocr_occ_ratio(crop_box.w, crop_box.h, img_w, img_h, resize_mode)
    if min_occ_ratio > 0.0 and occ_ratio < min_occ_ratio and crop_mode != "tight":
        if crop_mode == "match":
            recrop = compute_match_ytrim_crop(crop_box, img_w, img_h, min_occ_ratio)
            if recrop is not None:
                crop_box = clamp_box(recrop, iw, ih)
        else:
            crop_box = compute_expand_crop_box(bbox, iw, ih, 0.08, 0.16)

    crop_bgr = image[crop_box.y1:crop_box.y2 + 1, crop_box.x1:crop_box.x2 + 1]
    board_bgr = prepare_board_ocr_input_bgr888(
        crop_bgr,
        img_w,
        img_h,
        resize_mode,
        resize_kernel,
        preproc_mode,
        channel_order,
    )
    return board_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch / ONNX / RKNN outputs on one board-aligned sample.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--image", required=True, help="CCPD image path to verify.")
    parser.add_argument("--img-size", nargs=2, type=int, default=[94, 24], metavar=("W", "H"))
    parser.add_argument("--target-platform", default="rk3568")
    parser.add_argument("--ocr_channel_order", default="bgr", choices=["rgb", "bgr"])
    parser.add_argument("--ocr_crop_mode", default="match", choices=["fixed", "box", "tight", "box-pad", "match"])
    parser.add_argument("--ocr_resize_mode", default="letterbox", choices=["stretch", "letterbox"])
    parser.add_argument("--ocr_resize_kernel", default="nn", choices=["nn", "bilinear"])
    parser.add_argument("--ocr_preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"])
    parser.add_argument("--ocr_min_occ_ratio", default=0.90, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    board_bgr = build_board_input(
        image_path=image_path,
        img_w=args.img_size[0],
        img_h=args.img_size[1],
        crop_mode=args.ocr_crop_mode,
        resize_mode=args.ocr_resize_mode,
        resize_kernel=args.ocr_resize_kernel,
        preproc_mode=args.ocr_preproc,
        channel_order=args.ocr_channel_order,
        min_occ_ratio=args.ocr_min_occ_ratio,
    )
    nchw = normalize_for_torch(board_bgr)

    device = torch.device("cpu")
    model = LPRNetExport(class_num=len(CHARS), dropout_rate=0).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        pt_out = model(torch.from_numpy(nchw)).cpu().numpy()

    ort_sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_out = ort_sess.run(["output"], {"input": nchw})[0]

    if not Path(args.rknn).exists():
        raise FileNotFoundError(f"rknn not found: {args.rknn}")

    rknn = RKNN(verbose=False)
    ret = rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=args.target_platform,
        quant_img_RGB2BGR=False,
        float_dtype="float16",
        optimization_level=0,
    )
    if ret != 0:
        raise RuntimeError(f"rknn config failed: {ret}")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError(f"load_onnx for simulator failed: {ret}")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn build for simulator failed: {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    rknn_out = rknn.inference(inputs=[np.expand_dims(board_bgr, axis=0)], data_format=["nhwc"])[0]
    rknn.release()

    print(f"image={image_path}")
    print(f"board_input_shape={board_bgr.shape}")
    print(f"pytorch_pred={decode_logits(pt_out)}")
    print(f"onnx_pred={decode_logits(onnx_out)}")
    print(f"rknn_pred={decode_logits(rknn_out)}")
    print(f"pt_onnx_max_abs={float(np.max(np.abs(pt_out - onnx_out))):.8f}")
    print(f"pt_onnx_mean_abs={float(np.mean(np.abs(pt_out - onnx_out))):.8f}")
    print(f"pt_rknn_max_abs={float(np.max(np.abs(pt_out - rknn_out))):.8f}")
    print(f"pt_rknn_mean_abs={float(np.mean(np.abs(pt_out - rknn_out))):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
