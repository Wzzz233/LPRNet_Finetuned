#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from load_data import (
    Box,
    CHARS,
    clamp_box,
    compute_ocr_crop_box,
    parse_ccpd_bbox_from_name,
    prepare_board_ocr_input_bgr888,
    prepare_board_ocr_input_from_quad_bgr888,
    quad_to_box,
)


def parse_bbox(text):
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    return Box(*vals)


def parse_quad(text):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 8:
        raise ValueError("quad must be x1,y1,x2,y2,x3,y3,x4,y4")
    return np.asarray(vals, dtype=np.float32).reshape(4, 2)


def decode_lpr(prepared_img, weights_path):
    import torch

    from model.LPRNet import build_lprnet
    from test_LPRNet import greedy_decode_logits

    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    net.eval()

    x = prepared_img.astype("float32")
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, ...]
    with torch.no_grad():
        logits = net(torch.from_numpy(x)).detach().cpu().numpy()
    decoded = greedy_decode_logits(logits)[0]
    text = "".join(CHARS[int(idx)] for idx in decoded)
    return text


def draw_debug_overlay(image, bbox=None, quad=None):
    canvas = image.copy()
    if bbox is not None:
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
    if quad is not None:
        pts = np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Compare axis-aligned bbox crop vs OBB quad crop for OCR input.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--bbox", default=None, help="Axis-aligned bbox x1,y1,x2,y2.")
    parser.add_argument("--quad", default=None, help="Oriented quad x1,y1,...,x4,y4.")
    parser.add_argument("--ccpd-bbox-from-name", action="store_true", help="Parse bbox from CCPD filename when --bbox is omitted.")
    parser.add_argument("--output-dir", default="./artifacts/obb_crop_compare", help="Directory to save crops and JSON summary.")
    parser.add_argument("--crop-mode", default="match", choices=["fixed", "box", "tight", "box-pad", "match"], help="BBox crop mode.")
    parser.add_argument("--ocr-width", type=int, default=94, help="OCR input width.")
    parser.add_argument("--ocr-height", type=int, default=24, help="OCR input height.")
    parser.add_argument("--resize-mode", default="letterbox", choices=["stretch", "letterbox"], help="OCR resize mode.")
    parser.add_argument("--resize-kernel", default="nn", choices=["nn", "bilinear"], help="OCR resize kernel.")
    parser.add_argument("--preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"], help="OCR preproc mode.")
    parser.add_argument("--channel-order", default="bgr", choices=["bgr", "rgb"], help="OCR channel order.")
    parser.add_argument("--quad-pad-ratio", type=float, default=0.0, help="Extra padding ratio applied around the quad before warping.")
    parser.add_argument("--weights", default=None, help="Optional LPRNet weights to run OCR on saved crops.")
    parser.add_argument("--label", default=None, help="Optional ground-truth plate text.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] failed to read image: {image_path}")
        return 1

    img_h, img_w = image.shape[:2]
    bbox = parse_bbox(args.bbox) if args.bbox else None
    quad = parse_quad(args.quad) if args.quad else None
    if bbox is None and args.ccpd_bbox_from_name:
        bbox = parse_ccpd_bbox_from_name(image_path.name)
    if bbox is None and quad is not None:
        bbox = quad_to_box(quad, img_w=img_w, img_h=img_h)

    if bbox is None and quad is None:
        print("[ERROR] provide --bbox, --quad, or --ccpd-bbox-from-name")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "image": str(image_path),
        "ocr_input_size": [args.ocr_width, args.ocr_height],
        "label": args.label,
        "bbox": None,
        "quad": None,
        "results": {},
    }

    debug_overlay = draw_debug_overlay(image, bbox=bbox, quad=quad)
    overlay_path = output_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_path), debug_overlay)

    if bbox is not None:
        bbox = clamp_box(bbox, img_w, img_h)
        crop_box = compute_ocr_crop_box(bbox, img_w, img_h, args.crop_mode)
        raw_bbox = image[crop_box.y1:crop_box.y2 + 1, crop_box.x1:crop_box.x2 + 1]
        prepared_bbox, occ_bbox = prepare_board_ocr_input_bgr888(
            raw_bbox,
            args.ocr_width,
            args.ocr_height,
            args.resize_mode,
            args.resize_kernel,
            args.preproc,
            args.channel_order,
        )
        bbox_raw_path = output_dir / "bbox_raw.jpg"
        bbox_prepared_path = output_dir / "bbox_prepared.jpg"
        cv2.imwrite(str(bbox_raw_path), raw_bbox)
        cv2.imwrite(str(bbox_prepared_path), prepared_bbox)
        summary["bbox"] = {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}
        summary["results"]["bbox"] = {
            "crop_mode": args.crop_mode,
            "crop_box": {
                "x1": crop_box.x1,
                "y1": crop_box.y1,
                "x2": crop_box.x2,
                "y2": crop_box.y2,
            },
            "occupancy": occ_bbox,
            "raw_path": str(bbox_raw_path),
            "prepared_path": str(bbox_prepared_path),
        }

    if quad is not None:
        prepared_quad, occ_quad, warped_quad, ordered_quad, _ = prepare_board_ocr_input_from_quad_bgr888(
            image,
            quad,
            args.ocr_width,
            args.ocr_height,
            args.resize_mode,
            args.resize_kernel,
            args.preproc,
            args.channel_order,
            quad_pad_ratio=args.quad_pad_ratio,
        )
        quad_raw_path = output_dir / "quad_raw.jpg"
        quad_prepared_path = output_dir / "quad_prepared.jpg"
        cv2.imwrite(str(quad_raw_path), warped_quad)
        cv2.imwrite(str(quad_prepared_path), prepared_quad)
        summary["quad"] = np.asarray(ordered_quad, dtype=float).round(3).tolist()
        summary["results"]["quad"] = {
            "quad_pad_ratio": args.quad_pad_ratio,
            "occupancy": occ_quad,
            "raw_path": str(quad_raw_path),
            "prepared_path": str(quad_prepared_path),
        }

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"[ERROR] weights not found: {weights_path}")
            return 1
        for key in ("bbox", "quad"):
            prepared_path = summary["results"].get(key, {}).get("prepared_path")
            if not prepared_path:
                continue
            prepared = cv2.imread(prepared_path)
            pred = decode_lpr(prepared, weights_path)
            summary["results"][key]["prediction"] = pred
            if args.label is not None:
                summary["results"][key]["label_match"] = pred == args.label

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] summary saved: {summary_path.resolve()}")
    print(f"[OK] overlay saved: {overlay_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
