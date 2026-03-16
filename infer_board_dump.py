#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch

from data.load_data import CHARS
from model.LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits


def read_ppm_payload(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if not blob.startswith(b'P6'):
        raise ValueError(f'unsupported ppm format: {path}')
    i = 2
    tokens = []
    n = len(blob)
    while len(tokens) < 3:
        while i < n and blob[i] in b' \t\r\n':
            i += 1
        if i < n and blob[i] == ord('#'):
            while i < n and blob[i] not in b'\r\n':
                i += 1
            continue
        j = i
        while j < n and blob[j] not in b' \t\r\n':
            j += 1
        tokens.append(blob[i:j].decode('ascii'))
        i = j
    w, h, maxv = map(int, tokens)
    if maxv != 255:
        raise ValueError(f'unsupported ppm max value: {maxv}')
    while i < n and blob[i] in b' \t\r\n':
        i += 1
    payload = np.frombuffer(blob[i:], dtype=np.uint8)
    if payload.size != w * h * 3:
        raise ValueError(f'invalid ppm payload size: expected {w*h*3}, got {payload.size}')
    return payload.reshape(h, w, 3).copy()


def decode_logits(logits: torch.Tensor) -> str:
    decoded = greedy_decode_logits(logits.detach().cpu().numpy())
    return ''.join(CHARS[int(c)] for c in decoded[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run LPRNet on dumped board OCR PPM payload.')
    parser.add_argument('--weights', required=True, help='LPRNet .pth path')
    parser.add_argument('--image', required=True, help='PPM dump path from board OCR input')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    weights_path = Path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f'image not found: {image_path}')
    if not weights_path.exists():
        raise FileNotFoundError(f'weights not found: {weights_path}')

    img = read_ppm_payload(image_path)
    x = img.astype('float32')
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, ...]

    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(str(weights_path), map_location='cpu'))
    net.eval()

    with torch.no_grad():
        logits = net(torch.from_numpy(x))
    probs = torch.softmax(logits[0], dim=0)
    topk = torch.topk(probs[:, 0], k=5)
    top5 = [(CHARS[int(i)], float(v)) for v, i in zip(topk.values, topk.indices)]

    print(f'image={image_path}')
    print(f'weights={weights_path}')
    print(f'pred={decode_logits(logits)}')
    print(f'pos1_top5={top5}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
