#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from data.load_data import CHARS
from model.LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference verification with train/test-consistent preprocessing and decode.")
    parser.add_argument("--weights", default="./weights_red_stage3/Final_LPRNet_model.pth", help="Model weights path.")
    parser.add_argument("--img_dir", default="./balanced_ccpd_red_aug_ppm", help="Directory of ppm images.")
    parser.add_argument("--labels", default="./balanced_ccpd_red_aug_ppm/train_labels.txt", help="Label file path.")
    parser.add_argument("--img_w", type=int, default=94, help="Input width.")
    parser.add_argument("--img_h", type=int, default=24, help="Input height.")
    parser.add_argument("--num_samples", type=int, default=20, help="Random sample count.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser.parse_args()


def load_label_map(label_path: Path) -> Dict[str, str]:
    if not label_path.exists():
        alt = Path("./train_labels.txt")
        if alt.exists():
            label_path = alt
        else:
            raise FileNotFoundError(f"Cannot find label file: {label_path}")

    label_map: Dict[str, str] = {}
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            name, plate = parts
            label_map[Path(name).name] = plate.strip()
    if not label_map:
        raise RuntimeError(f"No valid labels parsed from: {label_path}")
    return label_map


def preprocess_like_train(img_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    # 1:1 replicate LPRDataLoader.__getitem__ + transform in data/load_data.py:
    # cv2.imread(BGR) -> resize(94x24) -> float32 -> minus 127.5 -> multiply 0.0078125 -> HWC->CHW
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Failed to read image by cv2: {img_path}")
    h, w, _ = image.shape
    if h != img_h or w != img_w:
        image = cv2.resize(image, (img_w, img_h))
    image = image.astype("float32")
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def classify_error(gt: str, pred: str) -> str:
    if gt == pred:
        return "正确"
    if len(pred) < len(gt):
        return "序列变短，疑似 CTC 塌陷（字符丢失）"
    if len(pred) > len(gt):
        return "序列变长，疑似重复解码或插入噪声字符"
    if gt and pred and gt[0] != pred[0]:
        return "省份汉字错误（首字符错）"
    conf_pairs = {("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("2", "Z"), ("Z", "2"), ("8", "B"), ("B", "8")}
    for g, p in zip(gt, pred):
        if g != p:
            if (g, p) in conf_pairs:
                return f"易混淆字符错误（{g}->{p}）"
            if g.isdigit() and p.isalpha():
                return f"数字误识别为字母（{g}->{p}）"
            if g.isalpha() and p.isdigit():
                return f"字母误识别为数字（{g}->{p}）"
            return f"普通字符替换错误（{g}->{p}）"
    return "未知类型错误"


def pick_samples(label_map: Dict[str, str], img_dir: Path, n: int, seed: int) -> List[Tuple[str, str, Path]]:
    rng = random.Random(seed)
    all_items = [(name, plate) for name, plate in label_map.items() if (img_dir / name).exists()]
    if len(all_items) < n:
        raise RuntimeError(f"Not enough images in {img_dir}: need {n}, got {len(all_items)}")

    by_prov: Dict[str, List[Tuple[str, str]]] = {}
    for name, plate in all_items:
        prov = plate[0] if plate else "?"
        by_prov.setdefault(prov, []).append((name, plate))

    selected: List[Tuple[str, str, Path]] = []
    used = set()

    # First pass: one per province to increase diversity
    provs = list(by_prov.keys())
    rng.shuffle(provs)
    for prov in provs:
        if len(selected) >= n:
            break
        item = rng.choice(by_prov[prov])
        if item[0] not in used:
            selected.append((item[0], item[1], img_dir / item[0]))
            used.add(item[0])

    # Second pass: random fill
    remain = [(name, plate) for name, plate in all_items if name not in used]
    rng.shuffle(remain)
    for name, plate in remain:
        if len(selected) >= n:
            break
        selected.append((name, plate, img_dir / name))
        used.add(name)
    return selected


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    img_dir = Path(args.img_dir)
    label_map = load_label_map(Path(args.labels))

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    samples = pick_samples(label_map, img_dir, args.num_samples, args.seed)

    device = torch.device("cpu")
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    state = torch.load(str(weights), map_location=device)
    lprnet.load_state_dict(state)
    lprnet.to(device)
    lprnet.eval()  # Force eval mode for BN/Dropout
    if lprnet.training:
        raise RuntimeError("Model is still in training mode.")

    batch = torch.cat([preprocess_like_train(p, args.img_w, args.img_h) for _, _, p in samples], dim=0).to(device)

    with torch.no_grad():
        logits = lprnet(batch)
    prebs = logits.cpu().detach().numpy()

    # Reuse official decode implementation from test_LPRNet.py
    decoded_labels = greedy_decode_logits(prebs)
    preds = ["".join(CHARS[int(c)] for c in label) for label in decoded_labels]

    print(f"=== 终极盲盒推理（随机{args.num_samples}张）===")
    print(f"weights: {weights.resolve()}")
    print(f"img_dir: {img_dir.resolve()}")
    print(f"model_eval_mode: {not lprnet.training}")
    print()

    bad_cases: List[Tuple[str, str, str, str]] = []
    for idx, ((img_name, gt_text, _), pred) in enumerate(zip(samples, preds), start=1):
        reason = classify_error(gt_text, pred)
        tag = "[正确]" if gt_text == pred else "[错误]"
        print(f"[{idx}] 图片文件: {img_name}")
        print(f"    真实标签: {gt_text}")
        print(f"    模型预测: {pred} {tag}")
        print(f"    判定结果: {reason}")
        print("-" * 60)
        if gt_text != pred:
            bad_cases.append((img_name, gt_text, pred, reason))

    print("\n=== Bad Case 汇总 ===")
    if not bad_cases:
        print(f"本次 {len(samples)} 张全部预测正确。")
    else:
        for idx, (img_name, gt_text, pred, reason) in enumerate(bad_cases, start=1):
            print(f"[Bad {idx}] {img_name}")
            print(f"    真实标签: {gt_text}")
            print(f"    模型预测: {pred}")
            print(f"    错误分析: {reason}")
            if "序列变短" in reason or "序列变长" in reason:
                print("    简要判断: 更像是增强噪声/模糊过强导致时序不稳定。")
            elif "易混淆字符" in reason or "字母误识别为数字" in reason or "数字误识别为字母" in reason:
                print("    简要判断: 仍存在形近字母-数字混淆。")
            elif "省份汉字错误" in reason:
                print("    简要判断: 省份字头仍有长尾混淆。")
            else:
                print("    简要判断: 复杂纹理干扰导致局部字符判别错误。")
            print("-" * 60)
    print(f"总计: {len(samples)} 张，正确: {len(samples)-len(bad_cases)}，错误: {len(bad_cases)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
