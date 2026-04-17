#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import CHARS, LPRDataLoader, CCPDBoardDataLoader, UnifiedManifestDataset, PROVINCE_COUNT
from lpr_pipeline_policy import BOARD_PARAM_EXPECTED
from LPRNet import build_lprnet
from LPRNet_multihead import build_lprnet_multihead, build_lprnet_multihead_from_state_dict
from test_LPRNet import collate_fn, greedy_decode_logits


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser(description="Detailed LPRNet evaluation with plate/length/position metrics.")
    parser.add_argument("--img_size", default=[94, 24], nargs=2, type=int)
    parser.add_argument("--test_img_dirs", required=True)
    parser.add_argument("--txt_file", required=True)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--head_mode", default="single", choices=["single", "multihead"])
    parser.add_argument("--enhanced_green_head", default='', choices=['', 'expD', 'expE'], help='enhanced green8 head variant: expD=2-layer(256ch), expE=3-layer(512ch)')
    parser.add_argument("--lpr_max_len", default=8, type=int)
    parser.add_argument("--data_mode", default="standard", choices=["standard", "ccpd_board", "manifest"])
    parser.add_argument("--ocr_channel_order", default="bgr", choices=["rgb", "bgr"])
    parser.add_argument("--ocr_crop_mode", default="match", choices=["fixed", "box", "tight", "box-pad", "match", "obb_warp"])
    parser.add_argument("--ocr_resize_mode", default="letterbox", choices=["stretch", "letterbox"])
    parser.add_argument("--ocr_resize_kernel", default="nn", choices=["nn", "bilinear"])
    parser.add_argument("--ocr_preproc", default="none", choices=["none", "raw", "gray", "gray3", "bin"])
    parser.add_argument("--ocr_min_occ_ratio", default=0.90, type=float)
    parser.add_argument("--ocr_quad_pad_ratio", default=0.0, type=float)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--cuda", default=False, type=str2bool)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--decode_mode", default="greedy", choices=["greedy", "green_ctc_beam", "family_aware_beam"])
    parser.add_argument("--assume_family", default="", choices=["", "normal7", "green8", "special"], help="force a single family for all samples when dataset itself does not carry family metadata")
    parser.add_argument("--beam_size", default=30, type=int)
    parser.add_argument("--beam_topk", default=15, type=int)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--bad_case_topk", default=20, type=int)
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument("--seed", default=20260320, type=int)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--pos0_head_cols", default=0, type=int, help='spatial columns for pos0 head; 0 disables')
    parser.add_argument("--pos0_num_classes", default=31, type=int)
    return parser


def enforce_board_alignment_args(args):
    mismatches = []
    for key, expected in BOARD_PARAM_EXPECTED.items():
        actual = getattr(args, key, None)
        if isinstance(expected, float):
            try:
                ok = abs(float(actual) - float(expected)) <= 1e-6
            except Exception:
                ok = False
        else:
            ok = (actual == expected)
        if not ok:
            mismatches.append(f'{key}={actual} expected {expected}')
    if mismatches:
        print(f'[WARN] Board-aligned OCR params overridden: {"; ".join(mismatches)}')


def configure_runtime(seed, deterministic):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


GREEN_PROVINCES = set(CHARS[:PROVINCE_COUNT])
GREEN_ALPHA = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
GREEN_ALNUM = set("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
NORMAL7_ALPHA = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
NORMAL7_ALNUM = set("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")


def unwrap_dataset_and_index(dataset, index):
    current = dataset
    current_index = index
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        current_index = current.indices[current_index]
        current = current.dataset
    return current, current_index


def get_manifest_record(dataset, index):
    base_dataset, base_index = unwrap_dataset_and_index(dataset, index)
    records = getattr(base_dataset, "records", None)
    if records is None or base_index >= len(records):
        return None
    return records[base_index]


def get_sample_family(dataset, index):
    row = get_manifest_record(dataset, index)
    if not row:
        return None
    family = (row.get("family") or "").strip()
    return family or None


def green_prefix_valid(text):
    length = len(text)
    if length == 0:
        return True
    if text[0] not in GREEN_PROVINCES:
        return False
    if length >= 2 and text[1] not in GREEN_ALPHA:
        return False
    if length > 8:
        return False
    if length >= 3:
        if text[2] in {"D", "F"}:
            if any(c not in GREEN_ALNUM for c in text[3:]):
                return False
        else:
            if any(c not in GREEN_ALNUM for c in text[2:]):
                return False
            if length == 8 and text[7] not in {"D", "F"}:
                return False
    return True


def green_full_valid(text):
    if len(text) != 8:
        return False
    if not green_prefix_valid(text):
        return False
    if text[2] in {"D", "F"}:
        return all(c in GREEN_ALNUM for c in text[3:])
    return all(c in GREEN_ALNUM for c in text[2:7]) and text[7] in {"D", "F"}


def normal7_prefix_valid(text):
    length = len(text)
    if length == 0:
        return True
    if text[0] not in GREEN_PROVINCES:
        return False
    if length >= 2 and text[1] not in NORMAL7_ALPHA:
        return False
    if length > 7:
        return False
    if any(c not in NORMAL7_ALNUM for c in text[2:]):
        return False
    return True


def normal7_full_valid(text):
    return len(text) == 7 and normal7_prefix_valid(text)


def family_prefix_valid(family, text):
    if family == "green8":
        return green_prefix_valid(text)
    if family == "normal7":
        return normal7_prefix_valid(text)
    return True


def family_full_valid(family, text):
    if family == "green8":
        return green_full_valid(text)
    if family == "normal7":
        return normal7_full_valid(text)
    return True


def family_target_length(family):
    if family == "green8":
        return 8
    if family == "normal7":
        return 7
    return None


def constrained_ctc_beam_decode_single(logits_ct, family, beam_size=30, topk=15):
    if family not in {"normal7", "green8"}:
        return None

    blank_idx = len(CHARS) - 1
    topk = max(2, int(topk))
    beam_size = max(2, int(beam_size))
    log_probs = logits_ct - np.logaddexp.reduce(logits_ct, axis=0, keepdims=True)
    beams = {"": (0.0, -1e18)}
    max_len = family_target_length(family)

    for t in range(log_probs.shape[1]):
        next_beams = {}
        col = log_probs[:, t]
        idxs = np.argpartition(col, -topk)[-topk:]
        idxs = idxs[np.argsort(col[idxs])[::-1]]
        if blank_idx not in idxs:
            idxs = np.append(idxs, blank_idx)

        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams.get(prefix, (-1e18, -1e18))
            nb_pb = np.logaddexp(nb_pb, np.logaddexp(pb, pnb) + col[blank_idx])
            next_beams[prefix] = (nb_pb, nb_pnb)

            for c in idxs:
                if c == blank_idx:
                    continue
                ch = CHARS[int(c)]
                new_prefix = prefix + ch
                if max_len is not None and len(new_prefix) > max_len:
                    continue
                if not family_prefix_valid(family, new_prefix):
                    continue

                npb, npnb = next_beams.get(new_prefix, (-1e18, -1e18))
                if prefix and prefix[-1] == ch:
                    score = pb + col[c]
                else:
                    score = np.logaddexp(pb, pnb) + col[c]
                npnb = np.logaddexp(npnb, score)
                next_beams[new_prefix] = (npb, npnb)

                if prefix and prefix[-1] == ch:
                    rpb, rpnb = next_beams.get(prefix, (-1e18, -1e18))
                    rpnb = np.logaddexp(rpnb, pnb + col[c])
                    next_beams[prefix] = (rpb, rpnb)

        items = sorted(
            next_beams.items(),
            key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = dict(items[:beam_size])

    best_score = None
    best_text = ""
    for prefix, (pb, pnb) in beams.items():
        if not family_full_valid(family, prefix):
            continue
        score = np.logaddexp(pb, pnb)
        if best_score is None or score > best_score:
            best_score = score
            best_text = prefix
    if best_score is not None:
        return [CHARS.index(c) for c in best_text]

    fallback = []
    for prefix, (pb, pnb) in beams.items():
        if family_prefix_valid(family, prefix):
            score = np.logaddexp(pb, pnb)
            fallback.append((score, prefix))
    if fallback:
        fallback.sort(reverse=True)
        return [CHARS.index(c) for c in fallback[0][1]]
    return []


def green_constrained_ctc_beam_decode_single(logits_ct, beam_size=30, topk=15):
    blank_idx = len(CHARS) - 1
    topk = max(2, int(topk))
    beam_size = max(2, int(beam_size))
    log_probs = logits_ct - np.logaddexp.reduce(logits_ct, axis=0, keepdims=True)
    beams = {"": (0.0, -1e18)}  # prefix -> (pb, pnb)

    for t in range(log_probs.shape[1]):
        next_beams = {}
        col = log_probs[:, t]
        idxs = np.argpartition(col, -topk)[-topk:]
        idxs = idxs[np.argsort(col[idxs])[::-1]]
        if blank_idx not in idxs:
            idxs = np.append(idxs, blank_idx)

        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams.get(prefix, (-1e18, -1e18))
            nb_pb = np.logaddexp(nb_pb, np.logaddexp(pb, pnb) + col[blank_idx])
            next_beams[prefix] = (nb_pb, nb_pnb)

            for c in idxs:
                if c == blank_idx:
                    continue
                ch = CHARS[int(c)]
                new_prefix = prefix + ch
                if not green_prefix_valid(new_prefix):
                    continue

                npb, npnb = next_beams.get(new_prefix, (-1e18, -1e18))
                if prefix and prefix[-1] == ch:
                    score = pb + col[c]
                else:
                    score = np.logaddexp(pb, pnb) + col[c]
                npnb = np.logaddexp(npnb, score)
                next_beams[new_prefix] = (npb, npnb)

                if prefix and prefix[-1] == ch:
                    rpb, rpnb = next_beams.get(prefix, (-1e18, -1e18))
                    rpnb = np.logaddexp(rpnb, pnb + col[c])
                    next_beams[prefix] = (rpb, rpnb)

        items = sorted(
            next_beams.items(),
            key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = dict(items[:beam_size])

    best_score = None
    best_text = ""
    for prefix, (pb, pnb) in beams.items():
        if not green_full_valid(prefix):
            continue
        score = np.logaddexp(pb, pnb)
        if best_score is None or score > best_score:
            best_score = score
            best_text = prefix
    if best_score is not None:
        return [CHARS.index(c) for c in best_text]

    fallback = []
    for prefix, (pb, pnb) in beams.items():
        if green_prefix_valid(prefix) and len(prefix) <= 8:
            score = np.logaddexp(pb, pnb)
            fallback.append((score, prefix))
    if fallback:
        fallback.sort(reverse=True)
        return [CHARS.index(c) for c in fallback[0][1]]
    return []


def decode_logits(prebs, mode, beam_size, beam_topk, sample_families=None):
    if mode == "greedy":
        return greedy_decode_logits(prebs)
    if mode == "green_ctc_beam":
        return [
            green_constrained_ctc_beam_decode_single(prebs[i], beam_size=beam_size, topk=beam_topk)
            for i in range(prebs.shape[0])
        ]
    if mode == "family_aware_beam":
        decoded = []
        for i in range(prebs.shape[0]):
            family = sample_families[i] if sample_families is not None and i < len(sample_families) else None
            seq = constrained_ctc_beam_decode_single(prebs[i], family, beam_size=beam_size, topk=beam_topk)
            if seq is None:
                seq = greedy_decode_logits(prebs[i:i+1])[0]
            decoded.append(seq)
        return decoded
    raise ValueError(f"unsupported decode mode: {mode}")


def build_dataset(args):
    test_img_dirs = args.test_img_dirs
    if args.data_mode == "ccpd_board":
        return CCPDBoardDataLoader(
            test_img_dirs.split(","),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.txt_file,
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
            ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
        )
    if args.data_mode == "manifest":
        return UnifiedManifestDataset(
            manifest_path=args.test_img_dirs,
            img_size=args.img_size,
            lpr_max_len=args.lpr_max_len,
            split_filter='test',
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
            ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
        )
    return LPRDataLoader(test_img_dirs.split(","), args.img_size, args.lpr_max_len, txt_file=args.txt_file)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def province_macro_metrics(province_rows):
    if not province_rows:
        return 0.0, 0.0
    exact_scores = []
    first_scores = []
    for row in province_rows.values():
        exact_scores.append(safe_div(row["exact_plate_correct"], row["sample_count"]))
        first_scores.append(safe_div(row["first_char_correct"], row["sample_count"]))
    return float(np.mean(exact_scores)), float(np.mean(first_scores))


def major_non_major_metrics(province_rows):
    if not province_rows:
        return None, 0.0, 0.0, 0.0
    max_count = max(int(row["sample_count"]) for row in province_rows.values())
    major_provinces = sorted(
        province for province, row in province_rows.items() if int(row["sample_count"]) == max_count
    )
    major_count = sum(int(province_rows[province]["sample_count"]) for province in major_provinces)
    major_exact_total = sum(int(province_rows[province]["exact_plate_correct"]) for province in major_provinces)
    total = sum(int(row["sample_count"]) for row in province_rows.values())
    non_major_total = 0
    non_major_exact = 0
    for province, row in province_rows.items():
        if province in major_provinces:
            continue
        non_major_total += int(row["sample_count"])
        non_major_exact += int(row["exact_plate_correct"])
    return (
        "|".join(major_provinces),
        safe_div(major_count, total),
        safe_div(major_exact_total, major_count),
        safe_div(non_major_exact, non_major_total),
    )


def evaluate(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    state = torch.load(args.pretrained_model, map_location=device)
    if args.head_mode == 'multihead':
        net, inferred_cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=args.lpr_max_len,
            phase=False,
            class_num=len(CHARS),
            dropout_rate=args.dropout_rate,
            default_pos0_head_cols=max(4, int(args.pos0_head_cols)),
        )
        if args.enhanced_green_head and inferred_cfg['enhanced_green_head'] != args.enhanced_green_head:
            net = build_lprnet_multihead(
                lpr_max_len=args.lpr_max_len,
                phase=False,
                class_num=len(CHARS),
                dropout_rate=args.dropout_rate,
                enhanced_green_head=args.enhanced_green_head,
                pos0_head_cols=inferred_cfg['pos0_head_cols'],
                pos0_num_classes=inferred_cfg['pos0_num_classes'],
                adapter_families=inferred_cfg['adapter_families'],
            )
            if inferred_cfg['pos0_target_families']:
                net.enable_family_specific_pos0(inferred_cfg['pos0_target_families'], pos0_num_classes=inferred_cfg['pos0_num_classes'])
        net.load_state_dict(state, strict=False)
    else:
        net = build_lprnet(lpr_max_len=args.lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=args.dropout_rate)
        net.load_state_dict(state)
    net.to(device)
    net.eval()

    dataset = build_dataset(args)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(
        dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
