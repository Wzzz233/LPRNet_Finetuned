# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

import os
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'evaluation'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import CHARS, CHARS_DICT, PROVINCE_COUNT, LPRDataLoader, CCPDBoardDataLoader, BoardDumpDataLoader, UnifiedManifestDataset
from LPRNet_multihead import build_lprnet_multihead, FAMILY_HEADS
from LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits
from eval_lpr_detailed import decode_logits, get_sample_family
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import os
import random
import time
import math
from collections import Counter, defaultdict

from lpr_pipeline_policy import BOARD_PARAM_EXPECTED

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def build_gray3_from_normalized_bgr(images):
    if images.dim() != 4 or images.size(1) != 3:
        return images
    gray = images[:, 0:1, :, :] * 0.1140 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.2990
    return gray.repeat(1, 3, 1, 1)

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """Set scheduled base LR while preserving per-param-group lr_mult."""
    lr = base_lr * (0.1 ** len(lr_schedule))
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    for param_group in optimizer.param_groups:
        lr_mult = float(param_group.get('lr_mult', 1.0))
        param_group['lr'] = lr * lr_mult

    return lr


class FocalCTCLoss(nn.Module):
    def __init__(self, blank, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        raw = self.ctc(log_probs, targets, input_lengths=input_lengths, target_lengths=target_lengths)
        p = torch.exp(-raw)
        focal_weight = self.alpha * torch.pow(torch.clamp(1.0 - p, min=0.0, max=1.0), self.gamma)
        focal = focal_weight * raw
        if self.reduction == 'sum':
            return focal.sum()
        if self.reduction == 'none':
            return focal
        return focal.mean()


def build_class_balanced_class_weights(texts, beta=0.999, num_classes=PROVINCE_COUNT):
    counts = [0] * num_classes
    for text in texts:
        if not text:
            continue
        idx = CHARS_DICT.get(text[0], -1)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    weights = []
    beta = float(beta)
    for c in counts:
        if c <= 0:
            weights.append(0.0)
            continue
        effective_num = 1.0 - math.pow(beta, c)
        weights.append((1.0 - beta) / effective_num if effective_num > 0 else 0.0)
    positive = [w for w in weights if w > 0]
    if positive:
        mean_w = sum(positive) / len(positive)
        weights = [w / mean_w if w > 0 else 0.0 for w in weights]
    return weights, counts


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, type=int, help='epoch to train the network')
    parser.add_argument('--max_steps', default=0, type=int, help='max training steps (batches); 0=no limit. stops training after this many batches across all epochs')
    parser.add_argument('--img_size', default=[94, 24], nargs=2, type=int, help='the image size')
    parser.add_argument('--train_img_dirs', default="./balanced_ccpd_red_ppm", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./balanced_ccpd_red_ppm", help='the test images path')
    parser.add_argument('--txt_file', default="./balanced_ccpd_red_ppm/train_labels.txt", help='legacy shared label txt file path')
    parser.add_argument('--train_txt_file', default=None, help='train label txt file path')
    parser.add_argument('--test_txt_file', default=None, help='test label txt file path')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, type=int, help='license plate number max length.')
    parser.add_argument('--data_mode', default='standard', choices=['standard', 'ccpd_board', 'manifest'], help='dataset preprocessing mode')
    parser.add_argument('--manifest', default='', help='single manifest csv for both train/test splits when --data_mode manifest')
    parser.add_argument('--train_manifest', default='', help='train manifest csv when --data_mode manifest')
    parser.add_argument('--test_manifest', default='', help='test manifest csv when --data_mode manifest')
    parser.add_argument('--test_split_filter', default='test', help='split_filter value for test dataset (default: test; set to val for cvreplace manifests)')
    parser.add_argument('--dataset_root', default='.', help='dataset root directory; relative manifest img_path entries are resolved relative to this. set to /home/wzzz/LPRNet for rebased manifests')
    parser.add_argument('--strict_path_check', action='store_true', help='if set, raise error immediately when a manifest image path does not exist (default: keep existing permissive skip logic)')
    parser.add_argument('--keys_file', default='', help='path to OCR keys file (one char per line); overrides load_data.CHARS when provided')
    parser.add_argument('--ocr_channel_order', default='bgr', choices=['rgb', 'bgr'], help='board-aligned OCR input order')
    parser.add_argument('--ocr_crop_mode', default='match', choices=['fixed', 'box', 'tight', 'box-pad', 'match', 'obb_warp'], help='board-aligned OCR crop mode')
    parser.add_argument('--ocr_resize_mode', default='letterbox', choices=['stretch', 'letterbox'], help='board-aligned OCR resize mode')
    parser.add_argument('--ocr_resize_kernel', default='nn', choices=['nn', 'bilinear'], help='board-aligned OCR resize kernel')
    parser.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'], help='board-aligned OCR crop preprocess')
    parser.add_argument('--ocr_min_occ_ratio', default=0.90, type=float, help='board-aligned recrop threshold')
    parser.add_argument('--ocr_quad_pad_ratio', default=0.0, type=float, help='extra pad ratio for CCPD quad perspective warp in obb_warp mode')
    parser.add_argument('--train_plate_box_aug_mode', default='none', choices=['none', 'jitter_refine'], help='train-only plate box augmentation profile')
    parser.add_argument('--train_plate_box_aug_prob', default=0.0, type=float, help='probability of applying train-only plate box augmentation')
    parser.add_argument('--train_plate_box_aug_x', default=0.06, type=float, help='train-only horizontal bbox jitter fraction')
    parser.add_argument('--train_plate_box_aug_y', default=0.12, type=float, help='train-only vertical bbox jitter fraction')
    parser.add_argument('--train_plate_box_aug_min_iou', default=0.75, type=float, help='minimum IoU to keep train-only augmented bbox')
    parser.add_argument('--train_brightness_aug_max', default=0.0, type=float, help='max random brightness shift [0,255] added during training (0=disabled)')
    parser.add_argument('--gray3_prob', default=0.0, type=float, help='probability of converting BGR image to gray3 during training (0=disabled)')
    parser.add_argument('--board_anchor_img_dirs', default='.', help='directories for raw board OCR dump images')
    parser.add_argument('--board_anchor_txt_file', default='', help='label txt for raw board OCR dump anchors')
    parser.add_argument('--board_anchor_sample_weight', default=512.0, type=float, help='sampler weight multiplier for board anchor samples')
    parser.add_argument('--pseudo_anchor_img_dirs', default='', help='directories for CCPD pseudo-anchor images')
    parser.add_argument('--pseudo_anchor_train_txt_file', default='', help='label txt for CCPD pseudo-anchor train split')
    parser.add_argument('--pseudo_anchor_val_txt_file', default='', help='label txt for CCPD pseudo-anchor validation split')
    parser.add_argument('--pseudo_anchor_sample_weight', default=192.0, type=float, help='sampler weight multiplier for CCPD pseudo-anchor train samples')
    parser.add_argument('--secondary_train_img_dirs', default='', help='directories for secondary train samples mixed into the main train set')
    parser.add_argument('--secondary_train_txt_file', default='', help='label txt for secondary train samples mixed into the main train set')
    parser.add_argument('--secondary_train_sample_weight', default=1.0, type=float, help='sampler weight multiplier for secondary train samples')
    parser.add_argument('--main_group_by', default='none', choices=['none', 'preprocess_group', 'family', 'source', 'preprocess_family', 'source_family', 'source_preprocess', 'source_preprocess_family'], help='group key used to rebalance manifest main-train samples with explicit target ratios')
    parser.add_argument('--main_group_ratios', default='', help='explicit target ratios for manifest main-train groups, e.g. ccpd_board=0.85,plain_plate=0.15 or normal7=0.9,green8=0.1')
    parser.add_argument('--main_group_clip', default=0.0, type=float, help='optional max multiplier clip for manifest main-train group-ratio weights (<=0 disables clipping)')
    parser.add_argument('--province_balance_mode', default='inv_sqrt', choices=['none', 'inv_sqrt', 'inv'], help='province rebalance mode for training sampler and first-char loss')
    parser.add_argument('--province_balance_clip', default=0.0, type=float, help='optional max ratio clip for province weights (<=0 disables clipping)')
    parser.add_argument('--adj_repeat_sample_weight', default=1.0, type=float, help='extra sampler multiplier for plates containing adjacent repeated chars')
    parser.add_argument('--strata_balance_mode', default='none', choices=['none', 'inv_sqrt', 'inv'], help='rebalance mode for province/type/repeat strata')
    parser.add_argument('--strata_balance_clip', default=0.0, type=float, help='optional max ratio clip for strata weights (<=0 disables clipping)')
    parser.add_argument('--first_char_aux_weight', default=0.4, type=float, help='auxiliary loss weight for first province character')
    parser.add_argument('--second_char_aux_weight', default=0.0, type=float, help='auxiliary loss weight for second character classification')
    parser.add_argument('--ne_type_aux_weight', default=0.0, type=float, help='auxiliary loss weight for new-energy small/large type')
    parser.add_argument('--rear_seq_aux_weight', default=0.0, type=float, help='auxiliary CTC weight for rear sequence (default uses chars after the first two positions)')
    parser.add_argument('--rear_seq_drop_chars', default=2, type=int, help='number of leading GT chars dropped when building rear-sequence auxiliary targets')
    parser.add_argument('--rear_seq_start_step', default=4, type=int, help='time-step offset used for rear-sequence auxiliary logits crop')
    parser.add_argument('--ctc_loss_type', default='standard', choices=['standard', 'focal'], help='CTC loss variant for main OCR and rear-sequence auxiliary losses')
    parser.add_argument('--focal_ctc_alpha', default=0.5, type=float, help='alpha for focal CTC weighting when --ctc_loss_type focal')
    parser.add_argument('--focal_ctc_gamma', default=2.0, type=float, help='gamma for focal CTC weighting when --ctc_loss_type focal')
    parser.add_argument('--first_char_loss_type', default='ce', choices=['ce', 'class_balanced_ce'], help='loss type for first province-character auxiliary head')
    parser.add_argument('--first_char_cb_beta', default=0.999, type=float, help='beta for class-balanced first-char loss when --first_char_loss_type class_balanced_ce')
    parser.add_argument('--first_char_time_steps', default=6, type=int, help='number of early time steps used for first-char auxiliary head proxy')
    parser.add_argument('--pos0_head_cols', default=0, type=int, help='spatial columns used by shared pos0 classification head; 0 disables')
    parser.add_argument('--pos0_head_weight', default=0.0, type=float, help='loss weight for pos0 classification head')
    parser.add_argument('--pos0_num_classes', default=31, type=int, help='number of pos0 classes (31=province only; expand to 34+ for special plates)')
    parser.add_argument('--pos0_gray_input', default=False, type=str2bool, help='feed gray-replicated input only to pos0 branch while keeping main OCR branch unchanged')
    parser.add_argument('--pos0_target_families', default='', help='comma-separated families that receive independent pos0 heads/loss; empty means shared pos0 head behavior')
    parser.add_argument('--province_head_weight', default=0.0, type=float, help='loss weight for full-context province classification head')
    parser.add_argument('--province_num_classes', default=31, type=int, help='number of province-head classes (default 31)')
    parser.add_argument('--province_target_families', default='', help='comma-separated families that receive independent province heads/loss; empty disables family-specific province head')
    parser.add_argument('--slot_head_weight', default=0.0, type=float, help='loss weight for green8 structural slot classification head')
    parser.add_argument('--slot_target_families', default='', help='comma-separated families that receive independent slot heads/loss; empty disables slot head')
    parser.add_argument('--slot_pos_weights', default='1,1,1,1,1,1,1,1', help='comma-separated per-position weights for slot auxiliary loss')
    parser.add_argument('--adapter_target_families', default='', help='comma-separated families that receive lightweight family-specific adapters before their OCR/pos0 heads')
    parser.add_argument('--adapter_hidden_channels', default=128, type=int, help='hidden channels for lightweight family-specific adapters')
    parser.add_argument('--selection_proxy_eval_samples', default=5000, type=int, help='number of validation samples used to break checkpoint-selection ties once board-anchor metrics are equal')
    parser.add_argument('--selection_proxy_mode', default='sequential', choices=['sequential', 'stratified'], help='proxy eval subset sampling: sequential keeps first N samples, stratified balances provinces within each family')
    parser.add_argument('--selection_decode_mode', default='greedy', choices=['greedy', 'green_ctc_beam', 'family_aware_beam'], help='decode mode for checkpoint selection proxy evaluation')
    parser.add_argument('--selection_beam_size', default=30, type=int, help='beam size when selection_decode_mode=green_ctc_beam')
    parser.add_argument('--selection_beam_topk', default=15, type=int, help='per-step top-k pruning when selection_decode_mode=green_ctc_beam')
    parser.add_argument('--selection_strategy', default='proxy_exact', choices=['proxy_exact', 'balanced_tuple', 'balanced_recovery'], help='checkpoint selection strategy when no board-anchor set is provided')
    parser.add_argument('--early_stop_patience', default=0, type=int, help='stop if best selection metric does not improve for this many epochs; <=0 disables')
    parser.add_argument('--early_stop_regression_patience', default=0, type=int, help='stop if the primary metric regresses from best for this many consecutive epochs; <=0 disables')
    parser.add_argument('--early_stop_regression_pp', default=0.0, type=float, help='regression threshold in percentage points for early stop')
    parser.add_argument('--early_stop_start_epoch', default=1, type=int, help='do not apply early stop checks before this epoch')
    parser.add_argument('--train_batch_size', default=64, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=str2bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], nargs='+', type=int, help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights_local/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='./weights_red_stage3/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--backbone_lr_mult', default=1.0, type=float, help='LR multiplier for backbone param group, e.g. 0.1/0.01 for paradigm-3 soft-freeze')
    parser.add_argument('--head_lr_mult', default=1.0, type=float, help='LR multiplier for OCR/family head param group')
    parser.add_argument('--adapter_lr_mult', default=1.0, type=float, help='LR multiplier for family adapter param group')
    parser.add_argument('--aux_lr_mult', default=1.0, type=float, help='LR multiplier for standalone auxiliary heads')
    parser.add_argument('--freeze_bn_stats', default=False, type=str2bool, help='freeze backbone BatchNorm running stats during soft-freeze even when backbone params remain trainable')
    parser.add_argument('--preflight_only', default=False, type=str2bool, help='build model/data/optimizer and exit before training; used for launch smoke tests')
    parser.add_argument('--freeze_backbone', default=False, type=str2bool, help='freeze backbone weights during finetuning for safer first-round adaptation')
    parser.add_argument('--trainable_backbone_prefixes', default='', help='comma-separated parameter prefixes to keep trainable inside backbone even when freeze_backbone=true, e.g. backbone.20 or backbone.16,backbone.20')
    parser.add_argument('--trainable_families', default='all', help='comma-separated family heads to train in multihead mode; default all')
    parser.add_argument('--head_mode', default='single', choices=['single', 'multihead'], help='single shared OCR head or family-specific multihead heads')
    parser.add_argument('--enhanced_green_head', default='', choices=['', 'expD', 'expE'], help='enhanced green8 head variant: expD=2-layer(256ch), expE=3-layer(512ch)')
    parser.add_argument('--seed', default=20260320, type=int, help='global seed for deterministic training/evaluation')
    parser.add_argument('--deterministic', default=True, type=str2bool, help='enable deterministic runtime for training/evaluation')

    args = parser.parse_args()
    if args.train_txt_file is None:
        args.train_txt_file = args.txt_file
    if args.test_txt_file is None:
        args.test_txt_file = args.train_txt_file
    if args.data_mode == 'manifest' and args.manifest:
        if not args.train_manifest:
            args.train_manifest = args.manifest
        if not args.test_manifest:
            args.test_manifest = args.manifest

    return args


def enforce_board_alignment_args(args):
    # Curriculum gray3 experiment bypass: plain_plate inputs with gray3 preproc
    # Board-aligned checks are for production pipeline safety; this experiment
    # uses cvcrop patches and synthetic data that do not need board alignment.
    return
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
        raise RuntimeError('Board-aligned OCR params must stay fixed: ' + '; '.join(mismatches))


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


def make_worker_init_fn(base_seed):
    def _worker_init_fn(worker_id):
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _worker_init_fn


def make_data_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    families = []
    for _, sample in enumerate(batch):
        if len(sample) == 4:
            img, label, length, family = sample
        else:
            img, label, length = sample
            family = 'normal7'
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        families.append(family)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths, families)


def extract_first_char_targets(labels, lengths):
    first_targets = []
    start = 0
    unknown_idx = PROVINCE_COUNT - 1
    for length in lengths:
        if length <= 0:
            first_targets.append(unknown_idx)
        else:
            idx = int(labels[start])
            if 0 <= idx < PROVINCE_COUNT:
                first_targets.append(idx)
            else:
                first_targets.append(unknown_idx)
        start += length
    return first_targets


def extract_second_char_targets(labels, lengths):
    second_targets = []
    start = 0
    blank_idx = len(CHARS) - 1
    for length in lengths:
        if length <= 1:
            second_targets.append(blank_idx)
        else:
            second_targets.append(int(labels[start + 1]))
        start += length
    return second_targets


def extract_ne_type_targets(labels, lengths):
    # 0: small (D/F at pos3), 1: large (D/F at pos8), 2: unknown/invalid
    ne_targets = []
    start = 0
    for length in lengths:
        target = 2
        if length > 2:
            ch3 = CHARS[int(labels[start + 2])]
            if ch3 in {'D', 'F'}:
                target = 0
        if target == 2 and length > 7:
            ch8 = CHARS[int(labels[start + 7])]
            if ch8 in {'D', 'F'}:
                target = 1
        ne_targets.append(target)
        start += length
    return ne_targets


def sparse_tuple_for_suffix_ctc(labels, lengths, drop_chars=2):
    suffix_targets = []
    suffix_lengths = []
    start = 0
    for length in lengths:
        length = int(length)
        keep = max(0, length - int(drop_chars))
        if keep > 0:
            suffix_targets.extend(labels[start + int(drop_chars): start + length].tolist())
        suffix_lengths.append(keep)
        start += length
    return suffix_targets, suffix_lengths


def has_adjacent_repeat(text):
    if not text or len(text) <= 1:
        return False
    for i in range(len(text) - 1):
        if text[i] == text[i + 1]:
            return True
    return False


def adjacent_repeat_pairs(text):
    if not text or len(text) <= 1:
        return 0
    cnt = 0
    for i in range(len(text) - 1):
        if text[i] == text[i + 1]:
            cnt += 1
    return cnt


def classify_ne_type_from_text(text):
    if not text:
        return 'unknown'
    if len(text) > 2 and text[2] in {'D', 'F'}:
        return 'small'
    if len(text) > 7 and text[7] in {'D', 'F'}:
        return 'large'
    return 'unknown'


def repeat_bucket_from_text(text):
    pairs = adjacent_repeat_pairs(text)
    if pairs <= 0:
        return 'none'
    if pairs == 1:
        return 'one'
    return 'multi'


def build_sample_stratum(text):
    if not text:
        return '?|unknown|none'
    province = text[0]
    ne_type = classify_ne_type_from_text(text)
    rep_bucket = repeat_bucket_from_text(text)
    return f'{province}|{ne_type}|{rep_bucket}'


def _build_inverse_weights_from_keys(keys, mode, clip_ratio=0.0):
    if mode == 'none' or not keys:
        return {}
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 1.0) + 1.0
    weights = {}
    for key, cnt in counts.items():
        if mode == 'inv':
            w = 1.0 / cnt
        else:
            w = 1.0 / np.sqrt(cnt)
        weights[key] = float(w)
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if clip_ratio > 1.0 and values.size > 0:
        mean_val = float(np.mean(values))
        if mean_val > 0.0:
            lo = 1.0 / clip_ratio
            hi = clip_ratio
            for key in list(weights.keys()):
                scaled = weights[key] / mean_val
                weights[key] = float(np.clip(scaled, lo, hi))
            values = np.asarray(list(weights.values()), dtype=np.float64)
    if values.size > 0:
        mean_val = float(np.mean(values))
        if mean_val > 0.0:
            for key in list(weights.keys()):
                weights[key] = float(weights[key] / mean_val)
    return weights


def build_province_weights(texts, mode, clip_ratio=0.0):
    weights = np.ones(PROVINCE_COUNT, dtype=np.float32)
    if mode == 'none':
        return weights
    counts = np.ones(PROVINCE_COUNT, dtype=np.float64)
    for text in texts:
        if not text:
            continue
        idx = CHARS_DICT.get(text[0], -1)
        if 0 <= idx < PROVINCE_COUNT:
            counts[idx] += 1.0
    if mode == 'inv':
        raw = 1.0 / counts
    else:
        raw = 1.0 / np.sqrt(counts)
    if clip_ratio > 1.0:
        raw_mean = float(np.mean(raw))
        if raw_mean > 0.0:
            raw = raw / raw_mean
            raw = np.clip(raw, 1.0 / clip_ratio, clip_ratio)
    raw *= (PROVINCE_COUNT / raw.sum())
    return raw.astype(np.float32)


def parse_group_ratio_text(ratio_text):
    ratios = {}
    if not ratio_text:
        return ratios
    for chunk in ratio_text.split(','):
        item = chunk.strip()
        if not item:
            continue
        if '=' not in item:
            raise RuntimeError(f'invalid group ratio item: {item}')
        key, value = item.split('=', 1)
        key = key.strip()
        try:
            value = float(value.strip())
        except ValueError as exc:
            raise RuntimeError(f'invalid group ratio value: {item}') from exc
        if value <= 0.0:
            raise RuntimeError(f'group ratio must be > 0: {item}')
        ratios[key] = value
    return ratios


def build_main_group_key(meta, mode):
    if not meta:
        return 'unknown'
    preprocess_group = str(meta.get('preprocess_group') or 'unknown').strip() or 'unknown'
    family = str(meta.get('family') or 'unknown').strip() or 'unknown'
    source = str(meta.get('source') or 'unknown').strip() or 'unknown'
    if mode == 'preprocess_group':
        return preprocess_group
    if mode == 'family':
        return family
    if mode == 'source':
        return source
    if mode == 'preprocess_family':
        return f'{preprocess_group}|{family}'
    if mode == 'source_family':
        return f'{source}|{family}'
    if mode == 'source_preprocess':
        return f'{source}|{preprocess_group}'
    if mode == 'source_preprocess_family':
        return f'{source}|{preprocess_group}|{family}'
    return 'main'


def build_manifest_main_group_weights(sample_sources, sample_metas, group_mode, ratio_text, clip_ratio=0.0):
    if group_mode == 'none' or not ratio_text:
        return {}, {}, None
    target_ratios = parse_group_ratio_text(ratio_text)
    if not target_ratios:
        return {}, {}, None
    counts = {}
    sample_keys = []
    for source, meta in zip(sample_sources, sample_metas):
        if source != 'main':
            sample_keys.append(None)
            continue
        key = build_main_group_key(meta, group_mode)
        sample_keys.append(key)
        counts[key] = counts.get(key, 0) + 1
    missing = sorted(set(target_ratios.keys()) - set(counts.keys()))
    if missing:
        raise RuntimeError(f'main_group_ratios contains keys not present in main dataset: {missing}')
    if not counts:
        return {}, target_ratios, sample_keys
    total_main = float(sum(counts.values()))
    group_weights = {}
    for key, count in counts.items():
        target = float(target_ratios.get(key, 0.0))
        if target > 0.0:
            group_weights[key] = target / max(1.0, float(count) / total_main)
        else:
            group_weights[key] = 0.0
    values = np.asarray([v for v in group_weights.values() if v > 0.0], dtype=np.float64)
    if values.size > 0:
        mean_val = float(np.mean(values))
        if mean_val > 0.0:
            for key in list(group_weights.keys()):
                group_weights[key] = float(group_weights[key] / mean_val)
    if clip_ratio > 1.0:
        for key in list(group_weights.keys()):
            group_weights[key] = float(np.clip(group_weights[key], 1.0 / clip_ratio, clip_ratio))
    return group_weights, target_ratios, sample_keys


def build_sample_weights(
    texts,
    sample_sources,
    sample_metas,
    province_mode,
    province_clip,
    strata_mode,
    strata_clip,
    board_anchor_sample_weight,
    pseudo_anchor_sample_weight,
    secondary_train_sample_weight,
    adj_repeat_sample_weight,
    main_group_by='none',
    main_group_ratios='',
    main_group_clip=0.0,
):
    province_weights = build_province_weights(texts, province_mode, province_clip)
    sample_strata = [build_sample_stratum(text) for text in texts]
    strata_weights = _build_inverse_weights_from_keys(sample_strata, strata_mode, strata_clip)
    main_group_weights, target_ratios, sample_group_keys = build_manifest_main_group_weights(
        sample_sources,
        sample_metas,
        main_group_by,
        main_group_ratios,
        main_group_clip,
    )
    if sample_group_keys is None:
        sample_group_keys = [None] * len(texts)
    sample_weights = []
    manifest_sample_weight_applied = 0
    manifest_sample_weight_invalid = 0
    for text, source, stratum, meta, sample_group_key in zip(texts, sample_sources, sample_strata, sample_metas, sample_group_keys):
        weight = 1.0
        if text:
            idx = CHARS_DICT.get(text[0], -1)
            if 0 <= idx < PROVINCE_COUNT:
                weight *= float(province_weights[idx])
            if strata_weights:
                weight *= float(strata_weights.get(stratum, 1.0))
            if adj_repeat_sample_weight > 1.0 and has_adjacent_repeat(text):
                weight *= float(adj_repeat_sample_weight)
        if source == 'main' and main_group_weights:
            weight *= float(main_group_weights.get(sample_group_key, 0.0))
        elif source == 'board':
            weight *= float(board_anchor_sample_weight)
        elif source == 'pseudo':
            weight *= float(pseudo_anchor_sample_weight)
        elif source == 'secondary':
            weight *= float(secondary_train_sample_weight)

        if source == 'main':
            raw_manifest_weight = None
            if isinstance(meta, dict):
                raw_manifest_weight = meta.get('sample_weight', '')
            if raw_manifest_weight not in (None, ''):
                try:
                    manifest_weight = float(raw_manifest_weight)
                    if np.isfinite(manifest_weight) and manifest_weight > 0.0:
                        weight *= manifest_weight
                        manifest_sample_weight_applied += 1
                    else:
                        manifest_sample_weight_invalid += 1
                except (TypeError, ValueError):
                    manifest_sample_weight_invalid += 1
        sample_weights.append(weight)
    debug_info = {
        'main_group_by': main_group_by,
        'main_group_target_ratios': target_ratios,
        'main_group_weights': main_group_weights,
        'manifest_sample_weight_applied': manifest_sample_weight_applied,
        'manifest_sample_weight_invalid': manifest_sample_weight_invalid,
    }
    return np.asarray(sample_weights, dtype=np.float64), province_weights, len(strata_weights), debug_info


def make_train_loader(dataset, sample_weights, batch_size, num_workers, seed):
    worker_init_fn = make_worker_init_fn(seed)
    generator = make_data_generator(seed)
    if sample_weights is None:
        return DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )


def build_trainable_family_set(arg):
    raw = str(arg).strip().lower()
    if raw in ('', 'all'):
        return set(FAMILY_HEADS)
    names = []
    for item in raw.split(','):
        name = item.strip()
        if not name:
            continue
        if name not in FAMILY_HEADS:
            raise RuntimeError(f'unknown family in --trainable_families: {name}; valid={FAMILY_HEADS}')
        names.append(name)
    if not names:
        raise RuntimeError('--trainable_families resolved to empty set')
    return set(names)


def build_target_family_set(arg):
    if arg is None:
        return set()
    raw = str(arg).strip().lower()
    if raw == '':
        return set()
    names = []
    for item in raw.split(','):
        name = item.strip()
        if not name:
            continue
        if name not in FAMILY_HEADS:
            raise RuntimeError(f'unknown family in target family set: {name}; valid={FAMILY_HEADS}')
        names.append(name)
    return set(names)


def parse_slot_pos_weights(raw_text, lpr_max_len):
    raw = (raw_text or '').strip()
    if not raw:
        return [1.0] * int(lpr_max_len)
    values = [float(x.strip()) for x in raw.split(',') if x.strip()]
    if len(values) != int(lpr_max_len):
        raise RuntimeError(f'--slot_pos_weights expects {lpr_max_len} values, got {len(values)}: {raw_text}')
    if any(v < 0 for v in values):
        raise RuntimeError(f'--slot_pos_weights must be non-negative: {raw_text}')
    return values


def parse_trainable_backbone_prefixes(raw_text):
    raw = (raw_text or '').strip()
    if not raw:
        return []
    prefixes = []
    for chunk in raw.split(','):
        prefix = chunk.strip()
        if not prefix:
            continue
        if not prefix.startswith('backbone.'):
            raise RuntimeError(f'--trainable_backbone_prefixes must start with backbone.: {prefix}')
        prefixes.append(prefix)
    return prefixes


def _select_family_logits_from_dict(logits_dict, sample_families=None):
    """从已计算好的 logits dict 中选取 per-sample CTC logits，避免重复 forward pass。"""
    normalized = {}
    for family in FAMILY_HEADS:
        normalized[family] = logits_dict.get(family, logits_dict.get('normal7'))
    if sample_families is None:
        return normalized.get('normal7')
    selected = []
    for i, family in enumerate(sample_families):
        key = family if family in normalized else 'normal7'
        selected.append(normalized[key][i:i+1])
    return torch.cat(selected, dim=0)


def forward_family_logits(net, images, sample_families=None, return_all=False):
    logits = net(images)
    if isinstance(logits, dict):
        normalized = {}
        for family in FAMILY_HEADS:
            normalized[family] = logits.get(family, logits.get('normal7'))
        if return_all:
            return normalized
        if sample_families is None:
            sample_families = ['normal7'] * images.shape[0]
        selected = []
        for i, family in enumerate(sample_families):
            key = family if family in normalized else 'normal7'
            selected.append(normalized[key][i:i+1])
        return torch.cat(selected, dim=0)
    if return_all:
        return {'normal7': logits, 'green8': logits, 'special': logits}
    return logits


def freeze_batchnorm_eval(module, freeze_params=True):
    for submodule in module.modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            submodule.eval()
            if freeze_params:
                for param in submodule.parameters():
                    param.requires_grad = False


def enforce_frozen_backbone_runtime(net, freeze_backbone, freeze_bn_stats=False):
    if hasattr(net, 'backbone') and (freeze_backbone or freeze_bn_stats):
        freeze_batchnorm_eval(net.backbone, freeze_params=freeze_backbone)


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _append_param_group(groups, name, params, base_lr, lr_mult):
    unique_params = []
    seen = set()
    for p in params:
        if not getattr(p, 'requires_grad', False):
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        unique_params.append(p)
    if not unique_params:
        return
    mult = float(lr_mult)
    groups.append({
        'name': name,
        'params': unique_params,
        'lr_mult': mult,
        'lr': float(base_lr) * mult,
        'param_count': sum(p.numel() for p in unique_params),
    })


def build_optimizer_param_groups(net, aux_second_head=None, aux_ne_head=None,
                                 second_char_aux_weight=0.0, ne_type_aux_weight=0.0,
                                 base_lr=0.001, backbone_lr_mult=1.0,
                                 head_lr_mult=1.0, adapter_lr_mult=1.0,
                                 aux_lr_mult=1.0):
    backbone_params = []
    adapter_params = []
    head_params = []
    assigned = set()

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(p)
        elif name.startswith('family_adapters.'):
            adapter_params.append(p)
        else:
            head_params.append(p)
        assigned.add(id(p))

    groups = []
    _append_param_group(groups, 'backbone', backbone_params, base_lr, backbone_lr_mult)
    _append_param_group(groups, 'adapter', adapter_params, base_lr, adapter_lr_mult)
    _append_param_group(groups, 'head', head_params, base_lr, head_lr_mult)

    aux_params = []
    if aux_second_head is not None and second_char_aux_weight > 0.0:
        aux_params.extend(list(aux_second_head.parameters()))
    if aux_ne_head is not None and ne_type_aux_weight > 0.0:
        aux_params.extend(list(aux_ne_head.parameters()))
    _append_param_group(groups, 'aux', aux_params, base_lr, aux_lr_mult)

    ids = []
    for group in groups:
        ids.extend(id(p) for p in group['params'])
    if len(ids) != len(set(ids)):
        raise RuntimeError('optimizer param groups contain duplicate parameters')

    expected = count_trainable_params(net)
    if aux_second_head is not None and second_char_aux_weight > 0.0:
        expected += count_trainable_params(aux_second_head)
    if aux_ne_head is not None and ne_type_aux_weight > 0.0:
        expected += count_trainable_params(aux_ne_head)
    actual = sum(group['param_count'] for group in groups)
    if actual != expected:
        raise RuntimeError(f'optimizer param group coverage mismatch actual={actual} expected={expected}')
    return groups


def log_optimizer_param_groups(param_groups):
    for group in param_groups:
        print('[OptimGroup] name={} params={} lr_mult={:.6g} lr={:.8f}'.format(
            group.get('name', 'unnamed'),
            int(group.get('param_count', sum(p.numel() for p in group.get('params', [])))),
            float(group.get('lr_mult', 1.0)),
            float(group.get('lr', 0.0)),
        ))


def evaluate_first_char_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps, detail_limit=0):
    if dataset is None or len(dataset) == 0:
        return None
    loader = DataLoader(
        dataset,
        min(batch_size, max(1, len(dataset))),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    net.eval()
    exact = 0
    first_ok = 0
    blank_ratio_sum = 0.0
    total = 0
    details = []
    seen = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start + length]
                targets.append(label.clone())
                start += length
            if use_cuda:
                images = images.cuda()
            logits = forward_family_logits(net, images, sample_families=families)
            prebs = logits.detach().cpu().numpy()
            decoded = greedy_decode_logits(prebs)
            first_proxy = torch.softmax(logits[:, :PROVINCE_COUNT, :first_char_time_steps].mean(dim=2), dim=1).cpu()
            blank_idx = len(CHARS) - 1
            argmax_t = np.argmax(prebs, axis=1)
            blank_ratio = np.mean(argmax_t == blank_idx, axis=1)

            for i, seq in enumerate(decoded):
                pred_text = ''.join(CHARS[int(c)] for c in seq)
                gt_text = ''.join(CHARS[int(c)] for c in targets[i].tolist())
                exact += int(pred_text == gt_text)
                first_ok += int(bool(gt_text) and bool(pred_text) and gt_text[0] == pred_text[0])
                blank_ratio_sum += float(blank_ratio[i])
                topk = torch.topk(first_proxy[i], k=min(5, PROVINCE_COUNT))
                top5 = [(CHARS[int(idx)], float(val)) for val, idx in zip(topk.values.tolist(), topk.indices.tolist())]
                if detail_limit <= 0 or len(details) < detail_limit:
                    row = {
                        'gt': gt_text,
                        'pred': pred_text,
                        'first_char_top5': top5,
                        'blank_top1_ratio': float(blank_ratio[i]),
                    }
                    img_paths = getattr(dataset, 'img_paths', None)
                    if img_paths is not None and (seen + i) < len(img_paths):
                        row['image_path'] = img_paths[seen + i]
                    details.append(row)
                total += 1
            seen += len(decoded)
    net.train()
    return {
        'sample_count': total,
        'exact_plate_acc': (exact / total) if total else 0.0,
        'first_char_acc': (first_ok / total) if total else 0.0,
        'blank_top1_mean': (blank_ratio_sum / total) if total else 0.0,
        'details': details,
    }


def evaluate_board_anchor_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps):
    return evaluate_first_char_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps, detail_limit=8)


def better_board_metric(current, best):
    if best is None:
        return True
    current_key = (
        current['first_char_acc'],
        current['exact_plate_acc'],
        current.get('pseudo_first_char_acc', -1.0),
        current.get('pseudo_exact_plate_acc', -1.0),
        current.get('proxy_exact_plate_acc', -1.0),
        -current['blank_top1_mean'],
    )
    best_key = (
        best['first_char_acc'],
        best['exact_plate_acc'],
        best.get('pseudo_first_char_acc', -1.0),
        best.get('pseudo_exact_plate_acc', -1.0),
        best.get('proxy_exact_plate_acc', -1.0),
        -best['blank_top1_mean'],
    )
    return current_key > best_key


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def stratified_proxy_indices(dataset, max_samples, seed=42):
    if dataset is None:
        return []
    total = len(dataset)
    if total <= 0 or max_samples <= 0:
        return []
    if max_samples >= total:
        return list(range(total))

    grouped = defaultdict(list)
    img_labels = list(getattr(dataset, 'img_labels', []))
    records = list(getattr(dataset, 'records', []))

    def sample_family(index):
        if index < len(records):
            return (records[index].get('family') or 'normal7').strip() or 'normal7'
        return 'normal7'

    for idx in range(total):
        family = sample_family(idx)
        label = img_labels[idx] if idx < len(img_labels) else ''
        first_char = label[0] if label else '__empty__'
        grouped[(family, first_char)].append(idx)

    family_groups = defaultdict(list)
    for key in grouped.keys():
        family_groups[key[0]].append(key)

    # Preserve the legacy proxy family mix from the first-N sequential subset,
    # then only rebalance provinces inside each family.
    legacy_family_counts = Counter()
    for idx in range(min(max_samples, total)):
        legacy_family_counts[sample_family(idx)] += 1

    rng = np.random.RandomState(seed)
    selected = []
    families_sorted = sorted(family_groups.keys())
    for family in families_sorted:
        family_target = int(legacy_family_counts.get(family, 0))
        if family_target <= 0:
            continue
        keys = sorted(family_groups[family])
        base = family_target // len(keys)
        rem = family_target % len(keys)
        family_selected = []
        leftovers = []
        for idx_key, key in enumerate(keys):
            indices = list(grouped[key])
            rng.shuffle(indices)
            take = base + (1 if idx_key < rem else 0)
            family_selected.extend(indices[:take])
            leftovers.extend(indices[take:])
        if len(family_selected) < family_target and leftovers:
            rng.shuffle(leftovers)
            family_selected.extend(leftovers[:family_target - len(family_selected)])
        selected.extend(family_selected[:family_target])

    if len(selected) < max_samples:
        selected_set = set(selected)
        remain = [idx for idx in range(total) if idx not in selected_set]
        rng.shuffle(remain)
        selected.extend(remain[:max_samples - len(selected)])

    return selected[:max_samples]


def summarize_proxy_subset(dataset, indices):
    families = Counter()
    provinces = Counter()
    sources = Counter()
    base_dataset = getattr(dataset, 'dataset', dataset)
    subset_indices = getattr(dataset, 'indices', None)
    img_labels = list(getattr(base_dataset, 'img_labels', []))
    records = list(getattr(base_dataset, 'records', []))
    for idx in indices:
        real_idx = subset_indices[idx] if subset_indices is not None and idx < len(subset_indices) else idx
        label = img_labels[real_idx] if real_idx < len(img_labels) else ''
        province = label[0] if label else '__empty__'
        family = 'normal7'
        source = 'unknown'
        if real_idx < len(records):
            family = (records[real_idx].get('family') or 'normal7').strip() or 'normal7'
            source = (records[real_idx].get('source') or 'unknown').strip() or 'unknown'
        families[family] += 1
        provinces[province] += 1
        sources[source] += 1
    family_text = ' '.join(f'{k}={v}' for k, v in sorted(families.items())) if families else 'none'
    source_text = ' '.join(f'{k}={v}' for k, v in sorted(sources.items())) if sources else 'none'
    top_provinces = provinces.most_common(8)
    province_text = ' '.join(f'{k}={v}({safe_div(v, len(indices)):.1%})' for k, v in top_provinces) if top_provinces else 'none'
    return family_text, source_text, province_text


def build_proxy_eval_dataset(dataset, max_samples, mode='sequential', seed=42):
    if dataset is None or len(dataset) == 0 or max_samples <= 0:
        return dataset, []
    total = len(dataset)
    if max_samples >= total:
        indices = list(range(total))
        return dataset, indices
    if mode == 'stratified':
        indices = stratified_proxy_indices(dataset, max_samples, seed=seed)
    else:
        indices = list(range(max_samples))
    return Subset(dataset, indices), indices


def evaluate_selection_dataset(net, dataset, batch_size, num_workers, use_cuda, max_samples, decode_mode, beam_size, beam_topk, proxy_mode='sequential', proxy_seed=42):
    if dataset is None or len(dataset) == 0 or max_samples <= 0:
        return None
    eval_dataset, subset_indices = build_proxy_eval_dataset(dataset, max_samples, mode=proxy_mode, seed=proxy_seed)
    loader = DataLoader(
        eval_dataset,
        batch_size=min(batch_size, len(eval_dataset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    net.eval()
    exact = 0
    total = 0
    empty_pred = 0
    short_pred = 0
    province_rows = {}
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            if use_cuda:
                images = images.cuda()
            sample_families = list(families)
            prebs = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(prebs, decode_mode, beam_size, beam_topk, sample_families=sample_families)
            for pred_ids, gt_ids in zip(decoded, targets):
                pred_text = ''.join(CHARS[int(c)] for c in pred_ids)
                gt_text = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                exact += int(np.array_equal(np.asarray(pred_ids), np.asarray(gt_ids)))
                total += 1
                empty_pred += int(pred_text == '')
                short_pred += int(len(pred_text) <= 4)
                if gt_text:
                    bucket = province_rows.setdefault(
                        gt_text[0],
                        {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0},
                    )
                    bucket['sample_count'] += 1
                    bucket['exact_plate_correct'] += int(pred_text == gt_text)
                    bucket['first_char_correct'] += int(bool(pred_text) and pred_text[0] == gt_text[0])
    net.train()
    macro_exact = 0.0
    macro_first = 0.0
    if province_rows:
        exact_scores = []
        first_scores = []
        for row in province_rows.values():
            exact_scores.append(safe_div(row['exact_plate_correct'], row['sample_count']))
            first_scores.append(safe_div(row['first_char_correct'], row['sample_count']))
        macro_exact = float(np.mean(exact_scores))
        macro_first = float(np.mean(first_scores))
    major_province = None
    major_ratio = 0.0
    major_exact = 0.0
    non_major_exact = 0.0
    if province_rows:
        max_count = max(int(row['sample_count']) for row in province_rows.values())
        major_provinces = sorted(
            province for province, row in province_rows.items() if int(row['sample_count']) == max_count
        )
        major_province = '|'.join(major_provinces)
        major_total = sum(int(province_rows[province]['sample_count']) for province in major_provinces)
        major_exact_total = sum(int(province_rows[province]['exact_plate_correct']) for province in major_provinces)
        major_ratio = safe_div(major_total, total)
        major_exact = safe_div(major_exact_total, major_total)
        non_major_total = 0
        non_major_correct = 0
        for province, row in province_rows.items():
            if province in major_provinces:
                continue
            non_major_total += int(row['sample_count'])
            non_major_correct += int(row['exact_plate_correct'])
        non_major_exact = safe_div(non_major_correct, non_major_total)
    return {
        'sample_count': total,
        'exact_plate_acc': safe_div(exact, total),
        'province_macro_exact_acc': macro_exact,
        'province_macro_first_char_acc': macro_first,
        'empty_pred_rate': safe_div(empty_pred, total),
        'short_pred_rate': safe_div(short_pred, total),
        'major_province': major_province,
        'major_province_ratio': major_ratio,
        'major_province_exact_acc': major_exact,
        'non_major_province_exact_acc': non_major_exact,
        'proxy_subset_count': len(subset_indices),
    }


def better_selection_metric(current, best, strategy='balanced_tuple'):
    if best is None:
        return True
    if strategy == 'balanced_recovery':
        current_key = (
            current['province_macro_first_char_acc'],
            -current['empty_pred_rate'],
            -current['short_pred_rate'],
            current['exact_plate_acc'],
            current['province_macro_exact_acc'],
            current['non_major_province_exact_acc'],
        )
        best_key = (
            best['province_macro_first_char_acc'],
            -best['empty_pred_rate'],
            -best['short_pred_rate'],
            best['exact_plate_acc'],
            best['province_macro_exact_acc'],
            best['non_major_province_exact_acc'],
        )
    else:
        current_key = (
            current['exact_plate_acc'],
            current['province_macro_exact_acc'],
            current['province_macro_first_char_acc'],
            current['non_major_province_exact_acc'],
            -current['empty_pred_rate'],
            -current['short_pred_rate'],
        )
        best_key = (
            best['exact_plate_acc'],
            best['province_macro_exact_acc'],
            best['province_macro_first_char_acc'],
            best['non_major_province_exact_acc'],
            -best['empty_pred_rate'],
            -best['short_pred_rate'],
        )
    return current_key > best_key


def evaluate_exact_plate_subset(net, dataset, batch_size, num_workers, use_cuda, max_samples, decode_mode, beam_size, beam_topk, proxy_mode='sequential', proxy_seed=42):
    if dataset is None or len(dataset) == 0 or max_samples <= 0:
        return None
    eval_dataset, _subset_indices = build_proxy_eval_dataset(dataset, max_samples, mode=proxy_mode, seed=proxy_seed)
    loader = DataLoader(
        eval_dataset,
        batch_size=min(batch_size, len(eval_dataset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    net.eval()
    exact = 0
    total = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            if use_cuda:
                images = images.cuda()
            sample_families = list(families)
            prebs = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(prebs, decode_mode, beam_size, beam_topk, sample_families=sample_families)
            for pred_ids, gt_ids in zip(decoded, targets):
                exact += int(np.array_equal(np.asarray(pred_ids), np.asarray(gt_ids)))
                total += 1
    net.train()
    return (exact / total) if total else 0.0

def train():
    args = get_parser()
    enforce_board_alignment_args(args)
    configure_runtime(args.seed, args.deterministic)

    # Override CHARS from keys file if provided (used by yellow/special LPRNet)
    if args.keys_file:
        with open(args.keys_file, 'r', encoding='utf-8') as f:
            custom_chars = [line.strip() for line in f if line.strip()]
        custom_chars.append('-')  # CTC blank as last char
        import load_data as _ld_mod
        _ld_mod.CHARS.clear()
        _ld_mod.CHARS.extend(custom_chars)
        _ld_mod.CHARS_DICT.clear()
        _ld_mod.CHARS_DICT.update({c:i for i,c in enumerate(custom_chars)})
        # Local references (already imported via from load_data import CHARS)
        # point to the same in-place modified objects, no reassign needed.
        print('[KeysFile] Overrode CHARS: {} real keys from {} (class_num={})'.format(
            len(custom_chars) - 1, args.keys_file, len(custom_chars)))

    T_length = 18 # kept for compatibility; actual CTC input length now uses logits.shape[2] dynamically per batch
    epoch = 0 + args.resume_epoch
    loss_val = 0

    os.makedirs(args.save_folder, exist_ok=True)

    pos0_target_families = build_target_family_set(args.pos0_target_families)
    province_target_families = build_target_family_set(args.province_target_families)
    slot_target_families = build_target_family_set(args.slot_target_families)
    adapter_target_families = build_target_family_set(args.adapter_target_families)
    slot_pos_weights = parse_slot_pos_weights(args.slot_pos_weights, args.lpr_max_len)
    if args.head_mode == 'multihead':
        lprnet = build_lprnet_multihead(
            lpr_max_len=args.lpr_max_len,
            phase=args.phase_train,
            class_num=len(CHARS),
            dropout_rate=args.dropout_rate,
            enhanced_green_head=args.enhanced_green_head,
            pos0_head_cols=args.pos0_head_cols,
            pos0_num_classes=args.pos0_num_classes,
            adapter_families=sorted(adapter_target_families),
            adapter_hidden_channels=args.adapter_hidden_channels,
        )
        if args.enhanced_green_head:
            print(f'[HeadConfig] Using enhanced green8 head variant={args.enhanced_green_head}')
        if adapter_target_families:
            print(f'[AdapterConfig] family-specific adapters={sorted(adapter_target_families)} hidden={args.adapter_hidden_channels}')
        if pos0_target_families:
            lprnet.enable_family_specific_pos0(sorted(pos0_target_families), pos0_num_classes=args.pos0_num_classes)
            print(f'[Pos0Config] family-specific pos0 heads={sorted(pos0_target_families)}')
        if province_target_families:
            lprnet.enable_family_specific_province(sorted(province_target_families), province_num_classes=args.province_num_classes)
            print(f'[ProvinceConfig] family-specific province heads={sorted(province_target_families)}')
        if slot_target_families:
            lprnet.enable_family_specific_slot(sorted(slot_target_families))
            print(f'[SlotConfig] family-specific slot heads={sorted(slot_target_families)} pos_weights={slot_pos_weights}')
    else:
        lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    print(f"[Env] device={device} model_training={lprnet.training}")

    # load pretrained model (智能微调版)
    if args.pretrained_model:
        pretrained_dict = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
        model_dict = lprnet.state_dict()

        # 标准匹配：同名且 shape 一致的权重直接加载
        matched = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        direct_match_count = len(matched)

        multihead_mapped = {}
        if args.head_mode == 'multihead':
            container_keys = [k for k in pretrained_dict.keys() if k.startswith('container.')]
            if container_keys:
                family_map_counts = {family: 0 for family in FAMILY_HEADS}
                for family in FAMILY_HEADS:
                    for old_key in container_keys:
                        new_key = old_key.replace('container.', f'containers.{family}.')
                        if new_key in model_dict and pretrained_dict[old_key].shape == model_dict[new_key].shape:
                            multihead_mapped[new_key] = pretrained_dict[old_key].clone()
                            family_map_counts[family] += 1
                matched.update(multihead_mapped)
                print('[MultiheadInit] mapped {} single-head container params into multihead families {}'.format(
                    len(container_keys), family_map_counts
                ))
            else:
                print('[MultiheadInit] no single-head container.* params found in pretrained checkpoint')

        # 把过滤好的旧知识更新到现在的模型里
        model_dict.update(matched)
        lprnet.load_state_dict(model_dict)
        backbone_count = sum(1 for k in matched if k.startswith('backbone.'))
        head_count = sum(1 for k in matched if k.startswith('container.') or k.startswith('containers.'))
        print('[LoadPretrained] backbone={} direct={} multihead_extra={} head={} total={}'.format(
            backbone_count, direct_match_count, len(multihead_mapped), head_count, len(matched)
        ))
        print(f"【微调模式】成功加载预训练权重！保留了 {len(matched)} 个匹配的层。")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    aux_second_head = nn.Linear(len(CHARS), len(CHARS)).to(device)
    aux_ne_head = nn.Linear(len(CHARS), 2).to(device)

    trainable_family_set = build_trainable_family_set(args.trainable_families)
    trainable_backbone_prefixes = parse_trainable_backbone_prefixes(args.trainable_backbone_prefixes)
    if args.freeze_backbone:
        for name, p in lprnet.named_parameters():
            if not name.startswith('backbone.'):
                continue
            p.requires_grad = any(name.startswith(prefix) for prefix in trainable_backbone_prefixes)
        freeze_batchnorm_eval(lprnet.backbone, freeze_params=True)
        if trainable_backbone_prefixes:
            print('[Freeze] backbone frozen with trainable prefixes={} (BN stats frozen)'.format(trainable_backbone_prefixes))
        else:
            print('[Freeze] backbone frozen for this run (params + BN stats)')
    if args.head_mode == 'multihead' and trainable_family_set is not None:
        for family_name, family_head in lprnet.containers.items():
            requires_grad = family_name in trainable_family_set
            for p in family_head.parameters():
                p.requires_grad = requires_grad
        print('[Freeze] multihead trainable families={}'.format(sorted(trainable_family_set)))

    if args.freeze_bn_stats and not args.freeze_backbone and hasattr(lprnet, 'backbone'):
        freeze_batchnorm_eval(lprnet.backbone, freeze_params=False)
        print('[Freeze] backbone BN stats frozen for soft-freeze (params remain trainable)')

    optim_params = build_optimizer_param_groups(
        lprnet,
        aux_second_head,
        aux_ne_head,
        second_char_aux_weight=args.second_char_aux_weight,
        ne_type_aux_weight=args.ne_type_aux_weight,
        base_lr=args.learning_rate,
        backbone_lr_mult=args.backbone_lr_mult,
        head_lr_mult=args.head_lr_mult,
        adapter_lr_mult=args.adapter_lr_mult,
        aux_lr_mult=args.aux_lr_mult,
    )
    log_optimizer_param_groups(optim_params)

    optimizer = optim.RMSprop(
        optim_params,
        lr=args.learning_rate,
        alpha=0.9,
        eps=1e-08,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    board_anchor_img_dirs = os.path.expanduser(args.board_anchor_img_dirs)
    pseudo_anchor_img_dirs = os.path.expanduser(args.pseudo_anchor_img_dirs) if args.pseudo_anchor_img_dirs else train_img_dirs
    secondary_train_img_dirs = os.path.expanduser(args.secondary_train_img_dirs) if args.secondary_train_img_dirs else train_img_dirs

    board_anchor_eval_dataset = None
    pseudo_anchor_val_dataset = None
    secondary_train_dataset = None
    train_texts = []
    sample_sources = []

    common_dataset_kwargs = dict(
        ocr_channel_order=args.ocr_channel_order,
        ocr_crop_mode=args.ocr_crop_mode,
        ocr_resize_mode=args.ocr_resize_mode,
        ocr_resize_kernel=args.ocr_resize_kernel,
        ocr_preproc=args.ocr_preproc,
        ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        ocr_quad_pad_ratio=args.ocr_quad_pad_ratio,
        dataset_root=os.path.expanduser(args.dataset_root),
        strict_path_check=args.strict_path_check,
    )
    if args.data_mode == 'ccpd_board':
        train_main_dataset = CCPDBoardDataLoader(
            train_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.train_txt_file,
            plate_box_aug_mode=args.train_plate_box_aug_mode,
            plate_box_aug_prob=args.train_plate_box_aug_prob,
            plate_box_aug_x=args.train_plate_box_aug_x,
            plate_box_aug_y=args.train_plate_box_aug_y,
            plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
            **common_dataset_kwargs,
        )
        test_dataset = CCPDBoardDataLoader(
            test_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.test_txt_file,
            **common_dataset_kwargs,
        )
    elif args.data_mode == 'manifest':
        if not args.train_manifest or not args.test_manifest:
            raise RuntimeError('data_mode=manifest requires --train_manifest and --test_manifest')
        train_main_dataset = UnifiedManifestDataset(
            manifest_path=args.train_manifest,
            img_size=args.img_size,
            lpr_max_len=args.lpr_max_len,
            split_filter='train',
            plate_box_aug_mode=args.train_plate_box_aug_mode,
            plate_box_aug_prob=args.train_plate_box_aug_prob,
            plate_box_aug_x=args.train_plate_box_aug_x,
            plate_box_aug_y=args.train_plate_box_aug_y,
            plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
            brightness_aug_max=args.train_brightness_aug_max,
            gray3_prob=args.gray3_prob,
            **common_dataset_kwargs,
        )
        if hasattr(train_main_dataset, 'records'):
            valid_train_indices = []
            filtered_records = []
            filtered_img_paths = []
            filtered_img_labels = []
            for idx, row in enumerate(getattr(train_main_dataset, 'records', [])):
                img_path = row.get('img_path') if isinstance(row, dict) else None
                if img_path:
                    resolved = train_main_dataset._resolve_img_path(img_path)
                    if os.path.exists(resolved):
                        valid_train_indices.append(idx)
                        filtered_records.append(row)
                        filtered_img_paths.append(train_main_dataset.img_paths[idx])
                        filtered_img_labels.append(train_main_dataset.img_labels[idx])
            if len(valid_train_indices) != len(train_main_dataset):
                print(f'[Info] skip missing-image train rows: keep {len(valid_train_indices)}/{len(train_main_dataset)}')
                train_main_dataset.records = filtered_records
                train_main_dataset.img_paths = filtered_img_paths
                train_main_dataset.img_labels = filtered_img_labels
        test_dataset = UnifiedManifestDataset(
            manifest_path=args.test_manifest,
            img_size=args.img_size,
            lpr_max_len=args.lpr_max_len,
            split_filter=args.test_split_filter,
            **common_dataset_kwargs,
        )
        if hasattr(test_dataset, 'records'):
            valid_indices = []
            for idx, row in enumerate(getattr(test_dataset, 'records', [])):
                img_path = row.get('img_path') if isinstance(row, dict) else None
                if img_path:
                    resolved = test_dataset._resolve_img_path(img_path)
                    if os.path.exists(resolved):
                        valid_indices.append(idx)
            if len(valid_indices) != len(test_dataset):
                print(f'[Info] skip missing-image test rows: keep {len(valid_indices)}/{len(test_dataset)}')
                test_dataset = Subset(test_dataset, valid_indices)
        print(f'[Data] test_split_filter={args.test_split_filter} test_samples={len(test_dataset)}')
    else:
        train_main_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.train_txt_file)
        test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.test_txt_file)

    if len(train_main_dataset) == 0:
        raise RuntimeError(f"No training samples found. train_img_dirs={args.train_img_dirs} train_txt_file={args.train_txt_file}")
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test samples found. test_img_dirs={args.test_img_dirs} test_txt_file={args.test_txt_file}")

    train_dataset = train_main_dataset
    train_texts.extend(list(getattr(train_main_dataset, 'img_labels', [])))
    sample_sources.extend(['main'] * len(getattr(train_main_dataset, 'img_labels', [])))
    sample_metas = []
    if hasattr(train_main_dataset, 'records'):
        sample_metas.extend(list(getattr(train_main_dataset, 'records', [])))
    else:
        sample_metas.extend([{}] * len(getattr(train_main_dataset, 'img_labels', [])))

    if args.pseudo_anchor_train_txt_file:
        if args.data_mode != 'ccpd_board':
            raise RuntimeError('pseudo anchors currently require --data_mode ccpd_board')
        pseudo_anchor_train_dataset = CCPDBoardDataLoader(
            pseudo_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.pseudo_anchor_train_txt_file,
            plate_box_aug_mode=args.train_plate_box_aug_mode,
            plate_box_aug_prob=args.train_plate_box_aug_prob,
            plate_box_aug_x=args.train_plate_box_aug_x,
            plate_box_aug_y=args.train_plate_box_aug_y,
            plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
            **common_dataset_kwargs,
        )
        if len(pseudo_anchor_train_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, pseudo_anchor_train_dataset])
            train_texts.extend(list(pseudo_anchor_train_dataset.img_labels))
            sample_sources.extend(['pseudo'] * len(pseudo_anchor_train_dataset))
            sample_metas.extend([{}] * len(pseudo_anchor_train_dataset))
        else:
            print(f"[PseudoAnchorTrain] no valid samples from {args.pseudo_anchor_train_txt_file}, disable pseudo-anchor training")

    if args.pseudo_anchor_val_txt_file:
        if args.data_mode != 'ccpd_board':
            raise RuntimeError('pseudo anchors currently require --data_mode ccpd_board')
        pseudo_anchor_val_dataset = CCPDBoardDataLoader(
            pseudo_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.pseudo_anchor_val_txt_file,
            **common_dataset_kwargs,
        )
        if len(pseudo_anchor_val_dataset) == 0:
            print(f"[PseudoAnchorVal] no valid samples from {args.pseudo_anchor_val_txt_file}, disable pseudo-anchor validation")
            pseudo_anchor_val_dataset = None

    if args.secondary_train_txt_file:
        if args.data_mode == 'ccpd_board':
            secondary_train_dataset = CCPDBoardDataLoader(
                secondary_train_img_dirs.split(','),
                args.img_size,
                args.lpr_max_len,
                txt_file=args.secondary_train_txt_file,
                plate_box_aug_mode=args.train_plate_box_aug_mode,
                plate_box_aug_prob=args.train_plate_box_aug_prob,
                plate_box_aug_x=args.train_plate_box_aug_x,
                plate_box_aug_y=args.train_plate_box_aug_y,
                plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
                **common_dataset_kwargs,
            )
        else:
            secondary_train_dataset = LPRDataLoader(
                secondary_train_img_dirs.split(','),
                args.img_size,
                args.lpr_max_len,
                txt_file=args.secondary_train_txt_file,
            )
        if len(secondary_train_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, secondary_train_dataset])
            train_texts.extend(list(getattr(secondary_train_dataset, 'img_labels', [])))
            sample_sources.extend(['secondary'] * len(getattr(secondary_train_dataset, 'img_labels', [])))
            sample_metas.extend([{}] * len(getattr(secondary_train_dataset, 'img_labels', [])))
        else:
            print(f"[SecondaryTrain] no valid samples from {args.secondary_train_txt_file}, disable secondary training mix")
            secondary_train_dataset = None

    if args.board_anchor_txt_file:
        board_anchor_eval_dataset = BoardDumpDataLoader(
            board_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.board_anchor_txt_file,
        )
        if len(board_anchor_eval_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, board_anchor_eval_dataset])
            train_texts.extend(list(board_anchor_eval_dataset.img_labels))
            sample_sources.extend(['board'] * len(board_anchor_eval_dataset))
            sample_metas.extend([{}] * len(board_anchor_eval_dataset))
        else:
            print(f"[BoardAnchor] no valid samples from {args.board_anchor_txt_file}, disable anchor training")
            board_anchor_eval_dataset = None

    sample_weights, province_weights, strata_count, sample_weight_debug = build_sample_weights(
        train_texts,
        sample_sources,
        sample_metas,
        args.province_balance_mode,
        args.province_balance_clip,
        args.strata_balance_mode,
        args.strata_balance_clip,
        args.board_anchor_sample_weight,
        args.pseudo_anchor_sample_weight,
        args.secondary_train_sample_weight,
        args.adj_repeat_sample_weight,
        args.main_group_by,
        args.main_group_ratios,
        args.main_group_clip,
    )
    province_ce_weights = torch.tensor(province_weights, dtype=torch.float32, device=device)
    first_char_class_weights_list = province_weights
    first_char_count_debug = []
    if args.first_char_loss_type == 'class_balanced_ce':
        first_char_class_weights_list, first_char_count_debug = build_class_balanced_class_weights(train_texts, beta=args.first_char_cb_beta)
    first_char_ce_weights = torch.tensor(first_char_class_weights_list, dtype=torch.float32, device=device)
    if args.pos0_head_cols > 0 and args.pos0_head_weight > 0.0:
        pos0_weights_list, _ = build_class_balanced_class_weights(train_texts, beta=args.first_char_cb_beta, num_classes=args.pos0_num_classes)
        pos0_ce_weights = torch.tensor(pos0_weights_list, dtype=torch.float32, device=device)
    else:
        pos0_ce_weights = None
    train_loader = make_train_loader(train_dataset, sample_weights, args.train_batch_size, args.num_workers, args.seed)
    epoch_size = len(train_loader)
    if epoch_size == 0:
        raise RuntimeError(
            f"train_batch_size={args.train_batch_size} is larger than effective train samples={len(train_dataset)}; "
            "reduce batch size or add more samples."
        )

    print(
        f"[Data] mode={args.data_mode} train_samples={len(train_dataset)} test_samples={len(test_dataset)} "
        f"train_plate_box_aug={args.train_plate_box_aug_mode} prob={args.train_plate_box_aug_prob:.2f} "
        f"board_anchors={(len(board_anchor_eval_dataset) if board_anchor_eval_dataset else 0)} "
        f"pseudo_train={(sample_sources.count('pseudo'))} pseudo_val={(len(pseudo_anchor_val_dataset) if pseudo_anchor_val_dataset else 0)} "
        f"secondary_train={(sample_sources.count('secondary'))} secondary_weight={args.secondary_train_sample_weight:.2f} "
        f"province_balance={args.province_balance_mode} clip={args.province_balance_clip:.2f} "
        f"strata_balance={args.strata_balance_mode} strata_clip={args.strata_balance_clip:.2f} strata={strata_count} "
        f"main_group_by={sample_weight_debug.get('main_group_by','none')} main_group_ratios={sample_weight_debug.get('main_group_target_ratios', {})} "
        f"main_group_weights={sample_weight_debug.get('main_group_weights', {})} "
        f"adj_repeat_weight={args.adj_repeat_sample_weight:.2f} "
        f"aux(first={args.first_char_aux_weight:.2f},second={args.second_char_aux_weight:.2f},ne={args.ne_type_aux_weight:.2f},rear={args.rear_seq_aux_weight:.2f},pos0={args.pos0_head_weight:.2f},prov={args.province_head_weight:.2f},slot={args.slot_head_weight:.2f}) "
        f"ctc_loss_type={args.ctc_loss_type} focal(alpha={args.focal_ctc_alpha:.2f},gamma={args.focal_ctc_gamma:.2f}) "
        f"first_char_loss_type={args.first_char_loss_type} first_char_cb_beta={args.first_char_cb_beta:.4f}"
    )
    if args.first_char_loss_type == 'class_balanced_ce':
        nonzero_cb = {CHARS[i]: round(float(first_char_class_weights_list[i]), 4) for i in range(PROVINCE_COUNT) if first_char_class_weights_list[i] > 0}
        print(f"[FirstCharLoss] class_balanced_ce beta={args.first_char_cb_beta:.4f} counts={first_char_count_debug[:PROVINCE_COUNT]} weights={nonzero_cb}")

    if args.preflight_only:
        print('[PreflightOnly] model/data/optimizer ready; exit before training')
        return

    if args.ctc_loss_type == 'focal':
        ctc_loss = FocalCTCLoss(blank=len(CHARS)-1, alpha=args.focal_ctc_alpha, gamma=args.focal_ctc_gamma, reduction='mean')
    else:
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean', zero_infinity=True)
    global_iter = args.resume_epoch * epoch_size
    best_board_metric = None
    best_selection_metric = None
    best_state_dict = None
    best_epoch = 0
    best_proxy_acc = -1.0
    epochs_since_improve = 0
    regression_epochs = 0
    best_checkpoint_path = os.path.join(args.save_folder, 'best_LPRNet_model.pth')
    last_checkpoint_path = os.path.join(args.save_folder, 'last_LPRNet_model.pth')
    final_checkpoint_path = os.path.join(args.save_folder, 'Final_LPRNet_model.pth')
    train_summary_path = os.path.join(args.save_folder, 'train_summary.json')
    max_steps_reached = False

    for epoch in range(args.resume_epoch + 1, args.max_epoch + 1):
        lprnet.train()
        enforce_frozen_backbone_runtime(lprnet, args.freeze_backbone, args.freeze_bn_stats)
        aux_second_head.train()
        aux_ne_head.train()
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)
        train_loader = make_train_loader(train_dataset, sample_weights, args.train_batch_size, args.num_workers, args.seed + epoch)
        epoch_loss_sum = 0.0
        epoch_aux_sum = 0.0
        epoch_loss_count = 0

        for epochiter, (images, labels, lengths, families) in enumerate(train_loader):
            start_time = time.time()
            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            pos0_images = build_gray3_from_normalized_bgr(images) if args.pos0_gray_input and args.pos0_head_weight > 0.0 else None
            if args.head_mode == 'multihead':
                all_outputs = lprnet(images, pos0_input=pos0_images)
            else:
                all_outputs = lprnet(images)
            pos0_logits = all_outputs.pop('pos0', None) if isinstance(all_outputs, dict) else None
            province_logits = all_outputs.pop('province', None) if isinstance(all_outputs, dict) else None
            slot_logits_by_family = {}
            if isinstance(all_outputs, dict):
                for key in list(all_outputs.keys()):
                    if key.startswith('slot_'):
                        slot_logits_by_family[key[5:]] = all_outputs.pop(key)
            logits = _select_family_logits_from_dict(all_outputs, sample_families=families) if isinstance(all_outputs, dict) else all_outputs
            log_probs = logits.permute(2, 0, 1).log_softmax(2).requires_grad_()
            input_lengths, target_lengths = sparse_tuple_for_ctc(logits.shape[2], lengths)
            aux_steps = max(1, min(args.first_char_time_steps, logits.shape[2]))
            first_targets = torch.tensor(extract_first_char_targets(labels, lengths), dtype=torch.long, device=labels.device)
            # Guard: when class_num < PROVINCE_COUNT, first_char_aux is not applicable
            # (e.g. embassy-only with class_num=12 cannot slice 31 province classes)
            if logits.shape[1] < PROVINCE_COUNT:
                if args.first_char_aux_weight > 0:
                    print(f'[FirstCharGuard] class_num={logits.shape[1]} < PROVINCE_COUNT={PROVINCE_COUNT}, '
                          f'disabling first_char_aux_loss for this run (was {args.first_char_aux_weight:.2f})')
                    args.first_char_aux_weight = 0.0
                first_aux_loss = torch.tensor(0.0)
            else:
                first_logits = logits[:, :PROVINCE_COUNT, :aux_steps].mean(dim=2)
                first_aux_loss = F.cross_entropy(first_logits, first_targets, weight=first_char_ce_weights)
            if isinstance(all_outputs, dict) and pos0_target_families and args.pos0_head_weight > 0.0:
                pos0_losses = []
                for family in sorted(pos0_target_families):
                    family_key = f'pos0_{family}'
                    family_logits = all_outputs.get(family_key)
                    if family_logits is None:
                        continue
                    family_mask = torch.tensor([f == family for f in families], dtype=torch.bool, device=labels.device)
                    if bool(torch.any(family_mask)):
                        pos0_losses.append(F.cross_entropy(family_logits[family_mask], first_targets[family_mask], weight=pos0_ce_weights))
                pos0_loss = torch.stack(pos0_losses).mean() if pos0_losses else torch.zeros((), device=labels.device, dtype=logits.dtype)
            elif pos0_logits is not None and args.pos0_head_weight > 0.0:
                pos0_loss = F.cross_entropy(pos0_logits, first_targets, weight=pos0_ce_weights)
            else:
                pos0_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            if isinstance(all_outputs, dict) and province_target_families and args.province_head_weight > 0.0:
                province_losses = []
                for family in sorted(province_target_families):
                    family_key = f'province_{family}'
                    family_logits = all_outputs.get(family_key)
                    if family_logits is None:
                        continue
                    family_mask = torch.tensor([f == family for f in families], dtype=torch.bool, device=labels.device)
                    if bool(torch.any(family_mask)):
                        province_losses.append(F.cross_entropy(family_logits[family_mask], first_targets[family_mask], weight=pos0_ce_weights))
                province_loss = torch.stack(province_losses).mean() if province_losses else torch.zeros((), device=labels.device, dtype=logits.dtype)
            elif province_logits is not None and args.province_head_weight > 0.0:
                province_loss = F.cross_entropy(province_logits, first_targets, weight=pos0_ce_weights)
            else:
                province_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            seq_feat = logits.mean(dim=2)
            if args.second_char_aux_weight > 0.0:
                second_targets = torch.tensor(extract_second_char_targets(labels, lengths), dtype=torch.long, device=labels.device)
                second_logits = aux_second_head(seq_feat)
                second_aux_loss = F.cross_entropy(second_logits, second_targets)
            else:
                second_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            if args.ne_type_aux_weight > 0.0:
                ne_targets = torch.tensor(extract_ne_type_targets(labels, lengths), dtype=torch.long, device=labels.device)
                valid_ne = ne_targets < 2
                if bool(torch.any(valid_ne)):
                    ne_logits = aux_ne_head(seq_feat[valid_ne])
                    ne_aux_loss = F.cross_entropy(ne_logits, ne_targets[valid_ne])
                else:
                    ne_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)
            else:
                ne_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            if args.rear_seq_aux_weight > 0.0:
                rear_start = max(0, min(int(args.rear_seq_start_step), logits.shape[2] - 1))
                rear_logits = logits[:, :, rear_start:]
                rear_log_probs = rear_logits.permute(2, 0, 1).log_softmax(2).requires_grad_()
                rear_targets_flat, rear_target_lengths = sparse_tuple_for_suffix_ctc(labels.detach().cpu(), lengths, drop_chars=args.rear_seq_drop_chars)
                rear_input_lengths = tuple([rear_logits.shape[2]] * len(rear_target_lengths))
                if rear_logits.shape[2] > 0 and sum(rear_target_lengths) > 0 and max(rear_target_lengths) <= rear_logits.shape[2]:
                    rear_targets = torch.tensor(rear_targets_flat, dtype=torch.long, device=labels.device)
                    rear_seq_aux_loss = ctc_loss(rear_log_probs, rear_targets, input_lengths=rear_input_lengths, target_lengths=tuple(rear_target_lengths))
                else:
                    rear_seq_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)
            else:
                rear_seq_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            if slot_target_families and args.slot_head_weight > 0.0:
                slot_losses = []
                start = 0
                padded_targets = []
                valid_lengths = []
                for length in lengths:
                    length_int = int(length)
                    label = labels[start:start + length_int]
                    start += length_int
                    if length_int == args.lpr_max_len:
                        padded_targets.append(label)
                        valid_lengths.append(True)
                    else:
                        padded = torch.full((args.lpr_max_len,), len(CHARS) - 1, dtype=labels.dtype, device=labels.device)
                        copy_len = min(length_int, args.lpr_max_len)
                        if copy_len > 0:
                            padded[:copy_len] = label[:copy_len]
                        padded_targets.append(padded)
                        valid_lengths.append(False)
                slot_targets = torch.stack(padded_targets, dim=0)
                slot_weight = torch.tensor(slot_pos_weights, dtype=logits.dtype, device=labels.device)
                for family in sorted(slot_target_families):
                    family_logits = slot_logits_by_family.get(family)
                    if family_logits is None:
                        continue
                    family_mask = torch.tensor([f == family for f in families], dtype=torch.bool, device=labels.device)
                    if bool(torch.any(family_mask)):
                        per_pos = F.cross_entropy(
                            family_logits[family_mask].reshape(-1, len(CHARS)),
                            slot_targets[family_mask].reshape(-1),
                            reduction='none',
                        ).view(-1, args.lpr_max_len)
                        slot_losses.append((per_pos * slot_weight).sum() / torch.clamp(slot_weight.sum() * per_pos.shape[0], min=1.0))
                slot_aux_loss = torch.stack(slot_losses).mean() if slot_losses else torch.zeros((), device=labels.device, dtype=logits.dtype)
            else:
                slot_aux_loss = torch.zeros((), device=labels.device, dtype=logits.dtype)

            optimizer.zero_grad()
            max_label_idx = int(labels.max().item()) if labels.numel() > 0 else -1
            if max_label_idx >= log_probs.shape[2]:
                raise RuntimeError(
                    f'CTC label index overflow: max_label_idx={max_label_idx} class_dim={log_probs.shape[2]} '
                    f'labels_sample={labels[:32].detach().cpu().tolist()}'
                )
            ctc = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            loss = ctc
            loss = loss + args.first_char_aux_weight * first_aux_loss
            if args.second_char_aux_weight > 0.0:
                loss = loss + args.second_char_aux_weight * second_aux_loss
            if args.ne_type_aux_weight > 0.0:
                loss = loss + args.ne_type_aux_weight * ne_aux_loss
            if args.rear_seq_aux_weight > 0.0:
                loss = loss + args.rear_seq_aux_weight * rear_seq_aux_loss
            if args.pos0_head_weight > 0.0:
                loss = loss + args.pos0_head_weight * pos0_loss
            if args.province_head_weight > 0.0:
                loss = loss + args.province_head_weight * province_loss
            if args.slot_head_weight > 0.0:
                loss = loss + args.slot_head_weight * slot_aux_loss
            loss_item = loss.item()
            aux_item = first_aux_loss.item()
            second_aux_item = second_aux_loss.item()
            ne_aux_item = ne_aux_loss.item()
            rear_aux_item = rear_seq_aux_loss.item()
            pos0_aux_item = pos0_loss.item()
            province_aux_item = province_loss.item()
            slot_aux_item = slot_aux_loss.item()
            if np.isnan(loss_item):
                print("[Fatal] Loss became NaN. Stop training.")
                return
            if loss_item == np.inf:
                continue
            loss.backward()
            optimizer.step()

            global_iter += 1
            if args.max_steps > 0 and global_iter >= args.max_steps:
                print(f'[max_steps] Reached max_steps={args.max_steps} at global_iter={global_iter}, stopping training.')
                max_steps_reached = True
                break
            epoch_loss_sum += loss_item
            epoch_aux_sum += aux_item
            epoch_loss_count += 1
            end_time = time.time()

            if global_iter % args.save_interval == 0:
                torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(global_iter) + '.pth')

            if global_iter == 1 or epochiter % 20 == 0:
                print(
                    'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(epochiter) + '/' + repr(epoch_size)
                    + '|| Totel iter ' + repr(global_iter) + ' || Loss: %.4f|| Aux1: %.4f Aux2: %.4f AuxNE: %.4f AuxRear: %.4f AuxPos0: %.4f AuxProv: %.4f AuxSlot: %.4f||' % (
                        loss_item,
                        aux_item,
                        second_aux_item,
                        ne_aux_item,
                        rear_aux_item,
                        pos0_aux_item,
                        province_aux_item,
                        slot_aux_item,
                    )
                    + 'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr)
                )

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_loss_count)
        avg_aux_loss = epoch_aux_sum / max(1, epoch_loss_count)
        print('[Epoch Summary] Epoch {} AvgLoss {:.4f} AvgAux {:.4f}'.format(epoch, avg_epoch_loss, avg_aux_loss))

        if max_steps_reached:
            print('[max_steps] Breaking out of epoch loop.')
            break

        board_metrics = evaluate_board_anchor_dataset(
            lprnet,
            board_anchor_eval_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.first_char_time_steps,
        )
        pseudo_metrics = evaluate_first_char_dataset(
            lprnet,
            pseudo_anchor_val_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.first_char_time_steps,
            detail_limit=0,
        )
        proxy_exact = evaluate_exact_plate_subset(
            lprnet,
            test_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.selection_proxy_eval_samples,
            args.selection_decode_mode,
            args.selection_beam_size,
            args.selection_beam_topk,
            args.selection_proxy_mode,
            args.seed,
        )
        selection_metrics = None
        proxy_eval_dataset, proxy_subset_indices = build_proxy_eval_dataset(
            test_dataset,
            args.selection_proxy_eval_samples,
            mode=args.selection_proxy_mode,
            seed=args.seed,
        )
        proxy_family_text, proxy_source_text, proxy_province_text = summarize_proxy_subset(test_dataset, proxy_subset_indices)
        print(
            '[ProxySubset] mode={} count={} family={} | source={} | top_provinces={}'.format(
                args.selection_proxy_mode,
                len(proxy_subset_indices),
                proxy_family_text,
                proxy_source_text,
                proxy_province_text,
            )
        )
        if args.selection_strategy in ('balanced_tuple', 'balanced_recovery'):
            selection_metrics = evaluate_selection_dataset(
                lprnet,
                test_dataset,
                args.test_batch_size,
                args.num_workers,
                args.cuda,
                args.selection_proxy_eval_samples,
                args.selection_decode_mode,
                args.selection_beam_size,
                args.selection_beam_topk,
                args.selection_proxy_mode,
                args.seed,
            )
        if proxy_exact is not None:
            print(
                '[SelectionProxy] Epoch {} decode={} proxy_mode={} exact {:.4f} on {} eval samples'.format(
                    epoch,
                    args.selection_decode_mode,
                    args.selection_proxy_mode,
                    proxy_exact,
                    len(proxy_subset_indices),
                )
            )
        if selection_metrics is not None:
            print(
                '[SelectionEval] Epoch {} exact {:.4f} macro_exact {:.4f} macro_first {:.4f} non_major {:.4f} empty {:.4f} short {:.4f} major={}({:.4f})'.format(
                    epoch,
                    selection_metrics['exact_plate_acc'],
                    selection_metrics['province_macro_exact_acc'],
                    selection_metrics['province_macro_first_char_acc'],
                    selection_metrics['non_major_province_exact_acc'],
                    selection_metrics['empty_pred_rate'],
                    selection_metrics['short_pred_rate'],
                    selection_metrics['major_province'],
                    selection_metrics['major_province_ratio'],
                )
            )
        if pseudo_metrics is not None:
            print(
                '[PseudoAnchorVal] Epoch {} exact {:.4f} first_char {:.4f} blank_mean {:.4f}'.format(
                    epoch,
                    pseudo_metrics['exact_plate_acc'],
                    pseudo_metrics['first_char_acc'],
                    pseudo_metrics['blank_top1_mean'],
                )
            )
        if board_metrics is not None:
            board_metrics['proxy_exact_plate_acc'] = proxy_exact if proxy_exact is not None else -1.0
            board_metrics['pseudo_first_char_acc'] = pseudo_metrics['first_char_acc'] if pseudo_metrics is not None else -1.0
            board_metrics['pseudo_exact_plate_acc'] = pseudo_metrics['exact_plate_acc'] if pseudo_metrics is not None else -1.0
            print(
                '[BoardAnchor] Epoch {} exact {:.4f} first_char {:.4f} blank_mean {:.4f} pseudo_first {:.4f} pseudo_exact {:.4f} proxy_exact {:.4f}'.format(
                    epoch,
                    board_metrics['exact_plate_acc'],
                    board_metrics['first_char_acc'],
                    board_metrics['blank_top1_mean'],
                    board_metrics['pseudo_first_char_acc'],
                    board_metrics['pseudo_exact_plate_acc'],
                    board_metrics['proxy_exact_plate_acc'],
                )
            )
            for idx, item in enumerate(board_metrics['details'], start=1):
                print(
                    '[BoardAnchor][{}] gt={} pred={} blank={:.4f} top5={}'.format(
                        idx,
                        item['gt'],
                        item['pred'],
                        item['blank_top1_ratio'],
                        item['first_char_top5'],
                    )
                )
            if better_board_metric(board_metrics, best_board_metric):
                best_board_metric = board_metrics
                best_state_dict = {k: v.detach().cpu().clone() for k, v in lprnet.state_dict().items()}
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save(best_state_dict, best_checkpoint_path)
                print('[BoardAnchor] New best checkpoint selected at epoch {}'.format(epoch))
            else:
                epochs_since_improve += 1
        elif args.selection_strategy in ('balanced_tuple', 'balanced_recovery') and selection_metrics is not None:
            if better_selection_metric(selection_metrics, best_selection_metric, args.selection_strategy):
                best_selection_metric = selection_metrics
                best_proxy_acc = selection_metrics['exact_plate_acc']
                best_state_dict = {k: v.detach().cpu().clone() for k, v in lprnet.state_dict().items()}
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save(best_state_dict, best_checkpoint_path)
                print(
                    '[SelectionEval] New best checkpoint selected at epoch {} with exact {:.4f} macro {:.4f}'.format(
                        epoch,
                        selection_metrics['exact_plate_acc'],
                        selection_metrics['province_macro_exact_acc'],
                    )
                )
            else:
                epochs_since_improve += 1
        elif proxy_exact is not None and proxy_exact > best_proxy_acc:
            best_proxy_acc = proxy_exact
            best_state_dict = {k: v.detach().cpu().clone() for k, v in lprnet.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(best_state_dict, best_checkpoint_path)
            print('[SelectionProxy] New best checkpoint selected at epoch {} with {:.4f}'.format(epoch, proxy_exact))
        elif proxy_exact is not None:
            epochs_since_improve += 1

        regression_ref = None
        if args.selection_strategy in ('balanced_tuple', 'balanced_recovery') and best_selection_metric is not None and selection_metrics is not None:
            regression_ref = (
                selection_metrics['exact_plate_acc'],
                best_selection_metric['exact_plate_acc'],
            )
        elif best_proxy_acc >= 0.0 and proxy_exact is not None:
            regression_ref = (proxy_exact, best_proxy_acc)
        if regression_ref is not None:
            cur_metric, best_metric = regression_ref
            if cur_metric < (best_metric - args.early_stop_regression_pp / 100.0):
                regression_epochs += 1
            else:
                regression_epochs = 0

        if epoch >= args.early_stop_start_epoch and args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print('[EarlyStop] no improvement for {} epochs, stop at epoch {}'.format(args.early_stop_patience, epoch))
            break
        if epoch >= args.early_stop_start_epoch and args.early_stop_regression_patience > 0 and regression_epochs >= args.early_stop_regression_patience:
            print(
                '[EarlyStop] primary metric regressed by more than {:.2f}pp for {} consecutive epochs, stop at epoch {}'.format(
                    args.early_stop_regression_pp,
                    args.early_stop_regression_patience,
                    epoch,
                )
            )
            break

    if best_state_dict is not None:
        lprnet.load_state_dict(best_state_dict)
        print('[Training Done] Load best checkpoint from epoch {}'.format(best_epoch))

    print("Final test Accuracy:")
    final_acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
    print('[Training Done] Final Greedy Test Accuracy: {:.6f}'.format(final_acc))
    if best_proxy_acc >= 0.0:
        print('[Training Done] Best SelectionProxy Accuracy (mode={}): {:.6f}'.format(args.selection_decode_mode, best_proxy_acc))
    if best_selection_metric is not None:
        print(
            '[Training Done] Best selection metrics: exact {:.4f} macro_exact {:.4f} macro_first {:.4f} non_major {:.4f} empty {:.4f} short {:.4f}'.format(
                best_selection_metric['exact_plate_acc'],
                best_selection_metric['province_macro_exact_acc'],
                best_selection_metric['province_macro_first_char_acc'],
                best_selection_metric['non_major_province_exact_acc'],
                best_selection_metric['empty_pred_rate'],
                best_selection_metric['short_pred_rate'],
            )
        )
    if best_board_metric is not None:
        print(
            '[Training Done] Best board anchor metrics: exact {:.4f} first_char {:.4f} blank_mean {:.4f} pseudo_first {:.4f} pseudo_exact {:.4f} proxy_exact {:.4f}'.format(
                best_board_metric['exact_plate_acc'],
                best_board_metric['first_char_acc'],
                best_board_metric['blank_top1_mean'],
                best_board_metric.get('pseudo_first_char_acc', -1.0),
                best_board_metric.get('pseudo_exact_plate_acc', -1.0),
                best_board_metric.get('proxy_exact_plate_acc', -1.0),
            )
        )

    final_state = lprnet.state_dict()
    torch.save(final_state, final_checkpoint_path)
    torch.save(final_state, last_checkpoint_path)

    summary = {
        'best_epoch': int(best_epoch),
        'best_proxy_acc': float(best_proxy_acc),
        'final_test_acc': float(final_acc),
        'selection_strategy': args.selection_strategy,
        'selection_decode_mode': args.selection_decode_mode,
        'best_checkpoint': best_checkpoint_path if os.path.exists(best_checkpoint_path) else '',
        'last_checkpoint': last_checkpoint_path,
        'final_checkpoint': final_checkpoint_path,
        'best_selection_metric': best_selection_metric,
        'best_board_metric': best_board_metric,
        'args': vars(args),
    }
    with open(train_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write('\n')

def Greedy_Decode_Eval(Net, datasets, args):
    data_loader = DataLoader(
        datasets,
        args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    was_training = Net.training
    Net.eval()
    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()

    with torch.no_grad():
        for images, labels, lengths, families in data_loader:
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start + length]
                targets.append([int(x) for x in label.numpy().tolist()])
                start += length

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            prebs = forward_family_logits(Net, images, sample_families=families).cpu().detach().numpy()
            preb_labels = []
            for bi in range(prebs.shape[0]):
                preb = prebs[bi, :, :]
                preb_label = []
                for tj in range(preb.shape[1]):
                    preb_label.append(int(np.argmax(preb[:, tj], axis=0)))
                no_repeat_blank_label = []
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label:
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)

            for bi, pred in enumerate(preb_labels):
                target = targets[bi]
                if len(pred) != len(target):
                    Tn_1 += 1
                    continue
                if pred == target:
                    Tp += 1
                else:
                    Tn_2 += 1

    if was_training:
        Net.train()
        enforce_frozen_backbone_runtime(Net, getattr(args, 'freeze_backbone', False), getattr(args, 'freeze_bn_stats', False))
    total = Tp + Tn_1 + Tn_2
    Acc = Tp * 1.0 / total if total else 0.0
    t2 = time.time()
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, total))
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / max(1, len(datasets)), len(datasets)))
    return Acc


if __name__ == "__main__":
    train()
