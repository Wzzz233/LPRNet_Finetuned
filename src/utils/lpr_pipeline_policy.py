#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BOARD_OCR_CHANNEL_ORDER = 'bgr'
BOARD_OCR_CROP_MODE = 'obb_warp'
BOARD_OCR_RESIZE_MODE = 'letterbox'
BOARD_OCR_RESIZE_KERNEL = 'nn'
BOARD_OCR_PREPROC = 'none'
BOARD_OCR_MIN_OCC_RATIO = 0.90
BOARD_OCR_QUAD_PAD_RATIO = 0.0

BOARD_PARAM_KEYS = (
    'ocr_channel_order',
    'ocr_crop_mode',
    'ocr_resize_mode',
    'ocr_resize_kernel',
    'ocr_preproc',
    'ocr_min_occ_ratio',
    'ocr_quad_pad_ratio',
)

BOARD_PARAM_EXPECTED = {
    'ocr_channel_order': BOARD_OCR_CHANNEL_ORDER,
    'ocr_crop_mode': BOARD_OCR_CROP_MODE,
    'ocr_resize_mode': BOARD_OCR_RESIZE_MODE,
    'ocr_resize_kernel': BOARD_OCR_RESIZE_KERNEL,
    'ocr_preproc': BOARD_OCR_PREPROC,
    'ocr_min_occ_ratio': BOARD_OCR_MIN_OCC_RATIO,
    'ocr_quad_pad_ratio': BOARD_OCR_QUAD_PAD_RATIO,
}

PSEUDO_GEOM_ALLOWED_BBOX_SOURCE = {'detector_obb', 'mapping_csv'}
PSEUDO_GEOM_ALLOWED_QUAD_SOURCE = {'detector_obb', 'mapping_csv'}
STYLE_TRANSFER_HINTS = ('style_transfer', 'stylized', 'translated', 'cycle', 'realmix', 'fastcut')


@dataclass
class PolicyIssue:
    level: str
    message: str
    row_index: int | None = None
    img_path: str | None = None


def board_param_dict() -> dict:
    return dict(BOARD_PARAM_EXPECTED)


def apply_board_params(row: dict) -> dict:
    for key, value in BOARD_PARAM_EXPECTED.items():
        row[key] = value
    return row


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def row_uses_board_pipeline(row: dict) -> bool:
    preprocess_group = (row.get('preprocess_group') or '').strip()
    return preprocess_group == 'ccpd_board'


def row_is_plain_pipeline(row: dict) -> bool:
    preprocess_group = (row.get('preprocess_group') or '').strip()
    crop_mode = (row.get('ocr_crop_mode') or '').strip()
    return preprocess_group == 'plain_plate' or crop_mode == 'plain_plate'


def row_is_pseudo_geom(row: dict) -> bool:
    return (row.get('source') or '').strip() == 'pseudo_geom'


def row_style_transfer_hint(row: dict) -> bool:
    hay = ' '.join([
        str(row.get('dataset_name') or ''),
        str(row.get('img_path') or ''),
        str(row.get('img_rel_path') or ''),
    ]).lower()
    return any(tok in hay for tok in STYLE_TRANSFER_HINTS)


def validate_manifest_row(row: dict, row_index: int | None = None) -> list[PolicyIssue]:
    issues: list[PolicyIssue] = []
    img_path = row.get('img_path')
    preprocess_group = (row.get('preprocess_group') or '').strip()

    if row_uses_board_pipeline(row):
        for key, expected in BOARD_PARAM_EXPECTED.items():
            value = row.get(key)
            if key in ('ocr_min_occ_ratio', 'ocr_quad_pad_ratio'):
                if abs(_to_float(value) - float(expected)) > 1e-6:
                    issues.append(PolicyIssue('error', f'{key}={value} expected {expected}', row_index, img_path))
            else:
                if str(value).strip() != str(expected):
                    issues.append(PolicyIssue('error', f'{key}={value} expected {expected}', row_index, img_path))
        if str(row.get('has_bbox')).strip() not in {'1', 'True', 'true'}:
            issues.append(PolicyIssue('error', 'board pipeline row missing bbox', row_index, img_path))
        if str(row.get('has_quad')).strip() not in {'1', 'True', 'true'}:
            issues.append(PolicyIssue('error', 'board pipeline row missing quad', row_index, img_path))
        if str(row.get('can_perspective')).strip() not in {'1', 'True', 'true'}:
            issues.append(PolicyIssue('error', 'board pipeline row not marked can_perspective', row_index, img_path))
        if str(row.get('can_parse_ccpd_geom')).strip() not in {'1', 'True', 'true'}:
            issues.append(PolicyIssue('error', 'board pipeline row not marked can_parse_ccpd_geom', row_index, img_path))

    if row_is_plain_pipeline(row):
        if str(row.get('has_bbox')).strip() in {'1', 'True', 'true'} or str(row.get('has_quad')).strip() in {'1', 'True', 'true'}:
            issues.append(PolicyIssue('warning', 'plain pipeline row still claims bbox/quad; verify not placeholder geom', row_index, img_path))

    if row_is_pseudo_geom(row):
        bbox_source = (row.get('bbox_source') or '').strip()
        quad_source = (row.get('quad_source') or '').strip()
        if bbox_source not in PSEUDO_GEOM_ALLOWED_BBOX_SOURCE:
            issues.append(PolicyIssue('error', f'pseudo_geom bbox_source={bbox_source} not allowed', row_index, img_path))
        if quad_source not in PSEUDO_GEOM_ALLOWED_QUAD_SOURCE:
            issues.append(PolicyIssue('error', f'pseudo_geom quad_source={quad_source} not allowed', row_index, img_path))

    if row_style_transfer_hint(row):
        bbox_source = (row.get('bbox_source') or '').strip()
        quad_source = (row.get('quad_source') or '').strip()
        if bbox_source != 'detector_obb' or quad_source != 'detector_obb':
            issues.append(PolicyIssue('error', 'style-transfer-like sample must be re-OBB-labeled with detector_obb', row_index, img_path))

    if preprocess_group == 'ccpd_board' and (row.get('bbox_source') in {'none', '', None} or row.get('quad_source') in {'none', '', None}):
        issues.append(PolicyIssue('error', 'ccpd_board row cannot use empty/none bbox_source or quad_source', row_index, img_path))

    return issues


def summarize_issues(issues: Iterable[PolicyIssue]) -> dict:
    out = {'error': 0, 'warning': 0}
    for it in issues:
        out[it.level] = out.get(it.level, 0) + 1
    return out


def ensure_existing_file(path: str) -> bool:
    return Path(path).exists()
