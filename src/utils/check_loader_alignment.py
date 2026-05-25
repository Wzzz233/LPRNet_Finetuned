#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from data.load_data import CCPDBoardDataLoader, UnifiedManifestDataset, CHARS


def parse_args():
    ap = argparse.ArgumentParser(description='Check loader alignment: actual read success, label consistency, and tensor shape.')
    ap.add_argument('--data_mode', required=True, choices=['ccpd_board', 'manifest'])
    ap.add_argument('--test_img_dirs', required=True)
    ap.add_argument('--txt_file', required=True)
    ap.add_argument('--img_w', type=int, default=94)
    ap.add_argument('--img_h', type=int, default=24)
    ap.add_argument('--lpr_max_len', type=int, default=8)
    ap.add_argument('--ocr_channel_order', default='bgr')
    ap.add_argument('--ocr_crop_mode', default='obb_warp')
    ap.add_argument('--ocr_resize_mode', default='letterbox')
    ap.add_argument('--ocr_resize_kernel', default='nn')
    ap.add_argument('--ocr_preproc', default='none')
    ap.add_argument('--ocr_min_occ_ratio', type=float, default=0.90)
    ap.add_argument('--ocr_quad_pad_ratio', type=float, default=0.0)
    ap.add_argument('--sample_indices', default='0,1,2,10,100')
    return ap.parse_args()


def build_dataset(args):
    img_size = [args.img_w, args.img_h]
    if args.data_mode == 'ccpd_board':
        return CCPDBoardDataLoader(
            args.test_img_dirs.split(','),
            img_size,
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
    return UnifiedManifestDataset(
        manifest_path=args.test_img_dirs,
        img_size=img_size,
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


def decode_label(ids):
    return ''.join(CHARS[int(x)] for x in ids)


def main():
    args = parse_args()
    ds = build_dataset(args)
    indices = [int(x) for x in args.sample_indices.split(',') if x.strip()]
    report = {
        'data_mode': args.data_mode,
        'count': len(ds),
        'params': {
            'ocr_crop_mode': args.ocr_crop_mode,
            'ocr_channel_order': args.ocr_channel_order,
            'ocr_resize_mode': args.ocr_resize_mode,
            'ocr_resize_kernel': args.ocr_resize_kernel,
            'ocr_preproc': args.ocr_preproc,
            'ocr_min_occ_ratio': args.ocr_min_occ_ratio,
            'ocr_quad_pad_ratio': args.ocr_quad_pad_ratio,
        },
        'samples': []
    }
    for idx in indices:
        if idx < 0 or idx >= len(ds):
            report['samples'].append({'index': idx, 'error': 'out_of_range'})
            continue
        image, label, length = ds[idx]
        decoded = decode_label(label)
        sample = {
            'index': idx,
            'image_shape': list(image.shape),
            'length': int(length),
            'decoded_label': decoded,
            'path': ds.img_paths[idx] if hasattr(ds, 'img_paths') else None,
            'loader_label_matches_dataset_label': decoded == ds.img_labels[idx] if hasattr(ds, 'img_labels') else None,
        }
        if hasattr(ds, 'records') and idx < len(ds.records):
            row = ds.records[idx]
            sample['manifest_text'] = row.get('text')
            sample['manifest_family'] = row.get('family')
            sample['manifest_preprocess_group'] = row.get('preprocess_group')
            sample['manifest_label_matches_loader'] = decoded == row.get('text')
        report['samples'].append(sample)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
