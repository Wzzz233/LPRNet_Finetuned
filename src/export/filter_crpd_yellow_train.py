#!/usr/bin/env python3
"""Step 1: Build a filtered CRPD manifest with only yellow_single train entries."""
import csv, os

CRPD_MANIFEST = "/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv"
OUT_CSV = "/home/wzzz/LPRNet/manifests/crpd_yellow_train_only.csv"
FIELDS = ['img_path','img_rel_path','dataset_name','split','text','plate_len','family',
          'sub_type','source','is_real','need_tilt_aug','preprocess_group','has_bbox',
          'has_quad','can_parse_ccpd_geom','can_perspective','bbox_source','quad_source',
          'ocr_channel_order','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel',
          'ocr_preproc','ocr_min_occ_ratio','ocr_quad_pad_ratio']

rows = []
with open(CRPD_MANIFEST) as f:
    for row in csv.DictReader(f):
        if row.get('sub_type') == 'yellow_single' and row.get('split') == 'train':
            if os.path.exists(row['img_path']):
                rows.append({k: row.get(k, '') for k in FIELDS})

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} yellow_single train entries to {OUT_CSV}")
