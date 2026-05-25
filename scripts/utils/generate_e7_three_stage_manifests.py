#!/usr/bin/env python3
import csv
import os

E7_BASE = "/home/wzzz/LPRNet/tmp/green_board_native_e7_v2"
E2_MANIFEST = "/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv"
OUT_DIR = "/home/wzzz/LPRNet/manifests/e7_three_stage_v2"
os.makedirs(OUT_DIR, exist_ok=True)

key_provinces = ['苏', '沪', '浙']

# 读取E2
with open(E2_MANIFEST, 'r') as f:
    e2_rows = list(csv.DictReader(f))
    fieldnames = list(e2_rows[0].keys())

# 读取E7 details
details_path = os.path.join(E7_BASE, 'details', 'details.tsv')
with open(details_path, 'r') as f:
    details = list(csv.DictReader(f, delimiter='	'))

# 分离E7-simple和E7-medium
simple_details = [d for d in details if d.get('bucket') == 'geometry_clean']
medium_details = [d for d in details if d.get('bucket') == 'board_mid_occ']

print(f"E7-simple: {len(simple_details)}")
print(f"E7-medium: {len(medium_details)}")

# 为E7数据添加source标记
def convert_to_manifest_row(detail_row, source_name):
    return {
        'img_path': detail_row['out_img_path'],
        'img_rel_path': detail_row['out_img_rel_path'],
        'dataset_name': f'e7_{source_name}',
        'split': 'train',
        'text': detail_row['text'],
        'plate_len': '8',
        'family': 'green8',
        'sub_type': 'green_small',
        'source': source_name,
        'is_real': '0',
        'need_tilt_aug': '0',
        'preprocess_group': 'board_dump',
        'has_bbox': '0',
        'has_quad': '0',
        'can_parse_ccpd_geom': '0',
        'can_perspective': '0',
        'bbox_source': 'none',
        'quad_source': 'none',
        'ocr_channel_order': 'bgr',
        'ocr_crop_mode': 'board_dump',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'none',
        'ocr_min_occ_ratio': '1.0',
        'ocr_quad_pad_ratio': '0.0'
    }

# 生成E7-simple manifest
simple_rows = [convert_to_manifest_row(d, 'e7_simple') for d in simple_details]
with open(os.path.join(OUT_DIR, 'e7_simple.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(simple_rows)

# 生成E7-medium manifest  
medium_rows = [convert_to_manifest_row(d, 'e7_medium') for d in medium_details]
with open(os.path.join(OUT_DIR, 'e7_medium.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(medium_rows)

# 生成E7-full manifest
full_details = simple_details + medium_details
full_rows = [convert_to_manifest_row(d, 'e7_full') for d in full_details]
with open(os.path.join(OUT_DIR, 'e7_full.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(full_rows)

print(f"\n已生成E7 manifests:")
print(f"  - e7_simple.csv: {len(simple_rows)}")
print(f"  - e7_medium.csv: {len(medium_rows)}")
print(f"  - e7_full.csv: {len(full_rows)}")

# 分离E2数据
e2_keyprov = [r for r in e2_rows if r.get('text', '')[0] in key_provinces and r.get('family') == 'green8']
e2_other = [r for r in e2_rows if not (r.get('text', '')[0] in key_provinces and r.get('family') == 'green8')]

print(f"\nE2分离:")
print(f"  - 关键省份green8: {len(e2_keyprov)}")
print(f"  - 其他: {len(e2_other)}")

# 按省份分组E2
e2_by_prov = {}
for prov in key_provinces:
    e2_by_prov[prov] = [r for r in e2_keyprov if r.get('text', '')[0] == prov]
    print(f"  {prov}: {len(e2_by_prov[prov])}")

# 按省份分组E7
simple_by_prov = {prov: [r for r in simple_rows if r.get('text', '')[0] == prov] for prov in key_provinces}
medium_by_prov = {prov: [r for r in medium_rows if r.get('text', '')[0] == prov] for prov in key_provinces}

# 阶段1: 替据90%
print(f"\n生成阶段1 manifest (替据90%)...")
stage1_rows = list(e2_other)
for prov in key_provinces:
    e2_prov_rows = e2_by_prov[prov]
    e7_prov_rows = simple_by_prov[prov]
    keep = max(1, len(e2_prov_rows) // 10)
    stage1_rows.extend(e2_prov_rows[:keep])
    stage1_rows.extend(e7_prov_rows)
    print(f"  {prov}: 保留{keep}张E2 + {len(e7_prov_rows)}张E7")

with open(os.path.join(OUT_DIR, 'stage1_replace_90.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stage1_rows)
print(f"  总数: {len(stage1_rows)}")

# 阶段2: 替据70%
print(f"\n生成阶段2 manifest (替据70%)...")
stage2_rows = list(e2_other)
for prov in key_provinces:
    e2_prov_rows = e2_by_prov[prov]
    e7_prov_rows = medium_by_prov[prov]
    keep = max(1, int(len(e2_prov_rows) * 0.3))
    stage2_rows.extend(e2_prov_rows[:keep])
    stage2_rows.extend(e7_prov_rows)
    print(f"  {prov}: 保留{keep}张E2 + {len(e7_prov_rows)}张E7")

with open(os.path.join(OUT_DIR, 'stage2_replace_70.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stage2_rows)
print(f"  总数: {len(stage2_rows)}")

# 阶段3: 全量
print(f"\n生成阶段3 manifest (全量)...")
stage3_rows = list(e2_rows) + full_rows
with open(os.path.join(OUT_DIR, 'stage3_full.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stage3_rows)
print(f"  总数: {len(stage3_rows)}")

print(f"\n✅ 三套manifest已生成到 {OUT_DIR}")
