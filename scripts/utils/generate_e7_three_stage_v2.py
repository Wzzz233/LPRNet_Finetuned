#!/usr/bin/env python3
import csv
import os

E2_MANIFEST = "/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv"
E7_FULL_MANIFEST = "/home/wzzz/LPRNet/manifests/e7_three_stage_v2/e7_full.csv"
OUT_DIR = "/home/wzzz/LPRNet/manifests/e7_three_stage_v2"

key_provinces = ['苏', '沪', '浙']

# 读取E2
with open(E2_MANIFEST, 'r') as f:
    e2_rows = list(csv.DictReader(f))
    fieldnames = list(e2_rows[0].keys())

# 读取E7-full
with open(E7_FULL_MANIFEST, 'r') as f:
    e7_rows = list(csv.DictReader(f))

# 分离E2
e2_keyprov = [r for r in e2_rows if r.get('text', '')[0] in key_provinces and r.get('family') == 'green8']
e2_other = [r for r in e2_rows if not (r.get('text', '')[0] in key_provinces and r.get('family') == 'green8')]

# 按省份分组E2
e2_by_prov = {prov: [r for r in e2_keyprov if r.get('text', '')[0] == prov] for prov in key_provinces}

# 按省份分组E7
e7_by_prov = {prov: [r for r in e7_rows if r.get('text', '')[0] == prov] for prov in key_provinces}

print("数据分布:")
for prov in key_provinces:
    print(f"  {prov}: E2={len(e2_by_prov[prov])}, E7={len(e7_by_prov[prov])}")

# 计算替据比例
def generate_stage(e2_keep_ratio, stage_name):
    """生成某阶段的manifest"""
    stage_rows = list(e2_other)  # 保留所有非关键省份数据
    
    for prov in key_provinces:
        e2_prov = e2_by_prov[prov]
        e7_prov = e7_by_prov[prov]
        
        # 计算需要保留的E2数量
        keep_count = int(len(e2_prov) * e2_keep_ratio)
        keep_count = max(1, min(keep_count, len(e2_prov)))
        
        # 保留E2 + 全部E7
        stage_rows.extend(e2_prov[:keep_count])
        stage_rows.extend(e7_prov)
        
        total = keep_count + len(e7_prov)
        e7_ratio = len(e7_prov) / total * 100 if total > 0 else 0
        print(f"  {prov}: 保留{keep_count}张E2 + {len(e7_prov)}张E7, E7占{e7_ratio:.1f}%")
    
    # 保存
    out_path = os.path.join(OUT_DIR, f"{stage_name}.csv")
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stage_rows)
    
    print(f"  总数: {len(stage_rows)} → {out_path}\n")
    return len(stage_rows)

print("\n生成阶段1 (E2保留10% = E7占60%)...")
generate_stage(0.1, "stage1_v2")

print("生成阶段2 (E2保留30% = E7占40%)...")
generate_stage(0.3, "stage2_v2")

print("生成阶段3 (全量数据)...")
stage3_rows = list(e2_rows) + list(e7_rows)
with open(os.path.join(OUT_DIR, "stage3_v2.csv"), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stage3_rows)
print(f"  总数: {len(stage3_rows)} → {os.path.join(OUT_DIR, 'stage3_v2.csv')}")

print("\n✅ 修正版manifest生成完成!")
