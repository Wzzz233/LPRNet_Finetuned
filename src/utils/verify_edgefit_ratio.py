#!/usr/bin/env python3
"""
验证新的 edgefit 生成器是否产生正确的宽高比
"""
import json
import math
from pathlib import Path

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def analyze_quad(quad):
    """分析 quad 在透视变换后的宽高比"""
    top = distance(quad[0], quad[1])
    bottom = distance(quad[3], quad[2])
    left = distance(quad[0], quad[3])
    right = distance(quad[1], quad[2])
    
    warped_w = max(top, bottom)
    warped_h = max(left, right)
    aspect_ratio = warped_w / warped_h if warped_h > 0 else 0
    
    # 计算与 94x24 (ratio=3.92) 的差异
    target_ratio = 94 / 24
    ratio_diff = abs(aspect_ratio - target_ratio)
    
    # 预计 letterbox 占用比
    scale_w = 94 / warped_w
    scale_h = 24 / warped_h
    scale = min(scale_w, scale_h)
    occ_ratio = (warped_w * scale) / 94
    
    return {
        'warped_w': warped_w,
        'warped_h': warped_h,
        'aspect_ratio': aspect_ratio,
        'ratio_diff': ratio_diff,
        'occ_ratio': occ_ratio,
        'has_black_border': ratio_diff > 0.5,
    }

# 读取生成的数据
tsv_path = Path('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/details/accepted.tsv')

print("=== 验证旧版 Edgefit (H34C) ===\n")

with open(tsv_path, 'r') as f:
    import csv
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

# 统计 harder 样本的宽高比
print("Harder samples aspect ratio distribution:")
ratios = []
for r in rows:
    if r['difficulty'] == 'harder' and r['split'] == 'train':
        quad = json.loads(r['quad'])
        analysis = analyze_quad(quad)
        ratios.append(analysis['aspect_ratio'])

if ratios:
    print(f"  Count: {len(ratios)}")
    print(f"  Min: {min(ratios):.2f}")
    print(f"  Max: {max(ratios):.2f}")
    print(f"  Mean: {sum(ratios)/len(ratios):.2f}")
    
    # 分布统计
    bins = [(0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 10)]
    for lo, hi in bins:
        count = sum(1 for r in ratios if lo <= r < hi)
        print(f"  Ratio {lo}-{hi}: {count} ({100*count/len(ratios):.1f}%)")

print("\n=== 目标（CCPD Real）===")
print("  Aspect ratio: 2.5-3.1")
print("  Expected black borders: YES")

print("\n=== 对比 ===")
print("旧版 Edgefit: 宽高比 3.5-4.0 → 无黑边")
print("CCPD Real:    宽高比 2.5-3.1 → 有黑边")
print("目标:         让 Edgefit harder 产生 2.5-3.5 的宽高比")
