#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import math
from pathlib import Path

def parse_ccpd_quad_from_name(image_name):
    """从CCPD文件名解析quad"""
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 4:
        return None
    points_text = parts[3]
    points = []
    try:
        for item in points_text.split('_'):
            if '&' not in item:
                return None
            xs, ys = item.split('&', 1)
            points.append((float(xs), float(ys)))
    except ValueError:
        return None
    if len(points) != 4:
        return None
    return points

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def analyze_quad(quad):
    """分析quad的几何特征"""
    # 计算边长
    edges = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i+1)%4]
        edges.append(distance(p1, p2))
    
    top = edges[0]
    right = edges[1]
    bottom = edges[2]
    left = edges[3]
    
    # 计算宽高比
    avg_width = (top + bottom) / 2
    avg_height = (left + right) / 2
    aspect_ratio = avg_width / avg_height if avg_height > 0 else 0
    
    # 计算透视变形程度
    width_diff = abs(top - bottom) / max(top, bottom, 1e-6)
    height_diff = abs(left - right) / max(left, right, 1e-6)
    
    return {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'aspect_ratio': aspect_ratio,
        'width_diff': width_diff,
        'height_diff': height_diff,
        'warped_w': max(top, bottom),
        'warped_h': max(left, right),
    }

# 读取 manifest
manifest_path = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv'

with open(manifest_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 分析CCPD样本
print("=== CCPD (synthetic_exact_quad) 样本分析 ===\n")
ccpd_rows = [r for r in rows if r.get('source') == 'synthetic_exact_quad' and r.get('split') == 'train'][:5]
for r in ccpd_rows:
    img_path = r.get('img_path', '')
    text = r.get('text', '')
    quad = parse_ccpd_quad_from_name(img_path)
    if quad:
        analysis = analyze_quad(quad)
        print(f"Text: {text}")
        print(f"Path: {Path(img_path).name}")
        print(f"Quad: {[f'({p[0]:.1f},{p[1]:.1f})' for p in quad]}")
        print(f"  边长: T={analysis['top']:.1f}, B={analysis['bottom']:.1f}, L={analysis['left']:.1f}, R={analysis['right']:.1f}")
        print(f"  宽高比: {analysis['aspect_ratio']:.2f}")
        print(f"  透视变换后尺寸: {analysis['warped_w']:.0f} x {analysis['warped_h']:.0f}")
        print(f"  与94x24目标比: W_ratio={analysis['warped_w']/94:.2f}, H_ratio={analysis['warped_h']/24:.2f}")
        print()

print("\n=== Edgefit 样本分析 ===\n")
edgefit_rows = [r for r in rows if 'edgefit' in r.get('source', '') and r.get('split') == 'train'][:5]
for r in edgefit_rows:
    img_path = r.get('img_path', '')
    text = r.get('text', '')
    quad = parse_ccpd_quad_from_name(img_path)
    if quad:
        analysis = analyze_quad(quad)
        print(f"Text: {text}")
        print(f"Path: {Path(img_path).name}")
        print(f"Quad: {[f'({p[0]:.1f},{p[1]:.1f})' for p in quad]}")
        print(f"  边长: T={analysis['top']:.1f}, B={analysis['bottom']:.1f}, L={analysis['left']:.1f}, R={analysis['right']:.1f}")
        print(f"  宽高比: {analysis['aspect_ratio']:.2f}")
        print(f"  透视变换后尺寸: {analysis['warped_w']:.0f} x {analysis['warped_h']:.0f}")
        print(f"  与94x24目标比: W_ratio={analysis['warped_w']/94:.2f}, H_ratio={analysis['warped_h']/24:.2f}")
        print()

print("\n=== 关键差异分析 ===")
print("CCPD synthetic_exact_quad 的quad通常更规整，接近矩形")
print("Edgefit 的quad有明显的透视变形（上下边长度差异大）")
print("透视变换后，两者都会被矫正为矩形")
print("但edgefit的原始形变更极端，可能导致字符变形更严重")
