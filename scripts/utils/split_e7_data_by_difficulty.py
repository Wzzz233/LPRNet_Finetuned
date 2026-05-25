#!/usr/bin/env python3
"""
E7-V2 数据难度分割脚本
"""

import os
import shutil
import csv

E7_BASE = "/home/wzzz/LPRNet/tmp/green_board_native_e7_v2"

print("开始分割E7-V2数据...")

# 读取details
with open(os.path.join(E7_BASE, "details", "details.tsv"), 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

print(f"总样本数: {len(rows)}")

# 按bucket分组
simple_rows = [r for r in rows if r.get('bucket') == 'geometry_clean']
medium_rows = [r for r in rows if r.get('bucket') == 'board_mid_occ']

print(f"  simple (geometry_clean): {len(simple_rows)}")
print(f"  medium (board_mid_occ): {len(medium_rows)}")

# 创建目录
for split in ['simple', 'medium']:
    os.makedirs(os.path.join(E7_BASE, split, 'images', 'train'), exist_ok=True)

# 复制文件并生成labels
for split_name, split_rows in [('simple', simple_rows), ('medium', medium_rows)]:
    print(f"\n处理 {split_name} 数据...")
    
    labels = []
    copied = 0
    
    for row in split_rows:
        src_path = row.get('out_img_path', '')
        text = row.get('text', '')
        
        if not src_path or not os.path.exists(src_path):
            continue
        
        filename = os.path.basename(src_path)
        dst_path = os.path.join(E7_BASE, split_name, 'images', 'train', filename)
        
        shutil.copy2(src_path, dst_path)
        copied += 1
        
        rel_path = f"images/train/{filename}"
        labels.append(f"{rel_path} {text}")
    
    # 保存labels
    label_path = os.path.join(E7_BASE, split_name, 'train_labels.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
    
    print(f"  复制: {copied}张图片")
    print(f"  labels: {len(labels)}行")

print("\n分割完成!")
