#!/usr/bin/env python3
"""
步骤一：数据审计与标准化 label 清单生成
为所有数据源输出统一的 (img_path, text, family, source, split) CSV
"""

import csv
import glob
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

OUT_DIR = Path("labels/curriculum_gray3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CCPD 解码表
CCPD_PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                  '苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤',
                  '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
CCPD_ALPHABETS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
CCPD_ADS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '0','1','2','3','4','5','6','7','8','9']

def decode_ccpd_name(name: str) -> str:
    """从 CCPD 文件名解析车牌文字"""
    stem = name.replace('.jpg', '')
    parts = stem.split('-')
    codes = [int(x) for x in parts[-3].split('_')]
    result = CCPD_PROVINCES[codes[0]]
    result += CCPD_ALPHABETS[codes[1]]
    for c in codes[2:]:
        result += CCPD_ADS[c]
    return result


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['img_path', 'text', 'family', 'source', 'split'])
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {len(rows)} rows -> {path}")


def dedupe_by_image(rows):
    """按图像路径去重"""
    seen = set()
    out = []
    for r in rows:
        key = r[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_ccpd2019():
    """CCPD2019: 读取 prepared_labels"""
    print("\n[1/6] CCPD2019")
    base = Path("datasets/CCPD2019")
    
    for split in ['train', 'val', 'test']:
        src = Path(f"prepared_labels/ccpd2019/{split}_labels.txt")
        rows = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                rel_path = parts[0]
                text = parts[1]
                img_path = str((base / rel_path).resolve())
                rows.append((img_path, text, 'normal7', 'ccpd2019', split))
        write_csv(rows, OUT_DIR / f"ccpd2019_{split}.csv")
        print(f"    {split}: {len(rows)} samples")


def load_ccpd2020():
    """CCPD2020 green: 从文件名解析"""
    print("\n[2/6] CCPD2020 green")
    base = Path("datasets/CCPD2020/ccpd_green")
    
    for split in ['train', 'val', 'test']:
        img_dir = base / split
        rows = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            text = decode_ccpd_name(img_path.name)
            rows.append((str(img_path.resolve()), text, 'green8', 'ccpd2020', split))
        write_csv(rows, OUT_DIR / f"ccpd2020_{split}.csv")
        print(f"    {split}: {len(rows)} samples")


def load_cblprd():
    """CBLPRD cvcrop: 从文件名解析 text 和 type"""
    print("\n[3/6] CBLPRD cvcrop")
    
    # 定义接受的牌型 -> family
    TYPE_MAP = {
        '普通蓝牌': 'normal7',
        '黑色车牌': 'normal7',
        '新能源小型车': 'green8',
    }
    
    for split in ['train', 'val']:
        src_dir = Path(f"datasets/CBLPRD-330k_v1/cblprd_cv_geom/{split}")
        rows_blue = []
        rows_green = []
        
        for img_path in sorted(src_dir.glob("*.jpg")):
            name = img_path.name
            # 解析文件名: cvcrop-{quad}-{text}-{type}-{flag}-{orig_id}.jpg
            # orig_id 中可能含 '-' (如 CBLPRD-330k)，需从已知 type 反向定位
            plate_type = None
            for t in TYPE_MAP:
                if t in name:
                    plate_type = t
                    idx = name.index(t)
                    before = name[:idx].rstrip('-')
                    text = before.rsplit('-', 1)[-1]
                    break
            if plate_type is None:
                continue
            
            family = TYPE_MAP[plate_type]
            
            # 文字长度校验
            if family == 'normal7' and len(text) != 7:
                continue
            if family == 'green8' and len(text) != 8:
                continue
            
            # 验证文字只含有效字符
            valid_chars = set(CCPD_PROVINCES + list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'))
            if not all(c in valid_chars for c in text):
                continue
            
            row = (str(img_path.resolve()), text, family, 'cblprd', split)
            if family == 'normal7':
                rows_blue.append(row)
            else:
                rows_green.append(row)
        
        write_csv(rows_blue, OUT_DIR / f"cblprd_blue_{split}.csv")
        write_csv(rows_green, OUT_DIR / f"cblprd_green_{split}.csv")
        print(f"    {split}: blue={len(rows_blue)} green={len(rows_green)}")


def load_crpd():
    """CRPD_all: 从 label txt 读取"""
    print("\n[4/6] CRPD_all")
    base = Path("datasets/CRPD_all")
    
    for subset in ['CRPD_single', 'CRPD_double', 'CRPD_multi']:
        for split in ['train', 'val', 'test']:
            label_dir = base / subset / split / "labels"
            img_dir = base / subset / split / "images"
            if not label_dir.exists():
                continue
            
            rows = []
            for label_file in sorted(label_dir.glob("*.txt")):
                with open(label_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 10:
                            continue
                        text = parts[-1]
                        
                        # 只保留标准蓝牌 (7字符) 或绿牌 (8字符)
                        if len(text) == 7:
                            family = 'normal7'
                        elif len(text) == 8:
                            family = 'green8'
                        else:
                            continue
                        
                        # 找对应的图片
                        img_name = label_file.stem + ".jpg"
                        img_path = img_dir / img_name
                        if not img_path.exists():
                            # 尝试 png
                            img_path = img_dir / (label_file.stem + ".png")
                        
                        if img_path.exists():
                            rows.append((str(img_path.resolve()), text, family, f'crpd_{subset.lower()}', split))
            
            if rows:
                write_csv(rows, OUT_DIR / f"crpd_{subset.lower()}_{split}.csv")
                print(f"    {subset}/{split}: {len(rows)} samples")


def load_green_exact_quad():
    """green_exact_quad_synthetic_v1"""
    print("\n[5/6] green_exact_quad")
    base = Path("datasets/green_exact_quad_synthetic_v1")
    
    for split in ['train', 'val', 'test']:
        src = base / "manifests" / f"{split}_synthetic_labels.txt"
        rows = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                rel_path = parts[0]
                text = parts[1]
                img_path = str((base / rel_path).resolve())
                rows.append((img_path, text, 'green8', 'green_exact_quad', split))
        write_csv(rows, OUT_DIR / f"green_exact_quad_{split}.csv")
        print(f"    {split}: {len(rows)} samples")


def load_green_edgefit():
    """green_edgefit_tier3_full_v2"""
    print("\n[6/6] green_edgefit_tier3")
    base = Path("datasets/green_edgefit_tier3_full_v2")
    
    for split in ['train', 'val', 'test']:
        src = base / "manifests" / f"{split}_labels.txt"
        rows_simple = []
        rows_hard = []
        rows_extreme = []
        
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                rel_path = parts[0]
                text = parts[1]
                img_path = str((base / rel_path).resolve())
                
                # 从路径判断 tier
                if '/simple/' in rel_path:
                    rows_simple.append((img_path, text, 'green8', 'green_edgefit_simple', split))
                elif '/hard/' in rel_path:
                    rows_hard.append((img_path, text, 'green8', 'green_edgefit_hard', split))
                elif '/extreme/' in rel_path:
                    rows_extreme.append((img_path, text, 'green8', 'green_edgefit_extreme', split))
        
        write_csv(rows_simple, OUT_DIR / f"green_edgefit_simple_{split}.csv")
        write_csv(rows_hard, OUT_DIR / f"green_edgefit_hard_{split}.csv")
        write_csv(rows_extreme, OUT_DIR / f"green_edgefit_extreme_{split}.csv")
        print(f"    {split}: simple={len(rows_simple)} hard={len(rows_hard)} extreme={len(rows_extreme)}")


def audit_all():
    """全局审计：泄露检查 + 省份统计"""
    print("\n" + "="*60)
    print("[AUDIT] 全局审计")
    print("="*60)
    
    # 收集所有文件
    all_files = list(OUT_DIR.glob("*.csv"))
    
    # 按 split 聚合
    splits = defaultdict(list)
    for csv_file in all_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row['split']
                splits[split].append(row)
    
    # 图像级泄露检查
    print("\n[Leakage Check] 图像级泄露检查")
    train_imgs = set(r['img_path'] for r in splits.get('train', []))
    val_imgs = set(r['img_path'] for r in splits.get('val', []))
    test_imgs = set(r['img_path'] for r in splits.get('test', []))
    
    train_val_overlap = train_imgs & val_imgs
    train_test_overlap = train_imgs & test_imgs
    val_test_overlap = val_imgs & test_imgs
    
    print(f"  train ∩ val:   {len(train_val_overlap)}")
    print(f"  train ∩ test:  {len(train_test_overlap)}")
    print(f"  val ∩ test:    {len(val_test_overlap)}")
    
    if train_val_overlap:
        print("  WARNING: train/val 有重叠！")
    if train_test_overlap:
        print("  WARNING: train/test 有重叠！")
    if val_test_overlap:
        print("  WARNING: val/test 有重叠！")
    
    # 省份统计
    print("\n[Province Stats] 蓝牌 (normal7)")
    for split_name in ['train', 'val', 'test']:
        rows = [r for r in splits.get(split_name, []) if r['family'] == 'normal7']
        provs = Counter(r['text'][0] for r in rows)
        total = len(rows)
        print(f"\n  {split_name}: total={total}")
        for prov, cnt in provs.most_common(10):
            pct = cnt / total * 100
            print(f"    {prov}: {cnt} ({pct:.1f}%)")
    
    print("\n[Province Stats] 绿牌 (green8)")
    for split_name in ['train', 'val', 'test']:
        rows = [r for r in splits.get(split_name, []) if r['family'] == 'green8']
        provs = Counter(r['text'][0] for r in rows)
        total = len(rows)
        print(f"\n  {split_name}: total={total}")
        for prov, cnt in provs.most_common(10):
            pct = cnt / total * 100
            print(f"    {prov}: {cnt} ({pct:.1f}%)")
        # 均衡性指标
        if total > 0:
            max_cnt = max(provs.values())
            min_cnt = min(provs.values())
            print(f"    max/min ratio: {max_cnt/min_cnt:.2f}x")


def main():
    print("="*60)
    print("步骤一：数据审计与标准化 label 清单生成")
    print("="*60)
    
    load_ccpd2019()
    load_ccpd2020()
    load_cblprd()
    load_crpd()
    load_green_exact_quad()
    load_green_edgefit()
    
    audit_all()
    
    print("\n" + "="*60)
    print("步骤一完成。输出目录: labels/curriculum_gray3/")
    print("="*60)


if __name__ == '__main__':
    main()
