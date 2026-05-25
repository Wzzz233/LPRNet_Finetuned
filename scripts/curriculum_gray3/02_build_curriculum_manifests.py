#!/usr/bin/env python3
"""
步骤二：构建 Curriculum Manifests
- Stage A train: 简单样本基底
- Val: 超参调优/early stopping
- Test: 多档 held-out 评测（绝不进入训练）
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

LABEL_DIR = Path("labels/curriculum_gray3")
OUT_DIR = Path("manifests/curriculum_gray3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def write_manifest(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['img_path', 'text', 'family', 'source', 'split'])
        for r in rows:
            w.writerow([r['img_path'], r['text'], r['family'], r['source'], r['split']])
    print(f"  Wrote {len(rows)} rows -> {path}")


def province_cap_sample(rows, cap_dict, seed=42):
    """按省份下采样"""
    rng = random.Random(seed)
    by_prov = defaultdict(list)
    for r in rows:
        by_prov[r['text'][0]].append(r)
    
    out = []
    for prov, items in by_prov.items():
        cap = cap_dict.get(prov, float('inf'))
        if len(items) > cap:
            items = rng.sample(items, cap)
        out.extend(items)
    return out


def build_stageA_train():
    print("\n[Build] Stage A Train Manifest")
    
    # --- 蓝牌池 ---
    ccpd_blue = load_csv(LABEL_DIR / 'ccpd2019_train.csv')
    cblprd_blue = load_csv(LABEL_DIR / 'cblprd_blue_train.csv')
    crpd_single = load_csv(LABEL_DIR / 'crpd_crpd_single_train.csv')
    crpd_double = load_csv(LABEL_DIR / 'crpd_crpd_double_train.csv')
    crpd_multi = load_csv(LABEL_DIR / 'crpd_crpd_multi_train.csv')
    
    # 过滤CRPD特殊车牌（警、学、挂、领、使、字等不在标准字符集的车牌）
    SPECIAL_CHARS = set('警学挂领使字')
    def filter_crpd(rows):
        filtered = []
        for r in rows:
            if r['family'] == 'normal7' and not any(c in r['text'] for c in SPECIAL_CHARS):
                filtered.append(r)
        return filtered
    
    crpd_single_f = filter_crpd(crpd_single)
    crpd_double_f = filter_crpd(crpd_double)
    crpd_multi_f = filter_crpd(crpd_multi)
    print(f"  CRPD filtered: single {len(crpd_single)}->{len(crpd_single_f)}, double {len(crpd_double)}->{len(crpd_double_f)}, multi {len(crpd_multi)}->{len(crpd_multi_f)}")
    
    # 只给CBLPRD做皖cap（CCPD2019和CRPD保留全部，让inv_sqrt在采样阶段均衡）
    cblprd_blue_capped = province_cap_sample(cblprd_blue, {'皖': 5000}, seed=42)
    print(f"  CBLPRD blue after 皖cap=5000: {len(cblprd_blue_capped)} (raw={len(cblprd_blue)})")
    
    blue_all = []
    blue_all.extend([r for r in ccpd_blue if r['family'] == 'normal7'])      # CCPD2019 全部保留
    blue_all.extend(cblprd_blue_capped)                                        # CBLPRD 皖cap后
    blue_all.extend(crpd_single_f)                                             # CRPD 过滤后全部保留
    blue_all.extend(crpd_double_f)
    blue_all.extend(crpd_multi_f)
    
    print(f"  Blue pool total: {len(blue_all)} (CCPD={len([r for r in ccpd_blue if r['family']=='normal7'])}, CBLPRD={len(cblprd_blue_capped)}, CRPD={len(crpd_single_f)+len(crpd_double_f)+len(crpd_multi_f)})")
    
    # 随机下采样至目标 80k（如果需要）
    if len(blue_all) > 80000:
        blue_final = random.sample(blue_all, 80000)
    else:
        blue_final = blue_all
    print(f"  Blue final: {len(blue_final)}")
    
    # --- 绿牌池 ---
    cblprd_green = load_csv(LABEL_DIR / 'cblprd_green_train.csv')
    ccpd2020_green = load_csv(LABEL_DIR / 'ccpd2020_train.csv')
    green_exact = load_csv(LABEL_DIR / 'green_exact_quad_train.csv')
    green_edgefit = load_csv(LABEL_DIR / 'green_edgefit_simple_train.csv')
    
    # 绿牌：CCPD2020做皖cap，其余保留
    ccpd2020_green_capped = province_cap_sample(ccpd2020_green, {'皖': 3000}, seed=42)
    print(f"  CCPD2020 green after 皖cap=3000: {len(ccpd2020_green_capped)} (raw={len(ccpd2020_green)})")
    
    green_all = []
    green_all.extend([r for r in cblprd_green if r['family'] == 'green8'])      # CBLPRD green 全部
    green_all.extend(ccpd2020_green_capped)                                      # CCPD2020 green 皖cap后
    green_all.extend([r for r in green_exact if r['family'] == 'green8'])       # 生成数据全部
    green_all.extend([r for r in green_edgefit if r['family'] == 'green8'])
    
    print(f"  Green pool total: {len(green_all)}")
    
    # 随机下采样至目标 80k
    if len(green_all) > 80000:
        green_final = random.sample(green_all, 80000)
    else:
        green_final = green_all
    print(f"  Green final: {len(green_final)}")
    
    # 合并
    stageA = blue_final + green_final
    random.shuffle(stageA)
    write_manifest(stageA, OUT_DIR / 'train_stageA.csv')
    
    # 统计
    print(f"\n  Stage A total: {len(stageA)}")
    print(f"    normal7: {sum(1 for r in stageA if r['family']=='normal7')}")
    print(f"    green8:  {sum(1 for r in stageA if r['family']=='green8')}")
    
    # 省份统计
    blue_provs = Counter(r['text'][0] for r in stageA if r['family']=='normal7')
    green_provs = Counter(r['text'][0] for r in stageA if r['family']=='green8')
    print(f"\n  Blue province top5: {blue_provs.most_common(5)}")
    print(f"  Green province top5: {green_provs.most_common(5)}")
    if green_provs:
        print(f"  Green max/min ratio: {max(green_provs.values())/max(1,min(green_provs.values())):.2f}x")


def build_val():
    print("\n[Build] Val Manifest")
    
    # 过滤CRPD特殊车牌的通用函数
    SPECIAL_CHARS = set('警学挂领使字')
    def filter_special(rows):
        return [r for r in rows if not any(c in r['text'] for c in SPECIAL_CHARS)]
    
    # --- 蓝牌 val ---
    ccpd_val = load_csv(LABEL_DIR / 'ccpd2019_val.csv')
    ccpd_val_blue = [r for r in ccpd_val if r['family'] == 'normal7']
    ccpd_val_sampled = province_cap_sample(ccpd_val_blue, {p: 330 for p in set(r['text'][0] for r in ccpd_val_blue)}, seed=42)
    if len(ccpd_val_sampled) > 10000:
        ccpd_val_sampled = random.sample(ccpd_val_sampled, 10000)
    print(f"  CCPD2019 val sampled: {len(ccpd_val_sampled)}")
    
    cblprd_blue_val = load_csv(LABEL_DIR / 'cblprd_blue_val.csv')
    crpd_single_val = filter_special(load_csv(LABEL_DIR / 'crpd_crpd_single_val.csv'))
    crpd_double_val = filter_special(load_csv(LABEL_DIR / 'crpd_crpd_double_val.csv'))
    crpd_multi_val = filter_special(load_csv(LABEL_DIR / 'crpd_crpd_multi_val.csv'))
    print(f"  CRPD val filtered: single+double+multi={len(crpd_single_val)+len(crpd_double_val)+len(crpd_multi_val)}")
    
    blue_sources = [
        ccpd_val_sampled,
        cblprd_blue_val,
        crpd_single_val,
        crpd_double_val,
        crpd_multi_val,
    ]
    blue_all = []
    for src in blue_sources:
        blue_all.extend([r for r in src if r['family'] == 'normal7'])
    
    # 下采样至 6k
    if len(blue_all) > 6000:
        blue_final = random.sample(blue_all, 6000)
    else:
        blue_final = blue_all
    print(f"  Blue val final: {len(blue_final)}")
    
    # --- 绿牌 val ---
    green_sources = [
        load_csv(LABEL_DIR / 'cblprd_green_val.csv'),
        load_csv(LABEL_DIR / 'ccpd2020_val.csv'),
        load_csv(LABEL_DIR / 'green_exact_quad_val.csv'),
        load_csv(LABEL_DIR / 'green_edgefit_simple_val.csv'),
    ]
    green_all = []
    for src in green_sources:
        green_all.extend([r for r in src if r['family'] == 'green8'])
    
    # 下采样至 6k
    if len(green_all) > 6000:
        green_final = random.sample(green_all, 6000)
    else:
        green_final = green_all
    print(f"  Green val final: {len(green_final)}")
    
    val = blue_final + green_final
    random.shuffle(val)
    write_manifest(val, OUT_DIR / 'val.csv')
    
    print(f"\n  Val total: {len(val)}")
    print(f"    normal7: {sum(1 for r in val if r['family']=='normal7')}")
    print(f"    green8:  {sum(1 for r in val if r['family']=='green8')}")


def build_test():
    print("\n[Build] Test Manifests (多档)")
    
    # 1. 蓝牌简单: CCPD2019 base val 剩余（未进入 val 的部分）
    ccpd_val_all = load_csv(LABEL_DIR / 'ccpd2019_val.csv')
    ccpd_val_used = load_csv(OUT_DIR / 'val.csv')
    ccpd_val_used_paths = set(r['img_path'] for r in ccpd_val_used if r['source'] == 'ccpd2019')
    blue_simple = [r for r in ccpd_val_all if r['family']=='normal7' and r['img_path'] not in ccpd_val_used_paths]
    # 省份均衡下采样至 10k
    blue_simple = province_cap_sample(blue_simple, {p: 330 for p in set(r['text'][0] for r in blue_simple)}, seed=43)
    if len(blue_simple) > 10000:
        blue_simple = random.sample(blue_simple, 10000)
    write_manifest(blue_simple, OUT_DIR / 'test_blue_simple.csv')
    
    # 2. 蓝牌 hard: CCPD2019 hard test
    blue_hard = load_csv(LABEL_DIR / 'ccpd2019_test.csv')
    write_manifest(blue_hard, OUT_DIR / 'test_blue_hard.csv')
    
    # 3. 绿牌真实: CCPD2020 green test + CBLPRD green val
    green_real = load_csv(LABEL_DIR / 'ccpd2020_test.csv') + load_csv(LABEL_DIR / 'cblprd_green_val.csv')
    green_real = [r for r in green_real if r['family'] == 'green8']
    write_manifest(green_real, OUT_DIR / 'test_green_real.csv')
    
    # 4. 绿牌 simple
    green_simple = load_csv(LABEL_DIR / 'green_exact_quad_test.csv') + load_csv(LABEL_DIR / 'green_edgefit_simple_test.csv')
    green_simple = [r for r in green_simple if r['family'] == 'green8']
    write_manifest(green_simple, OUT_DIR / 'test_green_simple.csv')
    
    # 5. 绿牌 hard
    green_hard = load_csv(LABEL_DIR / 'green_edgefit_hard_test.csv')
    write_manifest(green_hard, OUT_DIR / 'test_green_hard.csv')
    
    # 6. 绿牌 extreme
    green_extreme = load_csv(LABEL_DIR / 'green_edgefit_extreme_test.csv')
    write_manifest(green_extreme, OUT_DIR / 'test_green_extreme.csv')


def leakage_check():
    print("\n[Audit] Manifest 泄露检查")
    
    train_paths = set(r['img_path'] for r in load_csv(OUT_DIR / 'train_stageA.csv'))
    val_paths = set(r['img_path'] for r in load_csv(OUT_DIR / 'val.csv'))
    
    test_files = [
        OUT_DIR / 'test_blue_simple.csv',
        OUT_DIR / 'test_blue_hard.csv',
        OUT_DIR / 'test_green_real.csv',
        OUT_DIR / 'test_green_simple.csv',
        OUT_DIR / 'test_green_hard.csv',
        OUT_DIR / 'test_green_extreme.csv',
    ]
    
    for tf in test_files:
        if not tf.exists():
            continue
        test_paths = set(r['img_path'] for r in load_csv(tf))
        t_v = len(train_paths & test_paths)
        v_t = len(val_paths & test_paths)
        print(f"  {tf.name}: train∩test={t_v}, val∩test={v_t}")
        if t_v > 0 or v_t > 0:
            print(f"    WARNING: 泄露检测到！")


def main():
    print("="*60)
    print("步骤二：构建 Curriculum Manifests")
    print("="*60)
    
    build_stageA_train()
    build_val()
    build_test()
    leakage_check()
    
    print("\n" + "="*60)
    print("步骤二完成。输出目录: manifests/curriculum_gray3/")
    print("="*60)


if __name__ == '__main__':
    main()
