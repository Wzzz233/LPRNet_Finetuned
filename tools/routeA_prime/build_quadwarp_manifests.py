#!/usr/bin/env python3
"""Route A' Phase 0+1: Audit + Build quadwarp manifests.

Phase 0 audits:
- quad field names in real/replace manifests
- board input availability
- R50 checkpoint
- old Route A issues

Phase 1 builds:
- train_real_replace_raw_v1.csv (merged real+replace, green8)
- train_real_replace_bal31_v1.csv (per-province balanced)
- train_real_replace_r50ratio_v1.csv (R50 ratio)
- val_real_major_holdout_v1.csv (from real holdout)
- val_boardlike_province_stress_v1.csv (province stress)
- val_real_nonmajor_only_v1.csv (if enough samples)
"""
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests_rebased/routeA_prime_quadwarp_20260512'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = ROOT / 'experiments/routeA_prime_quadwarp_20260512'
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Input manifests
REAL_MANIFEST = ROOT / 'manifests_rebased/ccpd2020_green_real_20260509/train_ccpd2020_green_real.csv'
REPLACE_MANIFEST = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/train_cvreplace_v4.csv'
STRESS_MANIFEST = ROOT / 'manifests_rebased/province_stress_pose_val_v1/province_stress_pose_val_v1.csv'
R50_MANIFEST = ROOT / 'manifests_rebased/a_ratio_sweep_20260510/train_A_ratio_r50.csv'

PROVINCES = ['京','津','冀','晋','蒙','辽','吉','黑','沪','苏','浙','皖','闽','赣','鲁','豫',
             '鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新','渝']


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows, reader.fieldnames


def align_row(row, target_fields, default=''):
    """Align a row to target_fields, filling missing columns with default."""
    return {k: row.get(k, default) for k in target_fields}


def compute_summary(rows, name):
    total = len(rows)
    prov_counts = Counter()
    source_counts = Counter()
    real_count = 0
    replace_count = 0
    text_counts = Counter()
    missing_paths = 0
    missing_quads = 0

    for r in rows:
        text = (r.get('text') or '').strip()
        src = (r.get('source') or '').strip()
        img_path = (r.get('img_path') or '').strip()
        has_quad = r.get('has_quad', '0').strip() in ('1', 'True')

        if text:
            prov_counts[text[0]] += 1
            text_counts[text] += 1
        if src:
            source_counts[src] += 1
        if 'ccpd2020' in src.lower() or 'real' in src.lower():
            real_count += 1
        else:
            replace_count += 1
        if img_path and not Path(img_path).exists():
            missing_paths += 1
        if not has_quad:
            missing_quads += 1

    return {
        'name': name,
        'total': total,
        'province_count': len(prov_counts),
        'province_counts': dict(prov_counts.most_common()),
        'source_top10': dict(source_counts.most_common(10)),
        'real_count': real_count,
        'replace_count': replace_count,
        'unique_texts': len(text_counts),
        'missing_paths': missing_paths,
        'missing_quads': missing_quads,
        'families': dict(Counter(r.get('family','').strip() for r in rows).most_common()),
        'splits': dict(Counter(r.get('split','').strip() for r in rows).most_common()),
    }


def main():
    print('=' * 60)
    print('Route A\' Phase 0+1: Audit + Build Manifests')
    print('=' * 60)

    # ── Phase 0: Audit ──
    print('\n--- Phase 0: Audit ---')

    # Real manifest
    real_rows, real_fields = load_csv(REAL_MANIFEST)
    print(f'\n  REAL manifest: {len(real_rows)} rows, {len(real_fields)} fields')
    qfields = [f for f in real_fields if 'quad' in f.lower()]
    print(f'  Fields with quad:')
    print(f'    {qfields}')
    assert 'quad_1x' in real_fields and 'quad_1y' in real_fields
    assert 'quad_2x' in real_fields and 'quad_2y' in real_fields
    assert 'quad_3x' in real_fields and 'quad_3y' in real_fields
    assert 'quad_4x' in real_fields and 'quad_4y' in real_fields

    replace_rows, replace_fields = load_csv(REPLACE_MANIFEST)
    print(f'\n  REPLACE manifest: {len(replace_rows)} rows, {len(replace_fields)} fields')
    assert 'quad_1x' in replace_fields

    # R50
    r50_rows, r50_fields = load_csv(R50_MANIFEST)
    r50_ckpt = ROOT / 'experiments/a_ratio_r50_20260510/best_LPRNet_model.pth'
    assert r50_ckpt.exists()
    print(f'\n  R50 checkpoint: {r50_ckpt}')
    print(f'  R50 manifest: {len(r50_rows)} rows')

    # Board inputs
    print(f'\n  Board inputs:')
    import cv2
    for name, path, pattern in [
        ('dump2 coarse', '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2', 'coarse_*.ppm'),
        ('dump2 ocrin', '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2', 'ocrin_*.ppm'),
        ('dump1 coarse', '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump', 'coarse_*.ppm'),
        ('dump1 ocrin', '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump', 'ocrin_*.ppm'),
        ('cluster2 crop', '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2', 'crop_*.ppm'),
        ('cluster2 ocrin', '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2', 'ocrin_*.ppm'),
    ]:
        from glob import glob
        files = sorted(glob(f'{path}/{pattern}'))
        if files:
            img = cv2.imread(files[0])
            print(f'    {name}: {len(files)} files, {img.shape[1]}x{img.shape[0]}')

    # Audit output
    audit = {
        'real_manifest': {'path': str(REAL_MANIFEST), 'rows': len(real_rows), 'fields': real_fields},
        'replace_manifest': {'path': str(REPLACE_MANIFEST), 'rows': len(replace_rows), 'fields': replace_fields},
        'r50_checkpoint': str(r50_ckpt),
        'r50_manifest': str(R50_MANIFEST),
        'board_dumps': {
            'dump2': '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2',
            'dump1': '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump',
            'cluster2': '/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416',
        },
        'notes': {
            'tiny_net_not_for_prime': '旧A tiny net (3 conv layers) 不可复用为主模型',
            'best_selection_fix': '旧A按test_acc选best导致多数类塌缩，不可复用',
            'eval_mismatch_fix': '旧A4用OCRIN(24×94)评估full-crop模型，评估口径错误',
        }
    }
    (EXP_DIR / 'PHASE0_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f'\n  Audit saved: {EXP_DIR / "PHASE0_AUDIT.json"}')

    # ── Phase 1: Build manifests ──
    print('\n--- Phase 1: Build manifests ---')

    # Unified schema: combine real+replace fields
    # Core: img_path, text, family, source, split, has_quad, quad_1x..quad_4y
    unified_fields = ['img_path', 'text', 'family', 'source', 'split',
                      'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
                      'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
                      'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
                      'preprocess_group', 'quad_source', 'bbox_source']

    # 1. Merge real + replace, filter to green8
    print('\n  [1] Merging real + replace...')
    all_rows = []
    for r in real_rows:
        if r.get('family','').strip() == 'green8':
            all_rows.append(align_row(r, unified_fields))
    print(f'    Real green8: {len(all_rows)} rows')

    for r in replace_rows:
        if r.get('family','').strip() == 'green8':
            all_rows.append(align_row(r, unified_fields))
    print(f'    + Replace green8: total {len(all_rows)} rows')

    # Write raw
    raw_csv = OUT_DIR / 'train_real_replace_raw_v1.csv'
    with open(raw_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(all_rows)
    raw_summary = compute_summary(all_rows, 'train_real_replace_raw_v1')
    print(f'    Saved: {raw_csv} ({raw_summary["total"]} rows)')
    print(f'    Province top5: {list(raw_summary["province_counts"].items())[:5]}')

    # 2. Balanced per-province
    print('\n  [2] Building balanced manifest...')
    by_prov = defaultdict(list)
    for r in all_rows:
        prov = r.get('text','').strip()[0] if r.get('text','').strip() else '?'
        by_prov[prov].append(r)

    min_prov = min(len(by_prov.get(p,[])) for p in PROVINCES if p in by_prov)
    print(f'    Min per-province: {min_prov}')
    bal_rows = []
    for p in PROVINCES:
        pool = by_prov.get(p, [])
        if len(pool) <= min_prov:
            bal_rows.extend(pool[:])
        else:
            bal_rows.extend(random.sample(pool, min_prov))
    random.shuffle(bal_rows)

    bal_csv = OUT_DIR / 'train_real_replace_bal31_v1.csv'
    with open(bal_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(bal_rows)
    bal_summary = compute_summary(bal_rows, 'train_real_replace_bal31_v1')
    print(f'    Saved: {bal_csv} ({bal_summary["total"]} rows)')
    print(f'    Real/Replace: {bal_summary["real_count"]}/{bal_summary["replace_count"]}')

    # 3. R50 ratio version
    print('\n  [3] Building R50 ratio manifest...')
    # R50 ratio is 50% real / 50% replace
    real_sample = random.sample(real_rows, len(real_rows))  # all real
    replace_sample = random.sample(replace_rows, min(len(replace_rows), len(real_sample) * 2))
    # Target: approximately 50-50
    target_total = len(real_rows) * 2  # ~11,538
    n_real = min(len(real_rows), target_total // 2)
    n_replace = target_total - n_real
    r50ratio_rows = []
    for r in random.sample(real_rows, n_real):
        if r.get('family','').strip() == 'green8':
            r50ratio_rows.append(align_row(r, unified_fields))
    for r in random.sample(replace_rows, min(n_replace, len(replace_rows))):
        if r.get('family','').strip() == 'green8':
            r50ratio_rows.append(align_row(r, unified_fields))
    random.shuffle(r50ratio_rows)

    r50ratio_csv = OUT_DIR / 'train_real_replace_r50ratio_v1.csv'
    with open(r50ratio_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(r50ratio_rows)
    r50ratio_summary = compute_summary(r50ratio_rows, 'train_real_replace_r50ratio_v1')
    print(f'    Saved: {r50ratio_csv} ({r50ratio_summary["total"]} rows)')
    print(f'    Real/Replace: {r50ratio_summary["real_count"]}/{r50ratio_summary["replace_count"]}')

    # 4. Holdout from real
    print('\n  [4] Building real holdout...')
    # Real data is only 5,769 rows - take 500-1000 for holdout
    holdout_rows = []
    for r in real_rows:
        if r.get('family','').strip() == 'green8':
            holdout_rows.append(align_row(r, unified_fields))
    # Shuffle and split 80/20
    random.shuffle(holdout_rows)
    holdout_n = max(500, min(1000, len(holdout_rows) // 5))
    holdout_test = holdout_rows[:holdout_n]
    holdout_train = holdout_rows[holdout_n:]

    holdout_csv = OUT_DIR / 'val_real_major_holdout_v1.csv'
    with open(holdout_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(holdout_test)
    holdout_summary = compute_summary(holdout_test, 'val_real_major_holdout_v1')
    print(f'    Saved: {holdout_csv} ({holdout_summary["total"]} rows)')
    # Check how many non-皖
    non_wan = sum(1 for r in holdout_test if r.get('text','').strip() and r['text'][0] != '皖')
    print(f'    Non-皖 in holdout: {non_wan}')

    # 5. Province stress
    print('\n  [5] Building province stress...')
    stress_rows, stress_fields = load_csv(STRESS_MANIFEST)
    stress_aligned = [align_row(r, unified_fields) for r in stress_rows if r.get('family','').strip() == 'green8']
    stress_csv = OUT_DIR / 'val_boardlike_province_stress_v1.csv'
    with open(stress_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(stress_aligned)
    stress_summary = compute_summary(stress_aligned, 'val_boardlike_province_stress_v1')
    print(f'    Saved: {stress_csv} ({stress_summary["total"]} rows)')

    # 6. Non-major only holdout
    print('\n  [6] Building non-major holdout...')
    non_major = [r for r in holdout_test if r.get('text','').strip() and r['text'][0] != '皖']
    nonmajor_csv = OUT_DIR / 'val_real_nonmajor_only_v1.csv'
    with open(nonmajor_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=unified_fields)
        w.writeheader()
        w.writerows(non_major)
    print(f'    Saved: {nonmajor_csv} ({len(non_major)} rows non-皖 holdout)')

    # Save all summaries
    all_summaries = {
        'train_real_replace_raw_v1': raw_summary,
        'train_real_replace_bal31_v1': bal_summary,
        'train_real_replace_r50ratio_v1': r50ratio_summary,
        'val_real_major_holdout_v1': holdout_summary,
        'val_boardlike_province_stress_v1': stress_summary,
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    print(f'\n  All summaries: {OUT_DIR / "summary.json"}')

    print('\nDone.')


if __name__ == '__main__':
    random.seed(20260512)
    main()
