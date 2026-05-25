#!/usr/bin/env python3
"""Merge E6A single-yaw and single-pitch datasets into one single-axis-visible pool.

Also reassign tiers by angle_score so merged output matches target train/proxy tier counts.
"""
import csv, json, shutil
from collections import Counter
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
YAW = ROOT / 'tmp/green_extreme_stageB1A_E6A_single_yaw_visible_20260427'
PITCH = ROOT / 'tmp/green_extreme_stageB1A_E6A_single_pitch_visible_20260427'
OUT = ROOT / 'tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427'
TARGETS = {
    'train': {'low': 100, 'mid': 120, 'high': 80},
    'proxy': {'low': 36, 'mid': 48, 'high': 40},
}
EXPECTED_TRAIN = {'云': 10, '京': 10, '冀': 10, '吉': 10, '宁': 10, '川': 10, '新': 10, '晋': 10, '桂': 10, '沪': 11, '津': 10, '浙': 10, '渝': 10, '湘': 10, '琼': 10, '甘': 10, '皖': 8, '粤': 10, '苏': 10, '蒙': 10, '藏': 10, '豫': 9, '贵': 9, '赣': 9, '辽': 9, '鄂': 9, '闽': 9, '陕': 10, '青': 9, '鲁': 9, '黑': 9}


def load_records(root):
    return json.loads((root / 'generation_meta.json').read_text(encoding='utf-8'))['records']


def copy_records(records, src_root, out_root):
    copied = []
    for rec in records:
        src = src_root / rec['file']
        dst = out_root / rec['file']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dict(rec))
    return copied


def reassign_tiers(records):
    for split in ['train', 'proxy']:
        subset = [r for r in records if r['split'] == split]
        subset.sort(key=lambda r: (float(r['angle_score']), float(r['angle_mean']), r['direction'], r['file']))
        low_n = TARGETS[split]['low']
        mid_n = TARGETS[split]['mid']
        for i, rec in enumerate(subset):
            if i < low_n:
                rec['tier'] = 'low'
            elif i < low_n + mid_n:
                rec['tier'] = 'mid'
            else:
                rec['tier'] = 'high'


def rewrite_paths(records):
    for rec in records:
        path = Path(rec['file'])
        rec['file'] = str(Path('images') / rec['split'] / rec['tier'] / path.name)


def move_images(records):
    for rec in records:
        old_candidates = [YAW / rec['file'], PITCH / rec['file'], OUT / rec['file']]
        src = next((p for p in old_candidates if p.exists()), None)
        if src is None:
            # find by basename after tier reassignment
            matches = list((OUT / 'images' / rec['split']).rglob(Path(rec['file']).name))
            if not matches:
                raise SystemExit(f'[FATAL] missing merged image basename={Path(rec["file"]).name}')
            src = matches[0]
        dst = OUT / rec['file']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst:
            shutil.move(str(src), str(dst))


def write_metrics(records):
    keys = ['file', 'split', 'tier', 'direction', 'province', 'text', 'source_exact', 'ratio', 'area', 'min_edge', 'angle_score', 'angle_mean', 'top_abs', 'bottom_abs', 'left_dev', 'right_dev', 'attempts']
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in records])


def cleanup_empty_dirs(root):
    for p in sorted(root.rglob('*'), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True)
    yaw_records = load_records(YAW)
    pitch_records = load_records(PITCH)
    records = []
    records += copy_records(yaw_records, YAW, OUT)
    records += copy_records(pitch_records, PITCH, OUT)

    got_train = Counter(r['province'] for r in records if r['split'] == 'train')
    if dict(sorted(got_train.items())) != EXPECTED_TRAIN:
        raise SystemExit(json.dumps({'fatal': 'train_province_quota_mismatch', 'expected': EXPECTED_TRAIN, 'got': dict(sorted(got_train.items()))}, ensure_ascii=False, indent=2))
    got_proxy = Counter(r['province'] for r in records if r['split'] == 'proxy')
    if any(v != 4 for v in got_proxy.values()) or len(got_proxy) != 31:
        raise SystemExit(json.dumps({'fatal': 'proxy_province_quota_mismatch', 'got': dict(sorted(got_proxy.items()))}, ensure_ascii=False, indent=2))

    reassign_tiers(records)
    rewrite_paths(records)
    move_images(records)
    cleanup_empty_dirs(OUT / 'images')

    meta = {
        'count': len(records),
        'out': str(OUT),
        'source_roots': [str(YAW), str(PITCH)],
        'records': records,
        'summary': {
            'by_split': dict(Counter(r['split'] for r in records)),
            'by_split_tier': dict(Counter(f"{r['split']}:{r['tier']}" for r in records)),
            'by_split_direction': dict(Counter(f"{r['split']}:{r['direction']}" for r in records)),
            'by_split_province': dict(Counter(f"{r['split']}:{r['province']}" for r in records)),
        },
    }
    (OUT / 'generation_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    write_metrics(records)
    print(json.dumps({'merged': len(records), 'out': str(OUT), 'summary': meta['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
