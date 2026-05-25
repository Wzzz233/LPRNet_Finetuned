#!/usr/bin/env python3
"""Generate StageB1A-E2 board-extreme data with coupled yaw+pitch perspective.

Target morphology from user guidance:
- both left/right and up/down tilt are present (3D perspective rather than pure 2D roll)
- near side border is thicker, far side is thinner
- rear half is locally compressed so tail characters look crowded
- crop remains board-warp recoverable for the existing ccpd_board toolchain
"""
import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/wzzz/LPRNet')
EXACT_DIR = ROOT / 'datasets/green_exact_quad_synthetic_v1'
LABELS = EXACT_DIR / 'manifests/train_synthetic_labels.txt'
BG_DIR = ROOT / 'tmp/green_extreme_pathB_probe_v2_stronger_20260425/backgrounds'
STATS = ROOT / 'reports/stageB_extreme_detector_vs_gt_stats_20260426/green_det_vs_gt_rows_valid.csv'
OUT = ROOT / 'tmp/green_extreme_stageB1A_E2_perspective_both_tightrear_20260427'
PLATE = (246, 72)
CANVAS = (460, 300)
TIERS = ['mid', 'high']
DIRECTIONS = ['left_up', 'left_down', 'right_up', 'right_down']
PROVINCES = list('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
TIER_PARAMS = {
    'mid': dict(
        ratio=(1.90, 2.30), height=(84, 106), yaw_side=(0.58, 0.78), pitch_far=(0.78, 0.92),
        pitch_near=(1.02, 1.14), shear=(8, 18), roll=(-6, 6), min_area=9000, min_edge=34,
        min_angle_gap=14, max_angle_gap=36, tail_compress=(0.80, 0.90), near_expand=(1.02, 1.10)
    ),
    'high': dict(
        ratio=(1.80, 2.18), height=(82, 104), yaw_side=(0.50, 0.70), pitch_far=(0.72, 0.88),
        pitch_near=(1.04, 1.18), shear=(12, 22), roll=(-7, 7), min_area=8400, min_edge=32,
        min_angle_gap=20, max_angle_gap=44, tail_compress=(0.72, 0.84), near_expand=(1.06, 1.16)
    ),
}


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], np.float32)
    return ordered.astype(np.float32)


def parse_quad(name):
    ps = Path(name).stem.split('-')
    if len(ps) >= 4 and ps[3].count('&') == 4:
        pts = []
        for pair in ps[3].split('_'):
            x, y = pair.split('&')
            pts.append([float(x), float(y)])
        return order_quad(pts)
    return None


def angle(v):
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def qstats(q):
    q = order_quad(q)
    top = q[1] - q[0]
    right = q[2] - q[1]
    bottom = q[2] - q[3]
    left = q[3] - q[0]
    edges = [float(np.linalg.norm(top)), float(np.linalg.norm(right)), float(np.linalg.norm(bottom)), float(np.linalg.norm(left))]
    area = float(abs(cv2.contourArea(q.astype(np.float32))))
    ratio = max(edges[0], edges[2]) / max(edges[1], edges[3])
    vals = [abs(angle(top)), abs(angle(bottom)), abs(abs(angle(left)) - 90), abs(abs(angle(right)) - 90)]
    return {
        'edges': edges,
        'area': area,
        'ratio': ratio,
        'min_edge': min(edges),
        'angle_score': max(vals),
        'angle_mean': sum(vals) / 4,
        'top_abs': vals[0],
        'bottom_abs': vals[1],
        'left_dev': vals[2],
        'right_dev': vals[3],
    }


def load_stats():
    with STATS.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def contains(poly, exact, margin=-1.5):
    poly = order_quad(poly)
    exact = order_quad(exact)
    return all(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) >= margin for p in exact)


def expand_cover(pseudo, exact, pad=4, max_iter=4):
    pseudo = order_quad(pseudo).copy()
    exact = order_quad(exact)
    for _ in range(max_iter):
        if contains(pseudo, exact):
            return pseudo, True
        c = pseudo.mean(axis=0)
        for p in exact:
            d = cv2.pointPolygonTest(pseudo, (float(p[0]), float(p[1])), True)
            if d < -1.5:
                idx = int(np.argmin(np.linalg.norm(pseudo - p, axis=1)))
                v = p - c
                n = np.linalg.norm(v) + 1e-6
                pseudo[idx] = p + (v / n) * pad
        pseudo = order_quad(pseudo)
    return pseudo, contains(pseudo, exact)


def pseudo_loose(q, rng, stats, tier):
    r = rng.choice(stats)
    s = qstats(q)
    w = max(s['edges'][0], s['edges'][2])
    h = max(s['edges'][1], s['edges'][3])
    amp = {'mid': (0.80, 1.12), 'high': (0.92, 1.24)}[tier]
    a = rng.uniform(*amp)
    ds = []
    for name in ['tl', 'tr', 'br', 'bl']:
        dx = np.clip(float(r[f'{name}_dx_n']) * a, -0.22, 0.22)
        dy = np.clip(float(r[f'{name}_dy_n']) * a, -0.26, 0.26)
        ds.append([dx * w, dy * h])
    pq = order_quad(q) + np.array(ds, np.float32)
    pq[:, 0] = np.clip(pq[:, 0], 0, CANVAS[0] - 1)
    pq[:, 1] = np.clip(pq[:, 1], 0, CANVAS[1] - 1)
    pq, ok = expand_cover(pq, q)
    return pq if ok else None


def build_rect_grid(cols=7, rows=2):
    xs = np.linspace(0, PLATE[0] - 1, cols + 1)
    ys = np.linspace(0, PLATE[1] - 1, rows + 1)
    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y])
    return np.asarray(pts, dtype=np.float32), cols + 1, rows + 1


def make_target_quad(rng, hdir, vdir, tier):
    p = TIER_PARAMS[tier]
    th = rng.uniform(*p['height'])
    ratio = rng.uniform(*p['ratio'])
    tw = th * ratio
    near_expand = rng.uniform(*p['near_expand'])
    far_side = rng.uniform(*p['yaw_side'])
    tail_compress = rng.uniform(*p['tail_compress'])

    if hdir == 'left':
        left_h = th * near_expand
        right_h = th * far_side
        x_top = [0.0, 0.40, 0.78, 1.0]
        x_bottom = [0.0, 0.30, 0.68, 1.0]
    else:
        left_h = th * far_side
        right_h = th * near_expand
        x_top = [0.0, 0.22, 0.60, 1.0]
        x_bottom = [0.0, 0.32, 0.70, 1.0]

    if hdir == 'left':
        x_top = [v ** (1.0 / max(0.55, tail_compress)) for v in x_top]
        x_bottom = [v ** (1.0 / max(0.55, tail_compress)) for v in x_bottom]
    else:
        x_top = [1.0 - (1.0 - v) ** (1.0 / max(0.55, tail_compress)) for v in x_top]
        x_bottom = [1.0 - (1.0 - v) ** (1.0 / max(0.55, tail_compress)) for v in x_bottom]

    if vdir == 'up':
        top_w = tw * rng.uniform(*p['pitch_far'])
        bottom_w = tw * rng.uniform(*p['pitch_near'])
        shear = -rng.uniform(*p['shear'])
    else:
        top_w = tw * rng.uniform(*p['pitch_near'])
        bottom_w = tw * rng.uniform(*p['pitch_far'])
        shear = rng.uniform(*p['shear'])

    cx = rng.uniform(185, 275)
    cy = rng.uniform(105, 195)
    tl = np.array([cx - top_w / 2, cy - left_h / 2 + shear], np.float32)
    tr = np.array([cx + top_w / 2, cy - right_h / 2 - shear], np.float32)
    br = np.array([cx + bottom_w / 2, cy + right_h / 2 - shear * 0.30], np.float32)
    bl = np.array([cx - bottom_w / 2, cy + left_h / 2 + shear * 0.30], np.float32)

    top_l = tl + (tr - tl) * x_top[1]
    top_r = tl + (tr - tl) * x_top[2]
    bot_r = bl + (br - bl) * x_bottom[2]
    bot_l = bl + (br - bl) * x_bottom[1]
    q = order_quad(np.array([top_l, top_r, bot_r, bot_l], np.float32))

    roll = math.radians(rng.uniform(*p['roll']))
    c = q.mean(axis=0)
    R = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]], np.float32)
    q = order_quad((q - c) @ R.T + c)
    return q


def warp_plate_piecewise(plate, q):
    dst = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
    mask = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
    h, w = plate.shape[:2]
    for y in range(h):
        ty = y / (h - 1) if h > 1 else 0.0
        left = q[0] * (1.0 - ty) + q[3] * ty
        right = q[1] * (1.0 - ty) + q[2] * ty
        xs = np.linspace(left[0], right[0], w)
        ys = np.linspace(left[1], right[1], w)
        src_row = plate[y]
        for x in range(w):
            ix = int(round(xs[x]))
            iy = int(round(ys[x]))
            if 0 <= ix < CANVAS[0] and 0 <= iy < CANVAS[1]:
                dst[iy, ix] = src_row[x]
                mask[iy, ix] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    return dst, mask.astype(np.float32) / 255.0


def gate(q, tier):
    p = TIER_PARAMS[tier]
    s = qstats(q)
    if not (p['ratio'][0] - 0.20 <= s['ratio'] <= p['ratio'][1] + 0.22):
        return False
    if s['area'] < p['min_area'] or s['min_edge'] < p['min_edge']:
        return False
    if s['angle_score'] < p['min_angle_gap'] or s['angle_score'] > p['max_angle_gap']:
        return False
    if q[:, 0].min() < 18 or q[:, 1].min() < 18 or q[:, 0].max() > CANVAS[0] - 18 or q[:, 1].max() > CANVAS[1] - 18:
        return False
    return True


def load_labels():
    rows = []
    with LABELS.open(encoding='utf-8') as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) >= 2 and (EXACT_DIR / ps[0]).exists() and len(ps[1]) == 8 and ps[1][0] in PROVINCES:
                rows.append((EXACT_DIR / ps[0], ps[1]))
    return rows


def choose_label(labels_by_prov, prov, rng):
    arr = labels_by_prov.get(prov) or []
    if not arr:
        arr = [x for bucket in labels_by_prov.values() for x in bucket]
    return rng.choice(arr)


def synth_text_for_prov(prov, rng):
    letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
    nums = '0123456789'
    third = rng.choice('DFABCEGHJK')
    tail = ''.join(rng.choice(nums if i >= 2 else letters + nums) for i in range(5))
    return prov + rng.choice(letters) + third + tail


def make_one(rng, stats, bgs, labels_by_prov, prov, tier, direction, split, idx):
    hdir, vdir = direction.split('_')
    attempts = 0
    while attempts < 5000:
        attempts += 1
        pth, src_text = choose_label(labels_by_prov, prov, rng)
        text = src_text if src_text[0] == prov else synth_text_for_prov(prov, rng)
        img = cv2.imread(str(pth))
        srcq = parse_quad(pth.name)
        if img is None or srcq is None:
            continue
        rect = np.float32([[0, 0], [PLATE[0] - 1, 0], [PLATE[0] - 1, PLATE[1] - 1], [0, PLATE[1] - 1]])
        plate = cv2.warpPerspective(img, cv2.getPerspectiveTransform(srcq, rect), PLATE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        q = make_target_quad(rng, hdir, vdir, tier)
        if not gate(q, tier):
            continue
        bg = cv2.resize(rng.choice(bgs), CANVAS)
        warped, mask = warp_plate_piecewise(plate, q)
        out = (warped * mask[..., None] + bg * (1.0 - mask[..., None])).astype(np.uint8)
        pq = pseudo_loose(q, rng, stats, tier)
        if pq is None or not contains(pq, q):
            continue
        s = qstats(q)
        bbox = f'{int(q[:,0].min())}&{int(q[:,1].min())}_{int(q[:,0].max())}&{int(q[:,1].max())}'
        qstr = '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in q)
        fname = f'E2persp-{bbox}-{qstr}-{split}-{tier}-{direction}-{prov}-{idx:04d}-{text}.jpg'
        return fname, out, {
            'file': fname,
            'split': split,
            'tier': tier,
            'direction': direction,
            'province': prov,
            'text': text,
            'source_exact': str(pth),
            'exact_quad': q.tolist(),
            'pseudo_quad': pq.tolist(),
            **{k: s[k] for k in ['ratio', 'area', 'min_edge', 'angle_score', 'angle_mean', 'top_abs', 'bottom_abs', 'left_dev', 'right_dev']},
            'attempts': attempts,
        }
    raise RuntimeError(f'failed make_one prov={prov} tier={tier} direction={direction} split={split}')


def allocate_equalprov(quota_per_prov, tiers):
    out = {tier: {} for tier in tiers}
    for i, prov in enumerate(sorted(quota_per_prov)):
        total = quota_per_prov[prov]
        base = total // len(tiers)
        rem = total % len(tiers)
        for j, tier in enumerate(tiers):
            out[tier][prov] = base + (1 if j < rem else 0)
        if i % 2 == 1 and len(tiers) == 2:
            out[tiers[0]][prov], out[tiers[1]][prov] = out[tiers[1]][prov], out[tiers[0]][prov]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260427)
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True)
    rng = random.Random(args.seed)
    stats = load_stats()
    bgs = [cv2.imread(str(p)) for p in sorted(BG_DIR.glob('bg_*.jpg'))]
    bgs = [b for b in bgs if b is not None]
    labels = load_labels()
    labels_by_prov = {p: [] for p in PROVINCES}
    for row in labels:
        labels_by_prov[row[1][0]].append(row)
    if not stats or not bgs or not labels:
        raise SystemExit('missing stats/backgrounds/labels')

    train_quota = {prov: 8 for prov in PROVINCES}
    proxy_quota = {prov: 3 for prov in PROVINCES}
    train_alloc = allocate_equalprov(train_quota, TIERS)
    proxy_alloc = allocate_equalprov(proxy_quota, TIERS)

    records = []
    for split, alloc in [('train', train_alloc), ('proxy', proxy_alloc)]:
        for tier in TIERS:
            per_tier_idx = 0
            direction_counts = Counter()
            for prov, cnt in sorted(alloc[tier].items()):
                for _ in range(cnt):
                    direction = DIRECTIONS[per_tier_idx % len(DIRECTIONS)]
                    per_tier_idx += 1
                    direction_counts[direction] += 1
                    fname, img, rec = make_one(rng, stats, bgs, labels_by_prov, prov, tier, direction, split, per_tier_idx)
                    sub = OUT / 'images' / split / tier
                    sub.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(sub / fname), img)
                    rec['file'] = str((sub / fname).relative_to(OUT))
                    records.append(rec)
            print(split, tier, 'count', sum(alloc[tier].values()), 'directions', dict(direction_counts))

    meta = {
        'count': len(records),
        'out': str(OUT),
        'tier_params': TIER_PARAMS,
        'train_alloc': train_alloc,
        'proxy_alloc': proxy_alloc,
        'directions': DIRECTIONS,
        'definition_note': 'Coupled yaw+pitch perspective with near-side expansion and far-tail compression; intended to mimic slanted-tightrear board-like failures.',
        'records': records,
        'summary': {
            'by_split': dict(Counter(r['split'] for r in records)),
            'by_split_tier': dict(Counter(f"{r['split']}:{r['tier']}" for r in records)),
            'by_split_direction': dict(Counter(f"{r['split']}:{r['direction']}" for r in records)),
            'by_split_prov': dict(Counter(f"{r['split']}:{r['province']}" for r in records)),
        },
    }
    (OUT / 'generation_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ['file', 'split', 'tier', 'direction', 'province', 'text', 'source_exact', 'ratio', 'area', 'min_edge', 'angle_score', 'angle_mean', 'top_abs', 'bottom_abs', 'left_dev', 'right_dev', 'attempts']
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in records])
    print(json.dumps({'generated': len(records), 'out': str(OUT), 'summary': meta['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
