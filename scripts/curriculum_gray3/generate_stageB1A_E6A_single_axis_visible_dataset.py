#!/usr/bin/env python3
"""Generate StageB1A-E6A single-axis visible perspective dataset.

Only pure left/right/up/down directions are used.
Low and mid are intentionally more visible than the previous axis-dominant E6.
"""
import argparse, csv, json, math, random, shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/wzzz/LPRNet')
EXACT_DIR = ROOT / 'datasets/green_exact_quad_synthetic_v1'
LABELS = EXACT_DIR / 'manifests/train_synthetic_labels.txt'
BG_DIR = ROOT / 'tmp/green_extreme_pathB_probe_v2_stronger_20260425/backgrounds'
STATS = ROOT / 'reports/stageB_extreme_detector_vs_gt_stats_20260426/green_det_vs_gt_rows_valid.csv'
OUT = ROOT / 'tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428'
PLATE_MIN = (470, 120)
PLATE_MAX_W = 640
PLATE_MAX_H = 192
PNG_COMPRESSION = 3
CANVAS = (920, 600)
PRE_NN_GAUSS_SIGMA = 0.42
TIERS = ['low', 'mid', 'high']
DIRECTIONS = ['left', 'right', 'up', 'down']
PROVINCES = list('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
TIER_PARAMS = {
    'low': dict(ratio=(2.04, 2.42), height=(84, 102), yaw_side=(0.84, 0.94), pitch=(0.86, 1.02), shear=(2.0, 4.5), roll=(-2.5, 2.5), min_area=11600, min_edge=46, min_angle_gap=4.0, max_angle_gap=14),
    'mid': dict(ratio=(2.00, 2.40), height=(84, 102), yaw_side=(0.80, 0.90), pitch=(0.84, 1.04), shear=(3.0, 6.5), roll=(-3.0, 3.0), min_area=10800, min_edge=42, min_angle_gap=6.5, max_angle_gap=18),
    'high': dict(ratio=(1.94, 2.30), height=(84, 100), yaw_side=(0.64, 0.80), pitch=(0.68, 1.12), shear=(6.0, 10.5), roll=(-4.0, 4.0), min_area=10400, min_edge=41, min_angle_gap=11.0, max_angle_gap=28),
}
DIRECTION_GROUPS = {
    'left': ('left', 'mid'),
    'right': ('right', 'mid'),
    'up': ('mid', 'up'),
    'down': ('mid', 'down'),
}
PSEUDO_AMP = {
    'low': (0.48, 0.82),
    'mid': (0.60, 0.94),
    'high': (0.72, 1.02),
}
YAW_MIX = {'low': 0.28, 'mid': 0.40, 'high': 0.62}
PITCH_MIX = {'low': 0.36, 'mid': 0.56, 'high': 0.78}


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
    return {'edges': edges, 'area': area, 'ratio': ratio, 'min_edge': min(edges), 'angle_score': max(vals), 'angle_mean': sum(vals) / 4, 'top_abs': vals[0], 'bottom_abs': vals[1], 'left_dev': vals[2], 'right_dev': vals[3]}


def valid_quad(q, min_edge, min_area):
    s = qstats(q)
    if s['min_edge'] < min_edge or s['area'] < min_area:
        return False
    q = order_quad(q)
    return all(np.linalg.norm(q[i] - q[j]) >= min_edge for i in range(4) for j in range(i + 1, 4))


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
    a = rng.uniform(*PSEUDO_AMP[tier])
    ds = []
    for name in ['tl', 'tr', 'br', 'bl']:
        dx = np.clip(float(r[f'{name}_dx_n']) * a, -0.16, 0.16)
        dy = np.clip(float(r[f'{name}_dy_n']) * a, -0.18, 0.18)
        ds.append([dx * w, dy * h])
    pq = order_quad(q) + np.array(ds, np.float32)
    pq[:, 0] = np.clip(pq[:, 0], 0, CANVAS[0] - 1)
    pq[:, 1] = np.clip(pq[:, 1], 0, CANVAS[1] - 1)
    pq, ok = expand_cover(pq, q)
    return pq if ok else None


def target_quad(rng, direction, tier, relax=1.0):
    p = TIER_PARAMS[tier]
    hdir, vdir = DIRECTION_GROUPS[direction]
    th = rng.uniform(*p['height'])
    ratio = rng.uniform(*p['ratio'])
    tw = th * ratio

    yaw_far = rng.uniform(*p['yaw_side'])
    yaw_near = rng.uniform(1.00, 1.08)
    yaw_mix = YAW_MIX[tier] * relax
    pitch_low, pitch_high = p['pitch']
    pitch_mix = PITCH_MIX[tier] * relax

    left_h = th * rng.uniform(0.97, 1.03)
    right_h = th * rng.uniform(0.97, 1.03)
    top_w = tw * rng.uniform(0.97, 1.02)
    bottom_w = tw * rng.uniform(0.97, 1.02)

    if hdir == 'left':
        left_h = th * (1.0 + (yaw_near - 1.0) * yaw_mix)
        right_h = th * (1.0 - (1.0 - yaw_far) * yaw_mix)
    elif hdir == 'right':
        left_h = th * (1.0 - (1.0 - yaw_far) * yaw_mix)
        right_h = th * (1.0 + (yaw_near - 1.0) * yaw_mix)

    if vdir == 'up':
        top_target = rng.uniform(pitch_low, 0.94)
        bottom_target = rng.uniform(1.00, pitch_high)
        top_w = tw * (1.0 - (1.0 - top_target) * pitch_mix)
        bottom_w = tw * (1.0 + (bottom_target - 1.0) * pitch_mix)
    elif vdir == 'down':
        top_target = rng.uniform(1.00, pitch_high)
        bottom_target = rng.uniform(pitch_low, 0.94)
        top_w = tw * (1.0 + (top_target - 1.0) * pitch_mix)
        bottom_w = tw * (1.0 - (1.0 - bottom_target) * pitch_mix)

    shear_mag = rng.uniform(*p['shear']) * relax
    if direction in ('left', 'right'):
        shear = rng.choice([-1.0, 1.0]) * shear_mag
    else:
        shear = -shear_mag if direction == 'up' else shear_mag

    cx = rng.uniform(370, 550)
    cy = rng.uniform(210, 390)
    tl = np.array([cx - top_w / 2, cy - left_h / 2 + shear], np.float32)
    tr = np.array([cx + top_w / 2, cy - right_h / 2 - shear], np.float32)
    br = np.array([cx + bottom_w / 2, cy + right_h / 2 - shear * 0.28], np.float32)
    bl = np.array([cx - bottom_w / 2, cy + left_h / 2 + shear * 0.28], np.float32)
    q = order_quad(np.array([tl, tr, br, bl], np.float32))
    roll = math.radians(rng.uniform(*p['roll']))
    c = q.mean(axis=0)
    R = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]], np.float32)
    return order_quad((q - c) @ R.T + c)


def gate(q, tier):
    p = TIER_PARAMS[tier]
    s = qstats(q)
    if not (p['ratio'][0] - 0.12 <= s['ratio'] <= p['ratio'][1] + 0.14):
        return False
    if s['area'] < p['min_area'] or s['min_edge'] < p['min_edge']:
        return False
    if s['angle_score'] < p['min_angle_gap'] or s['angle_score'] > p['max_angle_gap']:
        return False
    if q[:, 0].min() < 22 or q[:, 1].min() < 22 or q[:, 0].max() > CANVAS[0] - 22 or q[:, 1].max() > CANVAS[1] - 22:
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
    tail = ''.join(rng.choice(nums if i % 2 else letters + nums) for i in range(5))
    return prov + rng.choice(letters) + third + tail


def intermediate_plate_size(srcq):
    src_w = max(np.linalg.norm(srcq[1] - srcq[0]), np.linalg.norm(srcq[2] - srcq[3]))
    src_h = max(np.linalg.norm(srcq[3] - srcq[0]), np.linalg.norm(srcq[2] - srcq[1]))
    scale = min(1.0, PLATE_MAX_W / (src_w + 1e-6), PLATE_MAX_H / (src_h + 1e-6))
    plate_w = int(round(src_w * scale))
    plate_h = int(round(src_h * scale))
    plate_w = max(PLATE_MIN[0], min(PLATE_MAX_W, plate_w))
    plate_h = max(PLATE_MIN[1], min(PLATE_MAX_H, plate_h))
    return plate_w, plate_h


def enhance_plate_for_small_ocr(plate):
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 4))
    l_chan = clahe.apply(l_chan)
    blur = cv2.GaussianBlur(l_chan, (0, 0), 0.75)
    l_chan = cv2.addWeighted(l_chan, 1.28, blur, -0.28, 0)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def render_plate_to_canvas(img, srcq, dstq):
    plate_w, plate_h = intermediate_plate_size(srcq)
    rect = np.float32(
        [[0, 0], [plate_w - 1, 0], [plate_w - 1, plate_h - 1], [0, plate_h - 1]]
    )
    plate = cv2.warpPerspective(
        img,
        cv2.getPerspectiveTransform(srcq, rect),
        (plate_w, plate_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    plate = enhance_plate_for_small_ocr(plate)
    # Simulate a mild optical PSF before the later board-side 94x24 NN resize.
    # This smooths subpixel staircase artifacts without changing geometry.
    if PRE_NN_GAUSS_SIGMA > 0:
        plate = cv2.GaussianBlur(
            plate,
            (0, 0),
            sigmaX=PRE_NN_GAUSS_SIGMA,
            sigmaY=PRE_NN_GAUSS_SIGMA,
            borderType=cv2.BORDER_REPLICATE,
        )
    return cv2.warpPerspective(
        plate,
        cv2.getPerspectiveTransform(rect, dstq),
        CANVAS,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_TRANSPARENT,
    )


def make_one(rng, stats, bgs, labels_by_prov, prov, tier, direction, split, idx):
    attempts = 0
    while attempts < 5000:
        attempts += 1
        pth, src_text = choose_label(labels_by_prov, prov, rng)
        text = src_text if src_text[0] == prov else synth_text_for_prov(prov, rng)
        img = cv2.imread(str(pth))
        srcq = parse_quad(pth.name)
        if img is None or srcq is None:
            continue
        relax = 1.0
        if attempts > 1500:
            relax = 0.90
        if attempts > 3000:
            relax = 0.80
        q = target_quad(rng, direction, tier, relax=relax)
        if not gate(q, tier):
            continue
        bg = cv2.resize(rng.choice(bgs), CANVAS)
        # Keep the new E6A geometry, but render through a plate-sized intermediate.
        # This preserves edge detail better than pre-blurring the full source and
        # collapsing it directly into the target quad in one step.
        warped = render_plate_to_canvas(img, srcq, q)
        mask = np.zeros(CANVAS[::-1], np.float32)
        cv2.fillPoly(mask, [q.astype(np.int32)], 1.0)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.35)[..., None]
        out = (warped * mask + bg * (1 - mask)).astype(np.uint8)
        pq = pseudo_loose(q, rng, stats, tier)
        if pq is None or not valid_quad(pq, 12.0, 3500.0) or not contains(pq, q):
            continue
        s = qstats(q)
        bbox = f'{int(q[:,0].min())}&{int(q[:,1].min())}_{int(q[:,0].max())}&{int(q[:,1].max())}'
        qstr = '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in q)
        fname = f'E6Aaxis-{bbox}-{qstr}-{split}-{tier}-{direction}-{prov}-{idx:04d}-{text}.png'
        return fname, out, {'file': fname, 'split': split, 'tier': tier, 'direction': direction, 'province': prov, 'text': text, 'source_exact': str(pth), 'exact_quad': q.tolist(), 'pseudo_quad': pq.tolist(), **{k: s[k] for k in ['ratio', 'area', 'min_edge', 'angle_score', 'angle_mean', 'top_abs', 'bottom_abs', 'left_dev', 'right_dev']}, 'attempts': attempts}
    raise RuntimeError(f'failed make_one prov={prov} tier={tier} direction={direction} split={split}')


def allocate_train_quota(total_by_prov, tier_targets):
    provs = sorted(total_by_prov)
    weights = {p: total_by_prov[p] for p in provs}
    total = sum(weights.values())
    tier_alloc = {t: {} for t in tier_targets}
    for t, target in tier_targets.items():
        raw = [(p, weights[p] * target / total) for p in provs]
        base = {p: int(math.floor(v)) for p, v in raw}
        rem = target - sum(base.values())
        order = sorted(raw, key=lambda x: (x[1] - math.floor(x[1]), x[0]), reverse=True)
        for p, _ in order[:rem]:
            base[p] += 1
        tier_alloc[t] = base
    for _ in range(10000):
        diff = {p: weights[p] - sum(tier_alloc[t][p] for t in tier_targets) for p in provs}
        if all(v == 0 for v in diff.values()):
            break
        p_plus = next((p for p, v in diff.items() if v > 0), None)
        p_minus = next((p for p, v in diff.items() if v < 0), None)
        if p_plus is None or p_minus is None:
            break
        for t in ['low', 'mid', 'high']:
            if tier_alloc[t][p_minus] > 0:
                tier_alloc[t][p_minus] -= 1
                tier_alloc[t][p_plus] += 1
                break
    assert {t: sum(v.values()) for t, v in tier_alloc.items()} == tier_targets
    assert {p: sum(tier_alloc[t][p] for t in tier_targets) for p in provs} == weights
    return tier_alloc


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

    train_quota = {'云': 10, '京': 10, '冀': 10, '吉': 10, '宁': 10, '川': 10, '新': 10, '晋': 10, '桂': 10, '沪': 11, '津': 10, '浙': 10, '渝': 10, '湘': 10, '琼': 10, '甘': 10, '皖': 8, '粤': 10, '苏': 10, '蒙': 10, '藏': 10, '豫': 9, '贵': 9, '赣': 9, '辽': 9, '鄂': 9, '闽': 9, '陕': 10, '青': 9, '鲁': 9, '黑': 9}
    train_tier_targets = {'low': 100, 'mid': 120, 'high': 80}
    train_alloc = allocate_train_quota(train_quota, train_tier_targets)
    proxy_quota = {p: 4 for p in train_quota}
    proxy_tier_targets = {'low': 36, 'mid': 48, 'high': 40}
    proxy_alloc = allocate_train_quota(proxy_quota, proxy_tier_targets)

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
                    cv2.imwrite(str(sub / fname), img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
                    rec['file'] = str((sub / fname).relative_to(OUT))
                    records.append(rec)
            print(split, tier, 'count', sum(alloc[tier].values()), 'directions', dict(direction_counts))

    meta = {'count': len(records), 'out': str(OUT), 'tier_params': TIER_PARAMS, 'train_alloc': train_alloc, 'proxy_alloc': proxy_alloc, 'records': records,
            'summary': {'by_split': dict(Counter(r['split'] for r in records)), 'by_split_tier': dict(Counter(f"{r['split']}:{r['tier']}" for r in records)), 'by_split_prov': dict(Counter(f"{r['split']}:{r['province']}" for r in records)), 'by_split_direction': dict(Counter(f"{r['split']}:{r['direction']}" for r in records))}}
    (OUT / 'generation_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ['file', 'split', 'tier', 'direction', 'province', 'text', 'source_exact', 'ratio', 'area', 'min_edge', 'angle_score', 'angle_mean', 'top_abs', 'bottom_abs', 'left_dev', 'right_dev', 'attempts']
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in records])
    print(json.dumps({'generated': len(records), 'out': str(OUT), 'summary': meta['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
