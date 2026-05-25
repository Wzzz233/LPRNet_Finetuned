#!/usr/bin/env python3
"""
1) Complete geometry analysis with E1/E6A/E6B quad parsing fixed
2) Extract CCPD2020 test top-5% hard/extreme for contact sheet
"""

import csv, json, os, sys, math, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports/ccpd2020_quad_distribution_analysis'
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'src'))
from load_data import parse_ccpd_quad_from_name

# ── Metrics (same as before) ─────────────────────────────────────

def side_lengths(quad):
    p = quad.reshape(4, 2)
    sides = []
    for i in range(4):
        j = (i + 1) % 4
        sides.append(float(np.linalg.norm(p[j] - p[i])))
    return sides

def angle_score_from_quad(quad):
    p = quad.reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i - 1) % 4]
        b = p[i]
        c = p[(i + 1) % 4]
        v1 = a - b
        v2 = c - b
        dot = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            angles.append(0)
        else:
            cos_angle = dot / (n1 * n2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angles.append(abs(math.degrees(math.acos(cos_angle)) - 90))
    return float(np.mean(angles))

def skew_ratio(quad):
    p = quad.reshape(4, 2)
    top = float(np.linalg.norm(p[1] - p[0]))
    bot = float(np.linalg.norm(p[3] - p[2]))
    if bot < 1 or top < 1:
        return 1.0
    return max(top, bot) / min(top, bot)

def aspect_ratio_from_quad(quad):
    sides = side_lengths(quad)
    w = (sides[0] + sides[2]) / 2.0
    h = (sides[1] + sides[3]) / 2.0
    if h < 1:
        return 0
    return w / h

# ── Custom quad parsers ──────────────────────────────────────────

def parse_ccpd_standard(image_name):
    """Standard CCPD format: quad = parts[3] after splitting by '-'"""
    return parse_ccpd_quad_from_name(image_name)

def parse_edgefit_quad(image_name):
    """edgefit-tier3-{bbox}-{quad}... format: quad = parts[3]"""
    return parse_ccpd_quad_from_name(image_name)

def parse_e6_quad(image_name):
    """E6Aaxis/E6Baxis/E1mod-{bbox}-{quad}... format: quad = parts[2]"""
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    points_text = parts[2]
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
    return np.asarray(points, dtype=np.float32)

def detect_and_parse_quad(img_path):
    """Try standard CCPD format first, then E6/E1 format if that fails."""
    q = parse_ccpd_standard(img_path)
    if q is not None:
        return q, 'ccpd'
    q = parse_e6_quad(img_path)
    if q is not None:
        return q, 'e6'
    return None, 'unknown'

# ── Load data ────────────────────────────────────────────────────

def load_ccpd2020(split='train'):
    label_path = ROOT / f'labels/curriculum_gray3/ccpd2020_{split}.csv'
    rows = []
    with open(label_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            quad, fmt = detect_and_parse_quad(row['img_path'])
            if quad is None:
                continue
            rows.append({
                'img_path': row['img_path'],
                'text': row['text'],
                'province': row['text'][0] if row['text'] else '',
                'quad': quad,
                'quad_format': fmt,
            })
    return rows

def load_manifest_extreme(csv_path, max_samples=None):
    """Load rows from any manifest CSV that have 'extreme' source."""
    p = Path(csv_path)
    if not p.exists():
        return [], f'NOT_FOUND: {p.name}'
    rows = []
    with open(p, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get('source', '')
            if 'extreme' not in src.lower():
                continue
            quad, fmt = detect_and_parse_quad(row['img_path'])
            if quad is None:
                continue
            rows.append({
                'img_path': row['img_path'],
                'text': row['text'],
                'province': row['text'][0] if row['text'] else '',
                'quad': quad,
                'source': src,
                'tier': row.get('difficulty_tier', row.get('tier', '')),
                'quad_format': fmt,
            })
            if max_samples and len(rows) >= max_samples:
                break
    return rows, f'OK ({len(rows)} rows)'

def compute_metrics(rows, label):
    if not rows:
        return None
    angle_scores = [angle_score_from_quad(r['quad']) for r in rows]
    skews = [skew_ratio(r['quad']) for r in rows]
    ars = [aspect_ratio_from_quad(r['quad']) for r in rows]
    provinces = Counter(r['province'] for r in rows)
    formats = Counter(r.get('quad_format', '') for r in rows)
    arr = np.array(angle_scores)
    return {
        'label': label, 'n': len(rows),
        'angle_mean': float(arr.mean()), 'angle_std': float(arr.std()),
        'angle_p50': float(np.median(arr)), 'angle_p95': float(np.percentile(arr, 95)),
        'angle_max': float(arr.max()), 'angle_min': float(arr.min()),
        'skew_mean': float(np.mean(skews)), 'skew_max': float(max(skews)),
        'ar_p50': float(np.median(ars)),
        'provinces': dict(sorted(provinces.most_common(15))),
        'n_formats': dict(formats),
    }

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Full geometry comparison
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 1: COMPLETE GEOMETRY COMPARISON")
print("=" * 70)

# Load all datasets
all_data = {}

# Real CCPD2020
for split in ['train', 'test']:
    rows = load_ccpd2020(split)
    k = f'real_ccpd2020_{split}'
    all_data[k] = rows
    m = compute_metrics(rows, k)
    print(f"\nCCPD2020 {split}: {len(rows)} samples")
    if m:
        print(f"  angle: mean={m['angle_mean']:.2f}  p50={m['angle_p50']:.2f}  p95={m['angle_p95']:.2f}  max={m['angle_max']:.2f}  skew={m['skew_mean']:.3f}")

# Combined CCPD2020
all_ccpd = all_data['real_ccpd2020_train'] + all_data['real_ccpd2020_test']
all_data['real_ccpd2020_all'] = all_ccpd

# Synthetic extreme sources
extreme_manifests = {
    'old_tier3_extreme_proxy': 'manifests/curriculum_gray3_stageb_v1_difficulty/proxy_green_edgefit_extreme.csv',
    'old_tier3_extreme_train': 'labels/curriculum_gray3/green_edgefit_extreme_train.csv',
    'E1_moderate_LMH_new_proxy': 'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv',
    'E6AB_new_proxy': 'manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy/proxy_green_edgefit_extreme.csv',
    'E6A_single_axis_proxy': 'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/proxy_green_edgefit_extreme.csv',
}

for name, relpath in extreme_manifests.items():
    rows, status = load_manifest_extreme(str(ROOT / relpath), max_samples=500)
    all_data[name] = rows
    m = compute_metrics(rows, name)
    print(f"\n{name}: {status}")
    if m:
        print(f"  angle: mean={m['angle_mean']:.2f}  p50={m['angle_p50']:.2f}  p95={m['angle_p95']:.2f}  max={m['angle_max']:.2f}  skew={m['skew_mean']:.3f}")
        print(f"  formats: {m['n_formats']}")
        if m['n'] <= 20:
            print(f"  provinces: {m['provinces']}")

# Also try direct E6A/E6B dataset directories
sync_dirs = {
    'E6A_single_axis_train': '/home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260428/images/train',
    'E6B_compound_train': '/home/wzzz/LPRNet/tmp/green_extreme_stageB1A_E6B_compound_visible_20260428/images/train',
}

for name, d in sync_dirs.items():
    rows = []
    for f in sorted(Path(d).rglob('*.jpg'))[:500]:
        quad, fmt = detect_and_parse_quad(str(f))
        if quad is None:
            continue
        rows.append({
            'img_path': str(f),
            'text': str(f.stem).rsplit('-', 1)[-1] if '-' in str(f) else '',
            'province': '',
            'quad': quad,
            'source': name,
        })
    all_data[name] = rows
    m = compute_metrics(rows, name)
    print(f"\n{name}: {len(rows)} samples")
    if m:
        print(f"  angle: mean={m['angle_mean']:.2f}  p50={m['angle_p50']:.2f}  p95={m['angle_p95']:.2f}  max={m['angle_max']:.2f}  skew={m['skew_mean']:.3f}")
        print(f"  formats: {m['n_formats']}")

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
header = f"{'Dataset':<32} {'n':>6} {'angle_mean':>10} {'angle_p50':>10} {'angle_p95':>10} {'angle_max':>10} {'skew_mu':>7} {'skew_mx':>7}"
print(header)
print("-" * len(header))
for key in sorted(all_data):
    rows = all_data[key]
    m = compute_metrics(rows, key)
    if m is None:
        continue
    name = key.replace('real_', '').replace('synth_', '')
    print(f"{name:<32} {m['n']:>6} {m['angle_mean']:>10.2f} {m['angle_p50']:>10.2f} {m['angle_p95']:>10.2f} {m['angle_max']:>10.2f} {m['skew_mean']:>7.3f} {m['skew_max']:>7.3f}")

# Save metrics
metrics_all = {}
for key, rows in all_data.items():
    m = compute_metrics(rows, key)
    if m:
        metrics_all[key] = m

# ── CCPD2020 extreme percentile thresholds ──────────────────
angle_all = np.array([angle_score_from_quad(r['quad']) for r in all_ccpd])
thresholds = {}
for p in [50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]:
    thresholds[f'p{p}'] = float(np.percentile(angle_all, p))

print(f"\n\nCCPD2020 global angle_score percentiles:")
for k, v in thresholds.items():
    print(f"  {k}: {v:.2f}")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Extract CCPD2020 test hard/extreme for contact sheet
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: EXTRACTING CCPD2020 TEST EXTREME SAMPLES")
print("=" * 70)

test_rows = all_data['real_ccpd2020_test']
# Sort by angle_score descending (hardest first)
for r in test_rows:
    r['angle_score'] = angle_score_from_quad(r['quad'])

test_rows_sorted = sorted(test_rows, key=lambda r: -r['angle_score'])
top5pct = max(1, len(test_rows_sorted) // 20)  # ~250
extreme_samples = test_rows_sorted[:top5pct * 5]  # grab more for filtering

print(f"\nCCPD2020 test total: {len(test_rows)}")
print(f"Top 5% threshold: {top5pct} samples")
print(f"Range angle_score: [{extreme_samples[-1]['angle_score']:.2f}, {extreme_samples[0]['angle_score']:.2f}]")

# Filter: keep samples that actually have visible plates (Laplacian > 30)
valid = []
for r in extreme_samples[:200]:
    img = cv2.imread(r['img_path'], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    if lap > 30:  # not too blurry
        valid.append(r)
    if len(valid) >= 100:
        break

print(f"After blur filter (lap>30): {len(valid)} valid samples")
print(f"Province distribution:")
prov_counts = Counter(r['province'] for r in valid)
for p, c in prov_counts.most_common(20):
    print(f"  {p}: {c}")

# ── Generate contact sheet ──────────────────────────────────

def generate_contact_sheet(samples, save_path, label_fn=None, cols=6, max_rows=20):
    """Generate a grid contact sheet of plate images with overlays and labels."""
    from PIL import ImageDraw, ImageFont
    
    n = min(len(samples), cols * max_rows)
    rows_to_show = (n + cols - 1) // cols
    cell_w, cell_h = 300, 160
    
    canvas = Image.new('RGB', (cols * cell_w, rows_to_show * cell_h + 40), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 14)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
        except:
            pass
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 28)
    except:
        title_font = ImageFont.load_default()
    
    # Title
    draw.text((10, 5), f"CCPD2020 test extreme samples (angle_score >= {extreme_samples[n-1]['angle_score']:.1f})", 
              fill=(200, 200, 200), font=title_font)
    
    for idx in range(n):
        r = samples[idx]
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h + 40
        
        try:
            # Load and resize to cell
            img_pil = Image.open(r['img_path']).convert('RGB')
            img_draw = ImageDraw.Draw(img_pil)
            
            # Draw original quad on image
            quad = r['quad'].reshape(4, 2)
            pts = [(float(p[0]), float(p[1])) for p in quad]
            img_draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
            
            # Resize to fit cell
            img_pil.thumbnail((cell_w - 10, cell_h - 50), Image.LANCZOS)
            
            # Paste
            cx = x0 + (cell_w - img_pil.width) // 2
            canvas.paste(img_pil, (cx, y0 + 5))
            
            # Label
            angle = r.get('angle_score', angle_score_from_quad(r['quad']))
            label = f"{r['text']} | angle={angle:.1f}"
            draw.text((x0 + 5, y0 + cell_h - 22), label, fill=(180, 200, 255), font=font)
        except Exception as e:
            draw.text((x0 + 5, y0 + 20), f"ERR: {e}", fill=(255, 100, 100), font=font)
    
    canvas.save(save_path, quality=95)
    return save_path

# Generate 3 contact sheets: all extreme, angle 15-25, angle 25+
cs_dir = OUT / 'contact_sheets'
cs_dir.mkdir(parents=True, exist_ok=True)

# Sheet 1: All 100 valid extreme samples
cs_path = cs_dir / 'ccpd2020_test_extreme_contact_sheet.jpg'
print(f"\nGenerating contact sheet (n={len(valid)})...")
generate_contact_sheet(valid, str(cs_path), cols=5, max_rows=20)
print(f"  Saved: {cs_path}")

# Sheet 2: Samples with angle between 15 and 25 (hard/moderate)
mid_range = [r for r in valid if 15 <= r['angle_score'] <= 25]
cs_path_mid = cs_dir / 'ccpd2020_test_moderate_hard_angle15_25.jpg'
if mid_range:
    generate_contact_sheet(mid_range[:50], str(cs_path_mid), cols=5, max_rows=10)
    print(f"  Saved (angle 15-25, n={len(mid_range)}): {cs_path_mid}")

# Sheet 3: Samples with angle > 25 (truly extreme)
high_range = [r for r in valid if r['angle_score'] > 25]
cs_path_high = cs_dir / 'ccpd2020_test_truly_extreme_angle25plus.jpg'
if high_range:
    generate_contact_sheet(high_range[:50], str(cs_path_high), cols=5, max_rows=10)
    print(f"  Saved (angle>25, n={len(high_range)}): {cs_path_high}")

# Also copy to Windows QA
win_qa = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_extreme_contact_sheets')
win_qa.mkdir(parents=True, exist_ok=True)
for src in cs_dir.glob('*.jpg'):
    dst = win_qa / src.name
    import shutil
    shutil.copy2(str(src), str(dst))
    print(f"  Copied to Windows: {dst}")

# ── Final summary ────────────────────────────────────────────

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nComplete results saved to {OUT / 'distribution_analysis.json'}")
print(f"Contact sheets in: {cs_dir}")
print(f"Windows QA: {win_qa}")
print("\nDONE")
