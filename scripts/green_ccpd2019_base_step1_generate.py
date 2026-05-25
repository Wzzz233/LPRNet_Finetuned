#!/usr/bin/env python3
"""Generate green cvr replacement for CCPD2019 base."""
import csv, json, os, sys, math, random, time, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'utils'))

from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--train_per_province', type=int, default=2000)
parser.add_argument('--val_per_province', type=int, default=20)
parser.add_argument('--skip_generate', action='store_true')
args = parser.parse_args()

if args.smoke:
    TRAIN_PER_PROV = 5
    VAL_PER_PROV = 5
else:
    TRAIN_PER_PROV = args.train_per_province
    VAL_PER_PROV = args.val_per_province

POSE_JSONL = ROOT / 'datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl'
OUT_DIR = ROOT / 'datasets' / 'green_ccpd2019_base_cvreplace_posquad_v1_20260509'
MANIFEST_DIR = ROOT / 'manifests_rebased' / 'green_ccpd2019_base_cvreplace_posquad_v1_20260509'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_ccpd2019_base_cvr')

for d in [OUT_DIR, OUT_DIR/'images'/'train', OUT_DIR/'images'/'val', MANIFEST_DIR, WIN_QA]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(20260509)
ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

# ── Init plate generation ──────────────────────────────────────
print("Init generation pipeline...", flush=True)
_chars_gen = CharsImageGenerator('small_new_energy')
_tg = LicensePlateImageGenerator('small_new_energy')
_tmpl = _tg.generate_template_image(_chars_gen.plate_width, _chars_gen.plate_height)
_aug = ImageAugmentation('small_new_energy', _tmpl)
_aug.env_data_paths = [str(ROOT / p) for p in _aug.env_data_paths]
_smu = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(_smu):
    _aug.smu = cv2.imread(_smu)

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])
LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
DIGITS = list('0123456789')
used_texts = set()

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def make_new_text(province):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + ''.join(random.choice(DIGITS) for _ in range(5))
        if t not in used_texts:
            used_texts.add(t)
            return t

# ── CV transfer (L-only) ────────────────────────────────────────
def estimate_sharpness(bgr):
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def estimate_noise(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = cv2.boxFilter(gray, -1, (7,7), normalize=False)
    local_mean = cv2.boxFilter(gray, -1, (7,7), normalize=True)
    local_var = local_std - local_mean * local_mean * 49
    local_var = np.clip(local_var, 0, None)
    return float(np.median(np.sqrt(local_var)))

def transfer_cv_properties_lonly(plate_bgr, orig_patch_bgr):
    h, w = plate_bgr.shape[:2]
    oh, ow = orig_patch_bgr.shape[:2]
    orig_resized = cv2.resize(orig_patch_bgr, (w, h), interpolation=cv2.INTER_AREA) if (oh, ow) != (h, w) else orig_patch_bgr.copy()
    plate_lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lp, Lo = plate_lab[:,:,0], orig_lab[:,:,0]
    mn_p, std_p = Lp.mean(), Lp.std()+1e-6
    mn_o, std_o = Lo.mean(), Lo.std()+1e-6
    ratio = max(0.3, min(3.0, std_o/std_p))
    plate_lab[:,:,0] = np.clip((Lp-mn_p)*ratio+mn_o, 0, 255)
    result = cv2.cvtColor(plate_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    orig_hsv = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    ill_orig = cv2.resize(cv2.resize(orig_hsv[:,:,2], (8,4), interpolation=cv2.INTER_AREA), (w,h), interpolation=cv2.INTER_LINEAR)
    ill_res = cv2.resize(cv2.resize(result_hsv[:,:,2], (8,4), interpolation=cv2.INTER_AREA), (w,h), interpolation=cv2.INTER_LINEAR)
    result_hsv[:,:,2] = np.clip(result_hsv[:,:,2]+(ill_orig-ill_res)*0.5, 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    sp, so = estimate_sharpness(result), estimate_sharpness(orig_resized)
    if sp > so*1.5:
        ksize = 3
        while ksize <= 15 and estimate_sharpness(result) > so*1.3:
            result = cv2.GaussianBlur(result, (ksize,ksize), 0); ksize += 2
    np_, np2 = estimate_noise(orig_resized), estimate_noise(result)
    if np_ > np2*1.2:
        result = np.clip(result.astype(np.float32)+np.random.randn(h,w,3).astype(np.float32)*min(np_-np2,30), 0,255).astype(np.uint8)
    return result

def enforce_green_plate_color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)
    green = ((s>30)&(v>40)&(h>=45)&(h<=100)).sum()
    blue = ((s>30)&(v>40)&(h>=100)&(h<=140)).sum()
    total = bgr.shape[0]*bgr.shape[1]
    return green/total, blue/total

# ── Load pose data ──────────────────────────────────────────────
print("Loading pose data...", flush=True)
all_pose = [json.loads(l) for l in open(POSE_JSONL)]
print(f"  Total pose entries: {len(all_pose)}", flush=True)

random.shuffle(all_pose)
val_entries = all_pose[:2000]
train_entries = all_pose[2000:]
print(f"  Train pool: {len(train_entries)}, Val pool: {len(val_entries)}", flush=True)

# ── Generator ────────────────────────────────────────────────────
MANIFEST_FIELDS = [
    'img_path','text','family','source','split','preprocess_group',
    'has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source',
    'quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y',
    'ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc',
    'ocr_channel_order','ocr_quad_pad_ratio',
]

def generate_split(pool, per_prov, split_name):
    n_needed = len(ALL_PROVINCES) * per_prov
    random.shuffle(pool)
    schedule = []
    for cycle in range(math.ceil(n_needed / max(len(pool),1))):
        for src in pool:
            schedule.append(src)
    schedule = schedule[:n_needed]

    prov_assign = []
    for i in range(n_needed):
        prov_assign.append(ALL_PROVINCES[i % len(ALL_PROVINCES)])

    results = []
    used_stems = set()
    color_skipped = 0

    for idx, (src_rec, prov) in enumerate(zip(schedule, prov_assign)):
        img_path = src_rec['img_path']
        gt_quad_raw = np.array(src_rec['gt_quad'])
        gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
        pose_quad = np.array(src_rec['pose_quad'])
        new_text = make_new_text(prov)

        try:
            img = cv2.imread(img_path)
            if img is None: raise ValueError(f"Cannot read {img_path}")
            h, w = img.shape[:2]
            M_fwd = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), SRC_RECT)
            plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
            raw_plate = build_plate(new_text)
            processed = transfer_cv_properties_lonly(raw_plate, plate_orig)
            gr, br = enforce_green_plate_color(processed)
            if br >= 0.10:
                color_skipped += 1; continue
            M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad.astype(np.float32))
            warped = cv2.warpPerspective(processed, M_inv, (w, h), flags=cv2.INTER_LINEAR)
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
            mask = cv2.GaussianBlur(mask, (3,3), 0).astype(np.float32)/255.0
            result = (img.astype(np.float32)*(1-mask[:,:,None]) + warped.astype(np.float32)*mask[:,:,None]).clip(0,255).astype(np.uint8)
            stem = Path(img_path).stem
            fname = f"{stem}_green_{new_text}.jpg"
            out_path = (OUT_DIR/'images'/split_name)/fname
            if not args.skip_generate:
                cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            used_stems.add(stem)
            row = {'img_path': str(out_path.relative_to(ROOT)), 'text': new_text, 'family': 'green8',
                'source': 'green_ccpd2019_base_cvreplace_posquad_v1', 'split': split_name,
                'preprocess_group': 'ccpd_board', 'has_quad': '1', 'can_parse_ccpd_geom': '0', 'can_perspective': '1',
                'quad_source': 'pose_yolov8n_ccpd2019_base', 'bbox_source': 'pose_yolov8n_ccpd2019_base',
                'quad_1x': f'{pose_quad[0][0]:.1f}', 'quad_1y': f'{pose_quad[0][1]:.1f}',
                'quad_2x': f'{pose_quad[1][0]:.1f}', 'quad_2y': f'{pose_quad[1][1]:.1f}',
                'quad_3x': f'{pose_quad[2][0]:.1f}', 'quad_3y': f'{pose_quad[2][1]:.1f}',
                'quad_4x': f'{pose_quad[3][0]:.1f}', 'quad_4y': f'{pose_quad[3][1]:.1f}',
                'ocr_crop_mode': 'obb_warp', 'ocr_resize_mode': 'letterbox', 'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'none', 'ocr_channel_order': 'bgr', 'ocr_quad_pad_ratio': '0.0'}
            results.append(row)
        except Exception as e:
            pass
    return results, used_stems, color_skipped

t0 = time.time()
train_rows, train_stems, train_skip = generate_split(train_entries, TRAIN_PER_PROV, 'train')
val_rows, val_stems, val_skip = generate_split(val_entries, VAL_PER_PROV, 'val')
print(f"\nTrain: {len(train_rows)} rows, {len(train_stems)} unique, {train_skip} skipped", flush=True)
print(f"Val:   {len(val_rows)} rows, {len(val_stems)} unique, {val_skip} skipped", flush=True)
print(f"Overlap: {len(train_stems & val_stems)}", flush=True)

prov_cnt = Counter(r['text'][0] for r in train_rows+val_rows if r['text'])
print(f"Province distribution:")
for p in sorted(prov_cnt.keys(), key=lambda x: -prov_cnt[x]):
    print(f"  {p}: {prov_cnt[p]}", flush=True)

for rows, fn in [(train_rows, 'train_base_cvreplace_posquad.csv'), (val_rows, 'val_base_cvreplace_posquad.csv')]:
    out = MANIFEST_DIR / fn
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS); w.writeheader(); w.writerows(rows)
    print(f"  {out}: {len(rows)} rows", flush=True)

meta = {'mode': 'smoke' if args.smoke else 'full', 'n_train': len(train_rows), 'n_val': len(val_rows),
        'unique_sources': len(train_stems | val_stems), 'color_skipped': train_skip+val_skip,
        'province_distribution': dict(prov_cnt), 'pose_rate': '199996/199996 (100%)'}
json.dump(meta, open(OUT_DIR/'generation_meta.json','w'), ensure_ascii=False, indent=2)
print(f"Elapsed: {(time.time()-t0)/60:.1f}min", flush=True)
print("Done.", flush=True)
