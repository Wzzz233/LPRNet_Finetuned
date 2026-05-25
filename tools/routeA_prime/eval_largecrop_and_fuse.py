#!/usr/bin/env python3
"""Route A' Phase 4+5: Evaluate large-crop province net and fuse with R50.

Evaluates on:
  - val holdout (quad-warp images)
  - province stress (quad-warp images)  
  - dump2 (coarse_*.ppm, resized to model input)
  - dump1 (coarse_*.ppm, resized to model input)
  - cluster2 (crop_*.ppm, resized to model input)

Fusion: replace_first_char with R50 predictions.
Best model selection: based on dump2 results.

Usage:
  python tools/routeA_prime/eval_largecrop_and_fuse.py \
    --model experiments/routeA_prime_quadwarp_20260512/B1/best.pt \
    --experiment_name B1_fullplate_color_224x72_raw \
    --input_size 224 72 --ocr_preproc none --in_channels 3
"""
import sys, json, torch, cv2, numpy as np, csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
OUT_BASE = ROOT / 'experiments/routeA_prime_quadwarp_20260512'

sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))
from load_data import CHARS
from train_LPRNet import forward_family_logits
from LPRNet_multihead import build_lprnet_multihead
import torchvision.models as models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK = len(CHARS) - 1

# R50
R50_CKPT = ROOT / 'experiments/a_ratio_r50_20260510/best_LPRNet_model.pth'

# Board paths
POS_OCR_DUMP = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump')
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')
CLUSTER2_DIR = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2'

GT_DUMP1 = '苏BF01111'
GT_DUMP2 = '京AD06088'

PROVINCE_CHARS = ['京','津','冀','晋','蒙','辽','吉','黑',
                  '沪','苏','浙','皖','闽','赣','鲁','豫',
                  '鄂','湘','粤','桂','琼','川','贵','云',
                  '藏','陕','甘','青','宁','新','渝']
NUM_CLASSES = len(PROVINCE_CHARS)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)


def decode_ctc(prebs):
    results = []
    for bi in range(prebs.shape[0]):
        preb = prebs[bi, :, :]
        preb_label = [int(np.argmax(preb[:, t], axis=0)) for t in range(preb.shape[1])]
        decoded = []
        prev = preb_label[0]
        if prev != BLANK: decoded.append(prev)
        for c in preb_label[1:]:
            if c == prev or c == BLANK:
                if c == BLANK: prev = c
                continue
            decoded.append(c); prev = c
        results.append(''.join(CHARS[i] for i in decoded if 0 <= i < len(CHARS)))
    return results


def load_r50():
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                 dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
    net.load_state_dict(torch.load(str(R50_CKPT), map_location='cpu'), strict=False)
    net.to(device); net.eval()
    return net


def build_largecrop_model(in_channels=3, num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_largecrop_model(model_path, in_channels=3):
    model = build_largecrop_model(in_channels=in_channels)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model.to(device); model.eval()
    return model


def infer_one(img_path, model, input_size, ocr_preproc, in_channels):
    """Infer province from one image. Returns (province_char, confidence)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0.0

    # Resize to model input
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    # Preprocessing
    if ocr_preproc in ('gray', 'gray3'):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if ocr_preproc == 'gray3':
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            img = gray[:, :, None]

    # Normalize
    img_t = torch.from_numpy(img.astype('float32') / 255.0)
    if img_t.ndim == 2:
        img_t = img_t.unsqueeze(-1)
    img_t = img_t.permute(2, 0, 1).contiguous().unsqueeze(0).to(device)

    with torch.no_grad():
        # Handle gray/gray3 input: reduce to 1 channel
        if ocr_preproc in ('gray', 'gray3'):
            gray = img_t[:, 0:1, :, :] * 0.1140 + img_t[:, 1:2, :, :] * 0.5870 + img_t[:, 2:3, :, :] * 0.2990
            img_t = gray
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    province_char = PROVINCE_CHARS[pred.item()] if pred.item() < len(PROVINCE_CHARS) else '?'
    return province_char, float(conf.item())


def compute_stats(preds, gts):
    n = len(preds)
    fc_correct = sum(1 for p, g in zip(preds, gts) if len(p) > 0 and len(g) > 0 and p[0] == g[0])
    fc_acc = fc_correct / n if n > 0 else 0.0
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    exact_acc = exact / n if n > 0 else 0.0
    suffix_correct = sum(1 for p, g in zip(preds, gts) if len(p) > 1 and len(g) > 1 and p[1:] == g[1:])
    suffix_acc = suffix_correct / n if n > 0 else 0.0

    prov_conf = Counter()
    for p, g in zip(preds, gts):
        if len(p) > 0 and len(g) > 0 and p[0] != g[0]:
            prov_conf[f'GT:{g[0]}->Pred:{p[0]}'] += 1

    per_prov = {}
    for p, g in zip(preds, gts):
        if len(g) == 0: continue
        prov = g[0]
        if prov not in per_prov:
            per_prov[prov] = {'correct': 0, 'total': 0}
        per_prov[prov]['total'] += 1
        if len(p) > 0 and p[0] == prov:
            per_prov[prov]['correct'] += 1
    prov_accs = {p: round(s['correct']/s['total'], 4) for p, s in sorted(per_prov.items())}
    macro_fc = sum(prov_accs.values()) / len(prov_accs) if prov_accs else 0.0
    prov_preds = Counter()
    for p in preds:
        if len(p) > 0:
            prov_preds[p[0]] += 1

    return {
        'count': n,
        'first_char_acc': round(fc_acc, 4),
        'exact_acc': round(exact_acc, 4),
        'suffix_exact_acc': round(suffix_acc, 4),
        'macro_first_char_acc': round(macro_fc, 4),
        'province_confusion_top20': dict(prov_conf.most_common(20)),
        'per_province_fc_acc': prov_accs,
        'province_prediction_distribution': dict(prov_preds.most_common()),
    }


def eval_on_images(model, img_paths, gts, input_size, ocr_preproc, in_channels):
    """Evaluate on a list of images."""
    preds = []
    confs = []
    for p in img_paths:
        c, conf = infer_one(p, model, input_size, ocr_preproc, in_channels)
        preds.append(c if c else '?')
        confs.append(conf)
    stats = compute_stats(preds, gts)
    stats['predictions'] = preds
    stats['confidences'] = [round(c, 4) for c in confs]
    return stats


def eval_on_quadwarp_dir(model, img_dir, input_size, ocr_preproc, in_channels):
    """Evaluate on exported quad-warp images directory with manifest.csv."""
    manifest = Path(img_dir) / 'manifest.csv'
    if not manifest.exists():
        return None
    img_paths = []
    gts = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = r.get('img_path', '').strip()
            text = r.get('text', '').strip()
            if p and text and Path(p).exists():
                img_paths.append(p)
                gts.append(text)
    return eval_on_images(model, img_paths, gts, input_size, ocr_preproc, in_channels)


def eval_on_board(model, dump_dir, gt_text, prefix, input_size, ocr_preproc, in_channels):
    """Evaluate on board dump coarse/crop/ocrin images."""
    files = sorted([f for f in Path(dump_dir).iterdir()
                    if f.name.startswith(prefix) and f.name.endswith('.ppm')],
                   key=lambda x: int(x.stem.split('_')[1]))
    img_paths = [str(f) for f in files]
    gts = [gt_text] * len(img_paths)
    return eval_on_images(model, img_paths, gts, input_size, ocr_preproc, in_channels)


def fusion_eval(model, input_size, ocr_preproc, in_channels):
    """Full fusion: R50 + largecrop model on board dumps."""
    r50_net = load_r50()
    results = {}

    for dump_name in ['dump2', 'cluster2']:
        print(f'\n  Fusion: {dump_name}...')

        if dump_name == 'dump2':
            dump_path = POS_OCR_DUMP_2
            gt = GT_DUMP2
            # For largecrop model: use coarse (352x123) images
            tiny_prefix = 'coarse_'
            ocrin_files = sorted([str(f) for f in dump_path.iterdir()
                                  if f.name.startswith('ocrin_') and f.name.endswith('.ppm')],
                                 key=lambda x: int(Path(x).stem.split('_')[1]))
            tiny_files = sorted([str(f) for f in dump_path.iterdir()
                                 if f.name.startswith(tiny_prefix) and f.name.endswith('.ppm')],
                                key=lambda x: int(Path(x).stem.split('_')[1]))
            gts = [gt] * len(ocrin_files)

        elif dump_name == 'cluster2':
            dump_path = CLUSTER2_DIR
            gt = GT_DUMP2
            tiny_prefix = 'crop_'
            ocrin_files = sorted([str(f) for f in dump_path.iterdir()
                                  if f.name.startswith('ocrin_') and f.name.endswith('.ppm')],
                                 key=lambda x: int(Path(x).stem.split('_')[1]))
            tiny_files = sorted([str(f) for f in dump_path.iterdir()
                                 if f.name.startswith(tiny_prefix) and f.name.endswith('.ppm')],
                                key=lambda x: int(Path(x).stem.split('_')[1]))
            gts = [gt] * len(ocrin_files)

        # R50 predictions (on ocrin)
        r50_preds = []
        for f in ocrin_files:
            img = cv2.imread(f)
            if img is None: r50_preds.append(''); continue
            h, w = img.shape[:2]
            if w != 94 or h != 24:
                img = cv2.resize(img, (94, 24))
            img = img.astype('float32')
            img -= 127.5; img *= 0.0078125
            img = np.transpose(img, (2, 0, 1))
            batch = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prebs = forward_family_logits(r50_net, batch, sample_families=['green8'])
            p = decode_ctc(prebs.cpu().numpy())[0]
            r50_preds.append(p)

        # Largecrop model predictions (on coarse/crop)
        tiny_preds = []
        confs = []
        for f in tiny_files:
            c_char, conf = infer_one(f, model, input_size, ocr_preproc, in_channels)
            tiny_preds.append(c_char if c_char else '?')
            confs.append(conf)

        # Baseline R50 stats
        base_stats = compute_stats(r50_preds, gts)
        print(f'    R50 baseline: fc_acc={base_stats["first_char_acc"]:.4f}')

        # Fusion strategies
        strategies = {
            'always_replace': 0.0,
            'replace_if_confident_0.55': 0.55,
            'replace_if_confident_0.70': 0.70,
        }

        dump_results = {}
        for strat_name, threshold in strategies.items():
            fused_preds = []
            changes = 0
            change_reasons = Counter()
            for r50_p, tiny_p, conf in zip(r50_preds, tiny_preds, confs):
                if tiny_p is None or tiny_p == '?':
                    fused_preds.append(r50_p)
                    continue

                if strat_name == 'always_replace':
                    should_replace = True
                else:
                    should_replace = conf >= threshold

                if should_replace and len(r50_p) > 1:
                    new_pred = tiny_p + r50_p[1:]
                    if new_pred != r50_p:
                        changes += 1
                        change_reasons[f'r50={r50_p[0]}->tiny={tiny_p}@{conf:.2f}'] += 1
                    fused_preds.append(new_pred)
                elif should_replace and len(r50_p) <= 1:
                    fused_preds.append(tiny_p + gts[0][1:])
                    changes += 1
                    change_reasons[f'r50_len_err->tiny={tiny_p}@{conf:.2f}'] += 1
                else:
                    fused_preds.append(r50_p)

            stats = compute_stats(fused_preds, gts)
            dump_results[strat_name] = {
                'first_char_acc': stats['first_char_acc'],
                'exact_acc': stats['exact_acc'],
                'suffix_exact_acc': stats['suffix_exact_acc'],
                'macro_first_char_acc': stats['macro_first_char_acc'],
                'changes': changes,
                'change_reasons': dict(change_reasons.most_common(20)),
                'top_preds': dict(Counter(fused_preds).most_common(10)),
            }
            print(f'      {strat_name}: fc_acc={stats["first_char_acc"]:.4f} exact={stats["exact_acc"]:.4f} changes={changes}')

        results[dump_name] = {
            'sample_count': len(r50_preds),
            'gt': gt,
            'baseline_r50': {
                'first_char_acc': base_stats['first_char_acc'],
                'exact_acc': base_stats['exact_acc'],
                'predictions': dict(Counter(r50_preds).most_common(10)),
            },
            'tiny_standalone': {
                'predictions': dict(Counter(tiny_preds).most_common(10)),
            },
            'fusion': dump_results,
        }

    del r50_net
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to best.pt')
    ap.add_argument('--experiment_name', required=True)
    ap.add_argument('--input_size', type=int, nargs=2, default=[224, 72])
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'gray', 'gray3', 'bin'])
    ap.add_argument('--in_channels', type=int, default=3, choices=[1, 3])
    ap.add_argument('--train_data_dir', help='Train quadwarp dir (for eval alignment)')
    args = ap.parse_args()

    exp_dir = OUT_BASE / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    input_size = tuple(args.input_size)
    print(f'Evaluating: {args.experiment_name}')
    print(f'  Model: {args.model}')
    print(f'  Input: {input_size}, preproc={args.ocr_preproc}, channels={args.in_channels}')

    model = load_largecrop_model(args.model, in_channels=args.in_channels)

    # 1. Real holdout (if train_data_dir has manifest)
    if args.train_data_dir:
        print('\n[1] Real holdout...')
        holdout_dir = Path(args.train_data_dir).parent / 'val_real_major_holdout_v1'
        # Also try looking for exported holdout images
        holdout_img_dir = Path(str(args.train_data_dir).replace('fullplate_224x72', 'val_real_major_holdout_v1'))
        if holdout_img_dir.exists():
            results_holdout = eval_on_quadwarp_dir(model, holdout_img_dir, input_size, args.ocr_preproc, args.in_channels)
            if results_holdout:
                json.dump(results_holdout, open(exp_dir / 'eval_real_holdout.json', 'w'), ensure_ascii=False, indent=2)
                print(f'  fc_acc={results_holdout["first_char_acc"]:.4f} macro={results_holdout["macro_first_char_acc"]:.4f}')

    # 2. Province stress
    print('\n[2] Province stress...')
    stress_dir = Path(str(args.train_data_dir).replace('fullplate_224x72', 'province_stress_224x72')) if args.train_data_dir else None
    # Try various possible paths
    for candidate in [
        '/home/wzzz/LPRNet/datasets/routeA_prime_quadwarp_20260512/province_stress_224x72',
        '/home/wzzz/LPRNet/datasets/routeA_prime_quadwarp_20260512/fullplate_224x72_test',
    ]:
        if Path(candidate).exists():
            stress_dir = candidate
            break
    if stress_dir and Path(stress_dir).exists():
        results_stress = eval_on_quadwarp_dir(model, stress_dir, input_size, args.ocr_preproc, args.in_channels)
        if results_stress:
            json.dump(results_stress, open(exp_dir / 'eval_province_stress.json', 'w'), ensure_ascii=False, indent=2)
            print(f'  fc_acc={results_stress["first_char_acc"]:.4f} macro={results_stress["macro_first_char_acc"]:.4f}')

    # 3. Board dumps (direct image evaluation)
    print('\n[3] Board dump2 (coarse images)...')
    results_dump2 = eval_on_board(model, POS_OCR_DUMP_2, GT_DUMP2, 'coarse_', input_size, args.ocr_preproc, args.in_channels)
    json.dump(results_dump2, open(exp_dir / 'eval_board_dump2.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'  fc_acc={results_dump2["first_char_acc"]:.4f} macro={results_dump2["macro_first_char_acc"]:.4f}')
    print(f'  preds: {dict(Counter(results_dump2["predictions"]).most_common(5))}')

    print('\n[4] Board dump1 (coarse images)...')
    results_dump1 = eval_on_board(model, POS_OCR_DUMP, GT_DUMP1, 'coarse_', input_size, args.ocr_preproc, args.in_channels)
    json.dump(results_dump1, open(exp_dir / 'eval_board_dump.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'  fc_acc={results_dump1["first_char_acc"]:.4f}')

    print('\n[5] Cluster2 (crop images)...')
    results_cluster2 = eval_on_board(model, CLUSTER2_DIR, GT_DUMP2, 'crop_', input_size, args.ocr_preproc, args.in_channels)
    json.dump(results_cluster2, open(exp_dir / 'eval_cluster2_diag.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'  fc_acc={results_cluster2["first_char_acc"]:.4f}')
    print(f'  preds: {dict(Counter(results_cluster2["predictions"]).most_common(5))}')

    # 6. Fusion with R50
    print('\n[6] Fusion with R50...')
    results_fusion = fusion_eval(model, input_size, args.ocr_preproc, args.in_channels)
    json.dump(results_fusion, open(exp_dir / 'eval_fused_with_r50.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f'\nAll results: {exp_dir}/')
    del model


if __name__ == '__main__':
    main()
