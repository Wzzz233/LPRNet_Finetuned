#!/usr/bin/env python3
"""Route A Phase 4+5: Eval + fusion for tiny province net with R50.

Evaluates a trained tiny province net on all Route A target points,
then fuses with R50 (replace only first char).

Usage:
  python tools/routeA/eval_and_fuse.py \
    --tiny_model /path/to/best.pt \
    --experiment_name A1_ocr94x24_color \
    --input_mode ocr_94x24 \
    --ocr_preproc none

Output in experiments/routeA_firstchar_r50_20260512/{experiment_name}/
  eval_real_holdout.json
  eval_province_stress.json
  eval_cluster2_board.json
  eval_dump2_board.json
  eval_fused_with_r50.json
"""
import sys, json, torch, cv2, numpy as np, csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
OUT_BASE = ROOT / 'experiments/routeA_firstchar_r50_20260512'

sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))
from load_data import CHARS, UnifiedManifestDataset
from train_LPRNet import forward_family_logits, collate_fn
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK = len(CHARS) - 1

R50_CKPT = ROOT / 'experiments/a_ratio_r50_20260510/best_LPRNet_model.pth'
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')
GT_DUMP1 = '苏BF01111'
GT_DUMP2 = '京AD06088'

# Province chars list (31 provinces)
PROVINCE_CHARS = ['京', '津', '冀', '晋', '蒙', '辽', '吉', '黑',
                  '沪', '苏', '浙', '皖', '闽', '赣', '鲁', '豫',
                  '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云',
                  '藏', '陕', '甘', '青', '宁', '新', '渝']


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


class TinyProvinceNet(torch.nn.Module):
    def __init__(self, num_classes=31, in_channels=3):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_tiny(model_path, in_channels=1):
    net = TinyProvinceNet(in_channels=in_channels)
    net.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    net.to(device); net.eval()
    return net


def load_tiny_color(model_path):
    return load_tiny(model_path, in_channels=3)


def infer_r50_one(net, ocrin_path):
    img = cv2.imread(str(ocrin_path))
    if img is None: return ''
    h, w = img.shape[:2]
    if w != 94 or h != 24:
        img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5; img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    batch = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prebs = forward_family_logits(net, batch, sample_families=['green8'])
    return decode_ctc(prebs.cpu().numpy())[0]


def infer_tiny_one(tiny_net, ocrin_path, ocr_preproc, input_mode):
    """Infer first-char province from an ocrin image using the tiny net."""
    img = cv2.imread(str(ocrin_path))
    if img is None:
        return None, 0.0
    
    if input_mode == 'ocr_94x24':
        h, w = img.shape[:2]
        if w != 94 or h != 24:
            img = cv2.resize(img, (94, 24))
    elif input_mode == 'full_crop':
        # For full_crop mode, use larger resize
        target_h = 64
        target_w = int(round(target_h * 128 / 48))
        img = cv2.resize(img, (target_w, target_h))

    # Apply preproc
    # gray3 uses in_channels=1 model (train's convert_batch_inputs reduces to 1ch)
    if ocr_preproc in ('gray', 'gray3'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]  # single channel
    elif ocr_preproc == 'bin':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    img_t = torch.from_numpy(img.astype('float32') / 255.0)
    if img_t.ndim == 2:
        img_t = img_t.unsqueeze(-1)
    img_t = img_t.permute(2, 0, 1).contiguous()
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = tiny_net(img_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    province_char = PROVINCE_CHARS[pred.item()] if pred.item() < len(PROVINCE_CHARS) else '?'
    return province_char, float(conf.item())


def compute_first_char_stats(preds, gts):
    n = len(preds)
    fc_correct = sum(1 for p, g in zip(preds, gts) if len(p) > 0 and len(g) > 0 and p[0] == g[0])
    fc_acc = fc_correct / n if n > 0 else 0.0
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    exact_acc = exact / n if n > 0 else 0.0
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
        'macro_first_char_acc': round(macro_fc, 4),
        'province_confusion_top20': dict(prov_conf.most_common(20)),
        'per_province_fc_acc': prov_accs,
        'province_prediction_distribution': dict(prov_preds.most_common()),
    }


def eval_tiny_on_manifest(tiny_net, manifest_path, split, ocr_preproc, input_mode, max_n=None):
    """Evaluate tiny net on manifest-based dataset."""
    import csv
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    preds = []
    gts = []
    for r in rows:
        if split and r.get('split', '').strip() != split:
            continue
        text = r.get('text', '').strip()
        img_path = r.get('img_path', '').strip()
        if not text or not img_path:
            continue
        full_path = str(ROOT / img_path) if not img_path.startswith('/') else img_path
        if not Path(full_path).exists():
            continue
        p, _ = infer_tiny_one(tiny_net, full_path, ocr_preproc, input_mode)
        if p:
            preds.append(p + text[1:] if p else text)
            gts.append(text)
        if max_n and len(preds) >= max_n:
            break
    return compute_first_char_stats(preds, gts)


def eval_tiny_on_board(tiny_net, ocrin_dir, gt_text, ocr_preproc, input_mode):
    """Evaluate tiny net on board dump ocrin images."""
    # For full_crop mode, use coarse_*.ppm (larger pre-OCR crop)
    prefix = 'coarse_' if input_mode == 'full_crop' else 'ocrin_'
    ocrin_files = sorted([f for f in Path(ocrin_dir).iterdir()
                          if f.name.startswith(prefix) and f.name.endswith('.ppm')],
                         key=lambda x: int(x.stem.split('_')[1]))
    tiny_preds = []
    full_preds = []
    confidences = []
    for f in ocrin_files:
        p_char, conf = infer_tiny_one(tiny_net, f, ocr_preproc, input_mode)
        confidences.append(conf)
        if p_char:
            tiny_preds.append(p_char)
            full_preds.append(p_char + gt_text[1:])
        else:
            tiny_preds.append('?')
            full_preds.append(gt_text)

    gts = [gt_text] * len(full_preds)
    stats = compute_first_char_stats(full_preds, gts)
    stats['predictions'] = full_preds
    stats['tiny_predictions'] = tiny_preds
    stats['confidences'] = [round(c, 4) for c in confidences]
    return stats


def eval_tiny_on_cluster2_csv(tiny_net, csv_path, ocr_preproc, input_mode):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    r50_preds = []
    tiny_preds = []
    full_preds = []
    gts = []
    confidences = []
    r50_net = load_r50()

    for r in rows:
        ocrin_path = r.get('local_ocrin_path', '')
        # For full_crop mode, use the larger crop image
        img_path = r.get('local_crop_path', ocrin_path) if input_mode == 'full_crop' else ocrin_path
        gt_text = r.get('gt_text', '')
        if not img_path or not Path(img_path).exists():
            continue

        # R50 prediction (always uses ocrin)
        r50_pred = infer_r50_one(r50_net, ocrin_path)
        r50_preds.append(r50_pred)

        # Tiny prediction (uses crop_path for full_crop mode)
        p_char, conf = infer_tiny_one(tiny_net, img_path, ocr_preproc, input_mode)
        confidences.append(conf)
        tiny_preds.append(p_char if p_char else '?')
        if p_char and len(r50_pred) > 1:
            full_preds.append(p_char + r50_pred[1:])
        elif p_char:
            full_preds.append(p_char + gt_text[1:])
        else:
            full_preds.append(gt_text)
        gts.append(gt_text)

    stats = compute_first_char_stats(full_preds, gts)
    stats['r50_preds'] = r50_preds
    stats['tiny_preds'] = tiny_preds
    stats['predictions'] = full_preds
    stats['confidences'] = [round(c, 4) for c in confidences]
    del r50_net
    return stats


def fusion_eval(tiny_net, ocr_preproc, input_mode):
    """Full fusion evaluation: R50 + tiny net on board dumps with 3 fusion strategies."""
    r50_net = load_r50()
    results = {}

    for dump_name, dump_path, gt_text in [
        ('cluster2', ROOT / 'tmp/ocr_dump_new_dump_20260416', GT_DUMP2),
        ('dump2', POS_OCR_DUMP_2, GT_DUMP2),
    ]:
        print(f'  Processing {dump_name}...')

        # Load samples
        if dump_name == 'cluster2':
            csv_path = dump_path / 'cluster2_wsl.csv'
            rows = []
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                for r2 in csv.DictReader(f):
                    rows.append(r2)
            # For full_crop: use local_crop_path (96x277) instead of local_ocrin_path (24x94)
            if input_mode == 'full_crop':
                tiny_img_paths = [r2.get('local_crop_path', r2.get('local_ocrin_path', '')) for r2 in rows]
            else:
                tiny_img_paths = [r2.get('local_ocrin_path', '') for r2 in rows]
            ocrin_files = [r2.get('local_ocrin_path', '') for r2 in rows]  # R50 always uses ocrin
            gts = [r2.get('gt_text', '') for r2 in rows]
        else:
            ocrin_files = sorted([str(f) for f in dump_path.iterdir()
                                  if f.name.startswith('ocrin_') and f.name.endswith('.ppm')],
                                 key=lambda x: int(Path(x).stem.split('_')[1]))
            # For full_crop: use coarse_*.ppm instead of ocrin_*.ppm
            if input_mode == 'full_crop':
                tiny_img_files = sorted([str(f) for f in dump_path.iterdir()
                                         if f.name.startswith('coarse_') and f.name.endswith('.ppm')],
                                        key=lambda x: int(Path(x).stem.split('_')[1]))
            else:
                tiny_img_files = ocrin_files
            gts = [gt_text] * len(ocrin_files)

        # Get R50 predictions
        r50_preds = []
        for f in ocrin_files:
            p = infer_r50_one(r50_net, f)
            r50_preds.append(p)

        # Get tiny net predictions
        tiny_preds = []
        confs = []
        # Determine tiny net input paths
        if dump_name == 'cluster2':
            tiny_inputs = tiny_img_paths
        else:
            tiny_inputs = tiny_img_files if input_mode == 'full_crop' else ocrin_files
        for f in tiny_inputs:
            p_char, conf = infer_tiny_one(tiny_net, f, ocr_preproc, input_mode)
            tiny_preds.append(p_char)
            confs.append(conf)

        # Compute baseline R50 first-char stats
        base_stats = compute_first_char_stats(r50_preds, gts)
        base_first_char_acc = base_stats['first_char_acc']
        base_exact_acc = base_stats['exact_acc']

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

                # Determine replacement
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
                else:
                    if should_replace and len(r50_p) <= 1:
                        fused_preds.append(tiny_p + gt_text[1:])
                        changes += 1
                        change_reasons[f'r50_len_err->tiny={tiny_p}@{conf:.2f}'] += 1
                    else:
                        fused_preds.append(r50_p)

            stats = compute_first_char_stats(fused_preds, gts)
            dump_results[strat_name] = {
                'first_char_acc': stats['first_char_acc'],
                'exact_acc': stats['exact_acc'],
                'changes': changes,
                'change_reasons': dict(change_reasons.most_common(20)),
                'top_preds': dict(Counter(fused_preds).most_common(10)),
                'per_province_fc_acc': stats['per_province_fc_acc'],
            }

        results[dump_name] = {
            'sample_count': len(r50_preds),
            'gt': gt_text if dump_name == 'dump2' else '京AD06088',
            'baseline_r50': {
                'first_char_acc': base_first_char_acc,
                'exact_acc': base_exact_acc,
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
    ap.add_argument('--tiny_model', required=True)
    ap.add_argument('--experiment_name', required=True)
    ap.add_argument('--input_mode', default='ocr_94x24', choices=['ocr_94x24', 'full_crop'])
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'gray', 'gray3', 'bin'])
    args = ap.parse_args()

    exp_dir = OUT_BASE / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f'Experiment dir: {exp_dir}')

    # Determine in_channels for tiny model
    # gray3 is trained with in_channels=1 (convert_batch_inputs reduces to 1ch)
    # gray also uses in_channels=1
    # none/bin uses in_channels=3
    in_channels = 1 if args.ocr_preproc in ('gray', 'gray3') else 3
    tiny_net = load_tiny(args.tiny_model, in_channels=in_channels)

    # ── Standalone tiny net eval ──
    print('\n[1] Tiny standalone on real holdout...')
    real_holdout = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512/test_real_holdout_v1.csv'
    holdout_results = eval_tiny_on_manifest(tiny_net, real_holdout, 'test', args.ocr_preproc, args.input_mode)
    json.dump(holdout_results, open(exp_dir / 'eval_real_holdout.json', 'w'), ensure_ascii=False, indent=2)
    print(f'  first_char_acc={holdout_results["first_char_acc"]:.4f}  macro={holdout_results["macro_first_char_acc"]:.4f}')

    print('\n[2] Tiny standalone on province stress...')
    prov_stress = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512/test_province_stress_v1.csv'
    stress_results = eval_tiny_on_manifest(tiny_net, prov_stress, 'test', args.ocr_preproc, args.input_mode)
    json.dump(stress_results, open(exp_dir / 'eval_province_stress.json', 'w'), ensure_ascii=False, indent=2)
    print(f'  first_char_acc={stress_results["first_char_acc"]:.4f}  macro={stress_results["macro_first_char_acc"]:.4f}')

    print('\n[3] Tiny standalone on cluster2 board dump...')
    cluster2_csv = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv'
    cluster2_results = eval_tiny_on_cluster2_csv(tiny_net, cluster2_csv, args.ocr_preproc, args.input_mode)
    json.dump(cluster2_results, open(exp_dir / 'eval_cluster2_board.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'  first_char_acc={cluster2_results["first_char_acc"]:.4f}')
    print(f'  top tiny preds: {Counter(cluster2_results.get("tiny_preds", [])).most_common(5)}')
    print(f'  top fused preds: {Counter(cluster2_results.get("predictions", [])).most_common(5)}')

    print('\n[4] Tiny standalone on dump2 static board...')
    dump2_results = eval_tiny_on_board(tiny_net, POS_OCR_DUMP_2, GT_DUMP2, args.ocr_preproc, args.input_mode)
    json.dump(dump2_results, open(exp_dir / 'eval_dump2_board.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'  first_char_acc={dump2_results["first_char_acc"]:.4f}')
    print(f'  top tiny preds: {Counter(dump2_results.get("tiny_predictions", [])).most_common(5)}')

    # ── Fusion eval with R50 ──
    print('\n[5] Fusion with R50 (3 strategies)...')
    fusion_results = fusion_eval(tiny_net, args.ocr_preproc, args.input_mode)
    json.dump(fusion_results, open(exp_dir / 'eval_fused_with_r50.json', 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)

    for dump_name in ['cluster2', 'dump2']:
        d = fusion_results.get(dump_name, {})
        print(f'\n  {dump_name}:')
        print(f'    baseline: fc_acc={d.get("baseline_r50",{}).get("first_char_acc","-")}  exact={d.get("baseline_r50",{}).get("exact_acc","-")}')
        for strat_name, sd in d.get('fusion', {}).items():
            print(f'    {strat_name}: fc_acc={sd["first_char_acc"]}  exact={sd["exact_acc"]}  changes={sd["changes"]}')

    print(f'\nAll eval results saved to: {exp_dir}/')
    del tiny_net


if __name__ == '__main__':
    main()
