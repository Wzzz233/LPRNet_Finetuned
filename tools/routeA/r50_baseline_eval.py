#!/usr/bin/env python3
"""Route A Phase 2: Evaluate R50 on Route A target points.

Measures first_char_acc on:
  1. test_real_holdout_v1.csv  (CCPD2020 green val)
  2. test_province_stress_v1.csv
  3. cluster2_wsl.csv (board ocrin)
  4. POS_OCR_DUMP_2 (static board control)

Output:
  /home/wzzz/LPRNet/experiments/routeA_firstchar_r50_20260512/r50_baseline_eval.json
  /home/wzzz/LPRNet/experiments/routeA_firstchar_r50_20260512/r50_baseline_eval.md
"""
import sys, json, torch, cv2, numpy as np, csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'experiments/routeA_firstchar_r50_20260512'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))
from load_data import CHARS, UnifiedManifestDataset
from train_LPRNet import forward_family_logits
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK = len(CHARS) - 1

R50_CKPT = ROOT / 'experiments/a_ratio_r50_20260510/best_LPRNet_model.pth'
assert R50_CKPT.exists(), f'R50 checkpoint not found: {R50_CKPT}'

# Board dump paths (Windows mounted)
POS_OCR_DUMP = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump')
POS_OCR_DUMP_2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')

# GT for board dumps
GT_DUMP1 = '苏BF01111'
GT_DUMP2 = '京AD06088'

# Manifest paths
REAL_HOLDOUT = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512/test_real_holdout_v1.csv'
PROV_STRESS = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512/test_province_stress_v1.csv'
CLUSTER2_CSV = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv'

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

def load_model():
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                 dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
    net.load_state_dict(torch.load(str(R50_CKPT), map_location='cpu'), strict=False)
    net.to(device); net.eval()
    return net

def infer_one_ocrin(net, ocrin_path):
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

def infer_manifest(net, manifest_path, split='test', max_n=None):
    ds = UnifiedManifestDataset(str(manifest_path), [94, 24], 8, split_filter=split,
                                ocr_preproc='none', dataset_root=str(ROOT))
    results = []
    from torch.utils.data import DataLoader
    from train_LPRNet import collate_fn
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            images = images.to(device)
            fams = [f if f else 'green8' for f in families]
            prebs = forward_family_logits(net, images, sample_families=fams)
            decoded = decode_ctc(prebs.cpu().numpy())
            results.extend(decoded)
            if max_n and len(results) >= max_n:
                break
    return results

def compute_first_char_stats(preds, gts):
    n = len(preds)
    fc_correct = sum(1 for p, g in zip(preds, gts) if len(p) > 0 and len(g) > 0 and p[0] == g[0])
    fc_acc = fc_correct / n if n > 0 else 0.0
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    exact_acc = exact / n if n > 0 else 0.0

    # Province confusion
    prov_conf = Counter()
    for p, g in zip(preds, gts):
        if len(p) > 0 and len(g) > 0 and p[0] != g[0]:
            prov_conf[f'GT:{g[0]}->Pred:{p[0]}'] += 1

    # Per-province accuracy
    per_prov = {}
    for p, g in zip(preds, gts):
        if len(g) == 0: continue
        prov = g[0]
        if prov not in per_prov:
            per_prov[prov] = {'correct': 0, 'total': 0}
        per_prov[prov]['total'] += 1
        if len(p) > 0 and p[0] == prov:
            per_prov[prov]['correct'] += 1

    prov_accs = {}
    for prov, stats in sorted(per_prov.items()):
        prov_accs[prov] = round(stats['correct'] / stats['total'], 4)

    macro_fc = sum(prov_accs.values()) / len(prov_accs) if prov_accs else 0.0

    # Province predictions distribution
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

def eval_on_board_dump(net, dump_path, gt):
    ocrin_files = sorted([f for f in dump_path.iterdir()
                          if f.name.startswith('ocrin_') and f.name.endswith('.ppm')],
                         key=lambda x: int(x.stem.split('_')[1]))
    preds = []
    for f in ocrin_files:
        p = infer_one_ocrin(net, f)
        preds.append(p)
    gts = [gt] * len(preds)
    stats = compute_first_char_stats(preds, gts)
    stats['predictions'] = preds
    return stats

def eval_cluster2_csv(net, csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    preds = []
    gts = []
    for r in rows:
        ocrin_path = r.get('local_ocrin_path', '')
        gt_text = r.get('gt_text', '')
        if not ocrin_path or not Path(ocrin_path).exists():
            continue
        p = infer_one_ocrin(net, ocrin_path)
        preds.append(p)
        gts.append(gt_text)
    stats = compute_first_char_stats(preds, gts)
    stats['predictions'] = preds
    stats['gts'] = gts
    return stats

def main():
    print('=' * 60)
    print('Route A Phase 2: R50 baseline on Route A target points')
    print('=' * 60)

    net = load_model()
    results = {
        'model': str(R50_CKPT),
        'config': {
            'checkpoint': 'best_LPRNet_model.pth',
            'decode': 'greedy (CTC)',
        }
    }

    # 1. Real holdout manifest
    print('\n[1] Real holdout manifest (CCPD2020 green val)...')
    preds_holdout = infer_manifest(net, REAL_HOLDOUT)
    # Load gts
    gts_holdout = []
    with open(REAL_HOLDOUT, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gts_holdout.append(r.get('text', '').strip())
    gts_holdout = gts_holdout[:len(preds_holdout)]
    results['real_holdout'] = compute_first_char_stats(preds_holdout, gts_holdout)
    print(f'  first_char_acc={results["real_holdout"]["first_char_acc"]:.4f}  macro={results["real_holdout"]["macro_first_char_acc"]:.4f}')

    # 2. Province stress
    print('\n[2] Province stress...')
    preds_stress = infer_manifest(net, PROV_STRESS)
    gts_stress = []
    with open(PROV_STRESS, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gts_stress.append(r.get('text', '').strip())
    gts_stress = gts_stress[:len(preds_stress)]
    results['province_stress'] = compute_first_char_stats(preds_stress, gts_stress)
    print(f'  first_char_acc={results["province_stress"]["first_char_acc"]:.4f}  macro={results["province_stress"]["macro_first_char_acc"]:.4f}')

    # 3. Cluster2 board dump
    print('\n[3] Cluster2 board dump...')
    results['cluster2'] = eval_cluster2_csv(net, CLUSTER2_CSV)
    print(f'  first_char_acc={results["cluster2"]["first_char_acc"]:.4f}')
    print(f'  top predictions: {Counter(results["cluster2"]["predictions"]).most_common(5)}')

    # 4. Dump2/static board
    print('\n[4] Dump2 static board control...')
    results['dump2_static'] = eval_on_board_dump(net, POS_OCR_DUMP_2, GT_DUMP2)
    print(f'  first_char_acc={results["dump2_static"]["first_char_acc"]:.4f}')
    print(f'  exact_acc={results["dump2_static"]["exact_acc"]:.4f}')
    print(f'  predictions: {Counter(results["dump2_static"]["predictions"]).most_common(5)}')

    # 5. Dump1 tilt (pos_ocr_dump) - for reference
    print('\n[5] Dump1 tilt (seg_B 11-40)...')
    preds_dump1 = []
    ocrin_files = sorted([f for f in POS_OCR_DUMP.iterdir()
                          if f.name.startswith('ocrin_') and f.name.endswith('.ppm')],
                         key=lambda x: int(x.stem.split('_')[1]))
    for i, f in enumerate(ocrin_files):
        if 11 <= i <= 40:  # seg_B
            p = infer_one_ocrin(net, f)
            preds_dump1.append(p)
    gts_dump1 = [GT_DUMP1] * len(preds_dump1)
    results['dump1_segB'] = compute_first_char_stats(preds_dump1, gts_dump1)
    print(f'  first_char_acc={results["dump1_segB"]["first_char_acc"]:.4f}')
    print(f'  predictions: {Counter(preds_dump1).most_common(5)}')

    # Save JSON
    out_json = OUT_DIR / 'r50_baseline_eval.json'
    json.dump(results, open(out_json, 'w'), ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f'\nSaved JSON: {out_json}')

    # Save MD
    out_md = OUT_DIR / 'r50_baseline_eval.md'
    lines = []
    lines.append('# R50 Baseline on Route A Target Points\n')
    lines.append(f'Checkpoint: `{R50_CKPT}`\n')
    lines.append(f'Decode: greedy CTC\n')
    lines.append('## Results\n\n')
    lines.append(f'| Target | first_char_acc | macro_fc_acc | exact_acc | count |')
    lines.append(f'|--------|:-:|:-:|:-:|:-:|')
    for name in ['real_holdout', 'province_stress', 'cluster2', 'dump2_static', 'dump1_segB']:
        d = results.get(name, {})
        lines.append(f'| {name} | {d.get("first_char_acc","-")} | {d.get("macro_first_char_acc","-")} | {d.get("exact_acc","-")} | {d.get("count","-")} |')
    lines.append('\n## Province Confusion\n')
    for name in ['real_holdout', 'province_stress', 'cluster2', 'dump2_static']:
        d = results.get(name, {})
        conf = d.get('province_confusion_top20', {})
        pred_dist = d.get('province_prediction_distribution', {})
        lines.append(f'\n### {name}\n')
        lines.append(f'province_confusion: {json.dumps(conf, ensure_ascii=False)}')
        lines.append(f'\nprediction_dist: {json.dumps(pred_dist, ensure_ascii=False)}')
        lines.append('')
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved MD: {out_md}')

    del net


if __name__ == '__main__':
    main()
