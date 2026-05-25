#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data.load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits

MODEL = '/home/wzzz/LPRNet/experiments/green_h29/H29B_anhui40_noleak/Final_LPRNet_model.pth'
MANIFEST = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv'
OUT_JSON = '/home/wzzz/LPRNet/experiments/green_h29/H29B_anhui40_noleak/diag_su_hu_predictions.json'
TARGETS = {'苏', '沪'}


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full = UnifiedManifestDataset(
        manifest_path=MANIFEST,
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter='test',
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i, row in enumerate(full.records) if (row.get('family') or '').strip() == 'green8' and (row.get('text') or '')[:1] in TARGETS]
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(MODEL, map_location=device))
    net.to(device)
    net.eval()

    records = []
    province_conf = defaultdict(Counter)
    exact_counter = Counter()
    first_counter = Counter()
    len_counter = Counter()
    suffix_counter = Counter()

    ds_indices = ds.indices
    cursor = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            batch_size = len(lengths)
            meta_rows = [full.records[ds_indices[cursor + b]] for b in range(batch_size)]
            cursor += batch_size
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            for meta, pred_ids, gt_ids in zip(meta_rows, decoded, targets):
                pred = ''.join(CHARS[int(c)] for c in pred_ids)
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                prov = gt[:1]
                pred_prov = pred[:1] if pred else ''
                province_conf[prov][pred_prov] += 1
                exact_counter[prov] += int(pred == gt)
                first_counter[prov] += int(bool(pred) and pred[0] == gt[0])
                len_counter[prov] += int(len(pred) == len(gt))
                suffix_counter[prov] += int(len(pred) >= 2 and len(gt) >= 2 and pred[1:] == gt[1:])
                records.append({
                    'province': prov,
                    'img_path': meta.get('img_path'),
                    'gt': gt,
                    'pred': pred,
                    'first_char_ok': bool(pred) and pred[0] == gt[0],
                    'exact_ok': pred == gt,
                    'len_ok': len(pred) == len(gt),
                    'suffix_ok': len(pred) >= 2 and len(gt) >= 2 and pred[1:] == gt[1:],
                    'pred_first_char': pred_prov,
                })

    summary = {}
    for prov in sorted(TARGETS):
        subset = [r for r in records if r['province'] == prov]
        total = len(subset)
        summary[prov] = {
            'sample_count': total,
            'exact_acc': exact_counter[prov] / total if total else 0.0,
            'first_char_acc': first_counter[prov] / total if total else 0.0,
            'len_acc': len_counter[prov] / total if total else 0.0,
            'suffix_acc': suffix_counter[prov] / total if total else 0.0,
            'first_char_confusion_top': province_conf[prov].most_common(10),
            'failure_examples': [r for r in subset if not r['exact_ok']][:25],
        }

    out = {
        'model': MODEL,
        'manifest': MANIFEST,
        'targets': sorted(TARGETS),
        'summary': summary,
        'all_records': records,
    }
    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out_json': OUT_JSON, 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
