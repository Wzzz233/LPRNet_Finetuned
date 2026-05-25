#!/usr/bin/env python3
import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from data.load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from train_LPRNet import forward_family_logits
from test_LPRNet import collate_fn

MODEL_PATH = '/home/wzzz/LPRNet/experiments/green_h25/H25D_rear_from_pos2/Final_LPRNet_model.pth'
MANIFEST_PATH = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv'
OUT_PATH = '/home/wzzz/LPRNet/experiments/green_h25/H25D_rear_from_pos2/province_confusion_green8.json'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = UnifiedManifestDataset(
        manifest_path=MANIFEST_PATH,
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
    loader = DataLoader(dataset, batch_size=300, shuffle=False, num_workers=8, collate_fn=collate_fn)

    net = build_lprnet_multihead(
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0.5,
        enhanced_green_head='expD',
    )
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()

    green_conf = defaultdict(Counter)
    overall_conf = defaultdict(Counter)
    green_pair_examples = defaultdict(list)
    green_total = Counter()
    overall_total = Counter()

    seen = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)

            for i, (pred_ids, gt_ids, family) in enumerate(zip(decoded, targets, sample_families)):
                pred_text = ''.join(CHARS[int(c)] for c in pred_ids)
                gt_text = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                gt_first = gt_text[0] if gt_text else '__empty__'
                pred_first = pred_text[0] if pred_text else '__empty__'
                overall_conf[gt_first][pred_first] += 1
                overall_total[gt_first] += 1
                if family == 'green8':
                    green_conf[gt_first][pred_first] += 1
                    green_total[gt_first] += 1
                    if gt_first != pred_first and len(green_pair_examples[(gt_first, pred_first)]) < 5:
                        row = dataset.records[seen + i]
                        green_pair_examples[(gt_first, pred_first)].append({
                            'image_path': row.get('img_path'),
                            'gt': gt_text,
                            'pred': pred_text,
                        })
            seen += len(sample_families)

    def summarize(conf, totals, topk=8):
        rows = {}
        for gt_first, pred_counter in sorted(conf.items()):
            total = totals[gt_first]
            rows[gt_first] = {
                'sample_count': total,
                'top_predictions': [
                    {
                        'pred': pred,
                        'count': count,
                        'ratio': count / total if total else 0.0,
                    }
                    for pred, count in pred_counter.most_common(topk)
                ]
            }
        return rows

    severe = []
    for gt_first, total in green_total.items():
        correct = green_conf[gt_first].get(gt_first, 0)
        acc = correct / total if total else 0.0
        severe.append((acc, gt_first, total))
    severe.sort()

    report = {
        'model': MODEL_PATH,
        'manifest': MANIFEST_PATH,
        'decode_mode': 'family_aware_beam',
        'focus_family': 'green8',
        'green8_confusion': summarize(green_conf, green_total, topk=8),
        'overall_confusion': summarize(overall_conf, overall_total, topk=5),
        'green8_worst_provinces': [
            {
                'province': province,
                'sample_count': total,
                'first_char_acc': acc,
                'top_wrong_predictions': [
                    {
                        'pred': pred,
                        'count': count,
                        'ratio': count / total if total else 0.0,
                        'examples': green_pair_examples.get((province, pred), []),
                    }
                    for pred, count in green_conf[province].most_common(6)
                    if pred != province
                ]
            }
            for acc, province, total in severe[:12]
        ],
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({
        'out_path': OUT_PATH,
        'worst_provinces': report['green8_worst_provinces'][:8],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
