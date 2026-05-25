import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from data.load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from train_LPRNet import forward_family_logits
from test_LPRNet import collate_fn

MODEL_PATH = '/home/wzzz/LPRNet/experiments/green_h25/H25D_rear_from_pos2/Final_LPRNet_model.pth'
MANIFEST_PATH = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv'
OUT_PATH = '/home/wzzz/LPRNet/experiments/green_h25/H25D_rear_from_pos2/province_confusion_green8_fast.json'

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_dataset = UnifiedManifestDataset(
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
    green_indices = [i for i, row in enumerate(full_dataset.records) if (row.get('family') or '').strip() == 'green8']
    dataset = Subset(full_dataset, green_indices)
    loader = DataLoader(dataset, batch_size=300, shuffle=False, num_workers=8, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5, enhanced_green_head='expD')
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()

    conf = defaultdict(Counter)
    totals = Counter()
    examples = defaultdict(list)
    seen = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            sample_families = list(families)
            images = images.to(device)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            for i, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                pred_text = ''.join(CHARS[int(c)] for c in pred_ids)
                gt_text = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                gt_first = gt_text[0] if gt_text else '__empty__'
                pred_first = pred_text[0] if pred_text else '__empty__'
                conf[gt_first][pred_first] += 1
                totals[gt_first] += 1
                if gt_first != pred_first and len(examples[(gt_first, pred_first)]) < 3:
                    row = full_dataset.records[green_indices[seen + i]]
                    examples[(gt_first, pred_first)].append({'img_path': row['img_path'], 'gt': gt_text, 'pred': pred_text})
            seen += len(sample_families)

    rows = {}
    worst = []
    for gt_first in sorted(conf.keys()):
        total = totals[gt_first]
        correct = conf[gt_first].get(gt_first, 0)
        acc = correct / total if total else 0.0
        top_preds = []
        for pred, count in conf[gt_first].most_common(8):
            item = {'pred': pred, 'count': count, 'ratio': count / total if total else 0.0}
            if pred != gt_first:
                item['examples'] = examples.get((gt_first, pred), [])
            top_preds.append(item)
        rows[gt_first] = {'sample_count': total, 'first_char_acc': acc, 'top_predictions': top_preds}
        worst.append((acc, gt_first, total))
    worst.sort()

    report = {
        'model': MODEL_PATH,
        'family': 'green8',
        'sample_count': len(green_indices),
        'province_confusion': rows,
        'worst_provinces': [
            {'province': p, 'sample_count': t, 'first_char_acc': a, 'top_predictions': rows[p]['top_predictions']}
            for a, p, t in worst[:12]
        ]
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\\n')
    print(json.dumps(report['worst_provinces'][:8], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
