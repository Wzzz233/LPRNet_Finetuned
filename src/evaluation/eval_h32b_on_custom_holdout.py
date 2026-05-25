#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data.load_data import CCPDBoardDataLoader, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def safe_div(a,b):
    return float(a)/float(b) if b else 0.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--txt', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--out_json', required=True)
    args=ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = CCPDBoardDataLoader(
        [args.image_root],
        imgSize=[94,24],
        lpr_max_len=8,
        txt_file=args.txt,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device)
    net.eval()

    total=0; exact=0; first=0
    prov_rows=defaultdict(lambda: {'sample_count':0,'exact':0,'first':0})
    idx0 = 0
    with torch.no_grad():
        for images, labels, lengths, _families in loader:
            start = 0
            targets=[]
            fams=[]
            for length in lengths:
                tgt = labels[start:start+length].numpy()
                targets.append(tgt)
                fams.append('green8')
                start += length
            images=images.to(device)
            logits = forward_family_logits(net, images, sample_families=fams).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=fams)
            for pred_ids, gt_ids in zip(decoded, targets):
                pred=''.join(CHARS[int(c)] for c in pred_ids)
                gt=''.join(CHARS[int(c)] for c in gt_ids.tolist())
                prov=gt[0] if gt else ''
                total += 1
                exact += int(pred==gt)
                first += int(bool(pred) and gt and pred[0]==gt[0])
                prov_rows[prov]['sample_count'] += 1
                prov_rows[prov]['exact'] += int(pred==gt)
                prov_rows[prov]['first'] += int(bool(pred) and gt and pred[0]==gt[0])

    out = {
        'sample_count': total,
        'exact_plate_acc': safe_div(exact,total),
        'first_char_acc': safe_div(first,total),
        'province_breakdown': {
            p: {
                'sample_count': v['sample_count'],
                'exact_plate_acc': safe_div(v['exact'], v['sample_count']),
                'first_char_acc': safe_div(v['first'], v['sample_count']),
            }
            for p,v in sorted(prov_rows.items())
        }
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
