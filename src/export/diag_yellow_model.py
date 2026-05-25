#!/usr/bin/env python3
"""Eval CRPD yellow accurately."""
import sys
sys.path.insert(0, '/home/wzzz/LPRNet/src')
sys.path.insert(0, '/home/wzzz/LPRNet/src/training')
import torch
import numpy as np

custom_chars = [l.strip() for l in open('/home/wzzz/LPRNet/keys/yellow_keys.txt', 'r', encoding='utf-8') if l.strip()]
custom_chars.append('-')
import load_data as _ld
_ld.CHARS.clear(); _ld.CHARS.extend(custom_chars)
_ld.CHARS_DICT.clear(); _ld.CHARS_DICT.update({c:i for i,c in enumerate(custom_chars)})
from load_data import CHARS, UnifiedManifestDataset
from LPRNet import build_lprnet

device = torch.device('cuda:0')
state = torch.load('/home/wzzz/LPRNet/experiments/special_yellow_v2/best_LPRNet_model.pth', map_location=device)
net = build_lprnet(lpr_max_len=8, phase=False, class_num=state['container.0.weight'].shape[0], dropout_rate=0)
net.load_state_dict(state)
net.to(device).eval()
blank_idx = len(CHARS) - 1

ds = UnifiedManifestDataset('/home/wzzz/LPRNet/manifests/yellow_train.csv', [94,24], 8,
    ocr_crop_mode='obb_warp', ocr_preproc='none', ocr_min_occ_ratio=0.0)

crpd_indices = [i for i, r in enumerate(ds.records) if 'crpd_' in r.get('source','')]
print(f'Total CRPD yellow samples: {len(crpd_indices)}')

correct, total = 0, 0
for idx in crpd_indices:
    row = ds.records[idx]
    gt = row['text']
    img_t = torch.from_numpy(ds[idx][0]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(img_t)
    pred_ids = logits.squeeze(0).argmax(dim=0).cpu().numpy()
    chars = []
    prev = -1
    for cid in pred_ids:
        if cid != prev and cid != blank_idx:
            chars.append(CHARS[int(cid)])
        prev = cid
    pred = ''.join(chars)
    if pred == gt:
        correct += 1
    total += 1
    if total % 1000 == 0:
        print(f'  ... {total}/{len(crpd_indices)} correct={correct}/{total}={correct/total*100:.1f}%')

print(f'\nCRPD yellow total: {correct}/{total} = {correct/total*100:.1f}%')
