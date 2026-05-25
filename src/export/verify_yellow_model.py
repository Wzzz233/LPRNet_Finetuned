#!/usr/bin/env python3
"""Verify yellow model: load checkpoint, run inference on training images, check results."""
import sys
sys.path.insert(0, '/home/wzzz/LPRNet/src')
sys.path.insert(0, '/home/wzzz/LPRNet/src/training')
import torch
import cv2
import numpy as np

# Override CHARS to match training
custom_chars = [l.strip() for l in open('/home/wzzz/LPRNet/keys/yellow_keys.txt', 'r', encoding='utf-8') if l.strip()]
custom_chars.append('-')
import load_data as _ld
_ld.CHARS.clear(); _ld.CHARS.extend(custom_chars)
_ld.CHARS_DICT.clear(); _ld.CHARS_DICT.update({c:i for i,c in enumerate(custom_chars)})
from load_data import CHARS, UnifiedManifestDataset
from LPRNet import build_lprnet

device = torch.device('cuda:0')

# 1. Load the TRAINING checkpoint directly
state = torch.load('/home/wzzz/LPRNet/experiments/special_yellow_v3/best_LPRNet_model.pth', map_location=device)
class_num = state['container.0.weight'].shape[0]
print(f'Checkpoint: class_num={class_num}, container.0.weight.shape={state["container.0.weight"].shape}')

net = build_lprnet(lpr_max_len=8, phase=False, class_num=class_num, dropout_rate=0)
net.load_state_dict(state)
net.to(device).eval()

# 2. Test on training manifest (CBLPRD yellow samples)
ds = UnifiedManifestDataset(
    '/home/wzzz/LPRNet/manifests/yellow_train.csv',
    img_size=[94, 24], lpr_max_len=8,
    ocr_crop_mode='obb_warp', ocr_channel_order='bgr',
    ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
    ocr_preproc='none', ocr_min_occ_ratio=0.0,
    split_filter='train')

blank_idx = len(CHARS) - 1
correct = 0
total = 0
errors = []

for idx in range(min(200, len(ds))):
    row = ds.records[idx]
    gt = row['text']
    img_tensor = torch.from_numpy(ds[idx][0]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = net(img_tensor)
    
    pred_ids = logits.squeeze(0).argmax(dim=0).cpu().numpy()
    pred_chars = []
    prev = -1
    for cid in pred_ids:
        if cid != prev and cid != blank_idx:
            pred_chars.append(CHARS[int(cid)])
        prev = cid
    pred = ''.join(pred_chars)
    
    if pred == gt:
        correct += 1
    else:
        errors.append((gt, pred, row.get('source', '')))
    total += 1

print(f'\n=== Yellow v3: Training set evaluation (first {total}) ===')
print(f'Train accuracy: {correct}/{total} = {correct/total*100:.1f}%')
print(f'\nSample errors (first 15):')
for gt, pred, src in errors[:15]:
    print(f'  GT={gt:12s} PRED={pred:12s} src={src}')

# 3. Test on CBLPRD val (test set, same as training evaluation)
print(f'\n=== Test set (CBLPRD val) by source ===')
ds_test = UnifiedManifestDataset(
    '/home/wzzz/LPRNet/manifests/yellow_test.csv',
    img_size=[94, 24], lpr_max_len=8,
    ocr_crop_mode='obb_warp', ocr_channel_order='bgr',
    ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
    ocr_preproc='none', ocr_min_occ_ratio=0.0)

from collections import defaultdict
by_src = defaultdict(lambda: {'ok':0,'tot':0})
for idx in range(len(ds_test)):
    row = ds_test.records[idx]
    gt = row['text']
    src = row.get('source','')
    img_tensor = torch.from_numpy(ds_test[idx][0]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(img_tensor)
    pred_ids = logits.squeeze(0).argmax(dim=0).cpu().numpy()
    pred_chars = []
    prev = -1
    for cid in pred_ids:
        if cid != prev and cid != blank_idx:
            pred_chars.append(CHARS[int(cid)])
        prev = cid
    pred = ''.join(pred_chars)
    by_src[src]['tot'] += 1
    if pred == gt:
        by_src[src]['ok'] += 1

for src, v in sorted(by_src.items()):
    print(f'  {src:25s}: {v["ok"]}/{v["tot"]} = {v["ok"]/v["tot"]*100:.1f}%')
