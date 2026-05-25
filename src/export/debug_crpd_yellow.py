#!/usr/bin/env python3
"""Debug CRPD yellow quad processing - save sample images."""
import sys, os, cv2
sys.path.insert(0, '/home/wzzz/LPRNet/src')
sys.path.insert(0, '/home/wzzz/LPRNet/src/training')
import torch
import numpy as np
from pathlib import Path

# Override CHARS
custom_chars = [l.strip() for l in open('/home/wzzz/LPRNet/keys/yellow_keys.txt', 'r', encoding='utf-8') if l.strip()]
custom_chars.append('-')
import load_data as _ld
_ld.CHARS.clear(); _ld.CHARS.extend(custom_chars)
_ld.CHARS_DICT.clear(); _ld.CHARS_DICT.update({c:i for i,c in enumerate(custom_chars)})
from load_data import CHARS, UnifiedManifestDataset

from LPRNet import build_lprnet

device = torch.device('cuda:0')
state = torch.load('/home/wzzz/LPRNet/experiments/special_yellow_v2/best_LPRNet_model.pth', map_location=device)
class_num = state['container.0.weight'].shape[0]
net = build_lprnet(lpr_max_len=8, phase=False, class_num=class_num, dropout_rate=0)
net.load_state_dict(state)
net.to(device).eval()

# Load manifest (training set which has CRPD data)
dataset = UnifiedManifestDataset(
    '/home/wzzz/LPRNet/manifests/yellow_train.csv',
    img_size=[94, 24], lpr_max_len=8,
    ocr_crop_mode='obb_warp', ocr_channel_order='bgr',
    ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
    ocr_preproc='none', ocr_min_occ_ratio=0.0,
    split_filter='train')

out_dir = Path('/home/wzzz/LPRNet/experiments/special_yellow_v2/crpd_debug')
out_dir.mkdir(exist_ok=True)

crpd_count = 0
for idx in range(len(dataset)):
    row = dataset.records[idx]
    source = row.get('source', '')
    if 'crpd_' not in source:
        continue
    
    gt = row['text']
    img_path = row['img_path']
    
    # Get processed image from dataset
    img_tensor, label, length, family = dataset[idx]
    
    # Also get original raw image
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        continue
    
    # Save processed image (94x24 after pipeline)
    proc_img = ((img_tensor * 128.0) + 127.5).astype('uint8').transpose(1, 2, 0)
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    img_t = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(img_t)
    pred_ids = logits.squeeze(0).argmax(dim=0).cpu().numpy()
    blank_idx = len(CHARS) - 1
    pred_chars = []
    prev = -1
    for cid in pred_ids:
        if cid != prev and cid != blank_idx:
            pred_chars.append(CHARS[int(cid)])
        prev = cid
    pred = ''.join(pred_chars)
    
    # Save debug images
    basename = os.path.basename(img_path)[:50]
    cv2.imwrite(str(out_dir / f'{basename}_raw.jpg'), raw_img)
    cv2.imwrite(str(out_dir / f'{basename}_warped.png'), cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR))
    
    crpd_count += 1
    print(f'[{crpd_count}] {basename}: GT={gt:10s} PRED={pred:10s} raw_shape={raw_img.shape}')
    
    if crpd_count >= 10:
        break

print(f'\nDebug images saved to {out_dir}')
