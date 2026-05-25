#!/usr/bin/env python3
"""Evaluate Stage 1 checkpoint on green8-only val (green8 guard check).
Canonical protocol: cv2.imread, forward_family_logits, post-province metrics.
"""
import sys, json, cv2, torch, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))
from load_data import CHARS
from train_LPRNet import forward_family_logits
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0')
BLANK = len(CHARS) - 1

# Load Stage 1 best checkpoint — exact same architecture as training
ckpt_path = ROOT / 'experiments/b_prime_stage1_real_multidomain_20260510/best_LPRNet_model.pth'
net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                             dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
# Remove multihead prefix from state dict keys if present
clean_state = {}
for k, v in state.items():
    clean_key = k.replace('module.', '', 1) if k.startswith('module.') else k
    clean_state[clean_key] = v
net.load_state_dict(clean_state, strict=False)
net.to(device)
net.eval()

# Load val_ccpd2020_green manifest
import csv
manifest_path = ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv'
with open(manifest_path) as f:
    reader = csv.DictReader(f)
    samples = []
    for r in reader:
        img_path = r['img_path']
        resolved = ROOT / img_path
        if resolved.exists():
            samples.append({'path': str(resolved), 'text': r['text'], 'family': r.get('family', 'green8')})

print(f"Loaded {len(samples)} green8 val samples")

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

exact_ok = 0
fc_ok = 0
total_chars = 0
correct_chars = 0

with torch.no_grad():
    for s in samples:
        img = cv2.imread(s['path'])
        if img is None: continue
        img = img.astype('float32')
        img -= 127.5; img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        batch = torch.from_numpy(img).unsqueeze(0).to(device)
        prebs = forward_family_logits(net, batch, sample_families=[s['family']])
        pred = decode_ctc(prebs.cpu().numpy())[0]
        
        gt = s['text']
        if pred == gt:
            exact_ok += 1
        if len(pred) > 0 and len(gt) > 0 and pred[0] == gt[0]:
            fc_ok += 1
        total_chars += len(gt)
        correct_chars += sum(1 for p, g in zip(pred, gt) if p == g)

n = len(samples)
print(f"\nGreen8 guard results:")
print(f"  exact: {exact_ok}/{n} = {exact_ok/n*100:.1f}%")
print(f"  fc: {fc_ok}/{n} = {fc_ok/n*100:.1f}%")
print(f"  char: {correct_chars}/{total_chars} = {correct_chars/total_chars*100:.1f}%")
print(f"\nR50 comparison (from earlier eval):")
print(f"  R50 green_val proxy_exact (greedy): 28.3% (epoch 7)")
print(f"  Stage 1 green_val exact: {exact_ok/n*100:.1f}%")

# Guard condition check
guard_exact = exact_ok/n*100
if guard_exact >= 10.0:
    print(f"\n✅ GREEN8 GUARD PASSED: exact={guard_exact:.1f}% >= 10.0%")
    print(f"  Proceed to Stage 2")
else:
    print(f"\n❌ GREEN8 GUARD FAILED: exact={guard_exact:.1f}% < 10.0%")
    print(f"  STOP — do not open Stage 2")
