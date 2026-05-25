#!/usr/bin/env python3
"""Convert expD checkpoint to expE warmstart for green8 head.
expD: Conv2d(516→256,3x3) → ReLU → Dropout → Conv2d(256→68,1x1)
expE: Conv2d(516→512,3x3) → ReLU → Drop → Conv2d(512→256,1x1) → ReLU → Drop → Conv2d(256→68,1x1)

Mapping:
- Backbone: copy all
- normal7/special containers: copy all (same architecture)
- expD conv0 weight [256,516,3,3] → expE conv0 first 256 ch, last 256 ch init small
- expD conv3 weight [68,256,1,1] → expE conv6 [68,256,1,1] (direct copy)
- expE conv3 [256,512,1,1]: new, identity-like init for 256→256 sub-block"""

import torch, sys
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'training'))

from LPRNet_multihead import build_lprnet_multihead
from load_data import CHARS

INPUT_CKP = ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/best_LPRNet_model.pth'
OUTPUT_CKP = ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/best_expE_warmstart.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load expD model ─────────────────────────────────────────────────
print("Loading expD model...", flush=True)
expD_state = torch.load(str(INPUT_CKP), map_location='cpu')
print(f"  expD keys: {len(expD_state)}", flush=True)

# ── Build expE model ────────────────────────────────────────────────
print("Building expE model...", flush=True)
expE_net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                   dropout_rate=0.5, enhanced_green_head='expE', pos0_head_cols=0)
expE_state = expE_net.state_dict()
print(f"  expE keys: {len(expE_state)}", flush=True)

# ── Build mapping ───────────────────────────────────────────────────
mapped = {}
missing = []
shape_mismatch = []

for key in expE_state:
    target_shape = expE_state[key].shape
    if key in expD_state:
        source_tensor = expD_state[key]
        if source_tensor.shape == target_shape:
            mapped[key] = source_tensor.clone()
        else:
            # Special handling for green8 container (expD→expE architecture diff)
            shape_mismatch.append(f"{key}: expD={list(source_tensor.shape)} expE={list(target_shape)}")
    else:
        missing.append(key)

print(f"\n  Directly mapped: {len(mapped)}")
print(f"  Shape mismatches: {len(shape_mismatch)}")
for s in shape_mismatch:
    print(f"    {s}")
print(f"  Missing in expD: {len(missing)}")
for k in missing:
    print(f"    {k}")

# ── Green8 head warmstart ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Green8 Head Warmstart Mapping")
print(f"{'='*60}")

# expD containers.green8.0.weight: [256, 516, 3, 3]
# expE containers.green8.0.weight: [512, 516, 3, 3]
d_conv0_w = expD_state['containers.green8.0.weight']  # [256,516,3,3]
d_conv0_b = expD_state['containers.green8.0.bias']    # [256]

e_conv0_w = torch.zeros(512, 516, 3, 3)
e_conv0_b = torch.zeros(512)

# Copy first 256 channels from expD
e_conv0_w[:256, :, :, :] = d_conv0_w
e_conv0_b[:256] = d_conv0_b

# Initialize last 256 channels with small noise
torch.nn.init.kaiming_normal_(e_conv0_w[256:, :, :, :], mode='fan_in', nonlinearity='relu')
e_conv0_w[256:, :, :, :] *= 0.1  # Scale down to not disrupt existing behavior
e_conv0_b[256:] = 0.0

mapped['containers.green8.0.weight'] = e_conv0_w
mapped['containers.green8.0.bias'] = e_conv0_b
print(f"  containers.green8.0: [256→512 ch] copied first 256, new 256 Kaiming*0.1")

# expE containers.green8.3.weight: [256, 512, 1, 1] — NEW middle layer
# Identity-like init: for each output channel i, weight[i, i if i<512 else i-256, 0, 0] = 1
e_conv3_w = torch.zeros(256, 512, 1, 1)
e_conv3_b = torch.zeros(256)

# Make first 256 input → first 256 output approximately identity
for i in range(256):
    e_conv3_w[i, i, 0, 0] = 1.0
    # Small random perturbation
    e_conv3_w[i, :, 0, 0] += torch.randn(512) * 0.01

mapped['containers.green8.3.weight'] = e_conv3_w
mapped['containers.green8.3.bias'] = e_conv3_b
print(f"  containers.green8.3: NEW [512→256 ch] identity-like init")

# expD containers.green8.3.weight: [68, 256, 1, 1]
# expE containers.green8.6.weight: [68, 256, 1, 1]
# Direct copy
mapped['containers.green8.6.weight'] = expD_state['containers.green8.3.weight'].clone()
mapped['containers.green8.6.bias'] = expD_state['containers.green8.3.bias'].clone()
print(f"  containers.green8.6: copied from expD green8.3 (direct match)")

# ── Check remaining unset keys ─────────────────────────────────────
e_remaining = [k for k in expE_state if k not in mapped]
if e_remaining:
    print(f"\n  WARNING: {len(e_remaining)} keys still not mapped:")
    for k in e_remaining:
        print(f"    {k}")
    # These should all be ReLU/Dropout (non-parameter) or filled below

# Load mapped state
print(f"\n  Loading mapped state into expE model...", flush=True)
expE_net.load_state_dict(mapped, strict=False)
missing_keys, unexpected_keys = expE_net.load_state_dict(mapped, strict=False)
print(f"  Missing keys (in model not in mapped): {len(missing_keys)}")
for k in missing_keys:
    print(f"    {k}")
print(f"  Unexpected keys (in mapped not in model): {len(unexpected_keys)}")
for k in unexpected_keys:
    print(f"    {k}")

expE_net.to(device)
expE_net.eval()

# ── Logit difference test ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Logit Difference Test: expD vs expE warmstart")
print(f"{'='*60}")

# Build expD for comparison
expD_net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                   dropout_rate=0.5, enhanced_green_head='expD', pos0_head_cols=0)
expD_net.load_state_dict(expD_state, strict=False)
expD_net.to(device)
expD_net.eval()

import numpy as np
import cv2

# Load a real test image
test_img_path = str(ROOT / 'datasets/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/images/val/0304-24_23-382&446_544&603-544&529_384&603_382&520_542&446-0_1_0_29_33_32_33-103-119_green_京DF53383.jpg')
img = cv2.imread(test_img_path)
if img is not None:
    img = cv2.resize(img, (94, 24))
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
    img_tensor = (img_tensor - 127.5) * 0.0078125
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_d = expD_net(img_tensor)
        out_e = expE_net(img_tensor)
        
        logits_d = out_d['green8']  # [1, 68, 18]
        logits_e = out_e['green8']  # [1, 68, 18]
        
        diff = (logits_d - logits_e).abs()
        print(f"  Green8 logit difference:")
        print(f"    Mean abs diff: {diff.mean().item():.4f}")
        print(f"    Max abs diff:  {diff.max().item():.4f}")
        print(f"    Std of diff:   {diff.std().item():.4f}")
        print(f"    Logit_d mean:  {logits_d.mean().item():.2f}")
        print(f"    Logit_e mean:  {logits_e.mean().item():.2f}")
        
        # Compare argmax predictions
        pred_d = logits_d.argmax(dim=1).cpu().numpy()[0]
        pred_e = logits_e.argmax(dim=1).cpu().numpy()[0]
        same = (pred_d == pred_e).mean()
        print(f"\n  Argmax agreement: {same*100:.1f}%")
        
        if same < 0.5:
            print(f"  ⚠️ WARNING: argmax agreement < 50%. Warmstart may be too aggressive.")
            print(f"  pred_d: {pred_d.tolist()}")
            print(f"  pred_e: {pred_e.tolist()}")
        else:
            print(f"  ✅ Warmstart preserves most predictions.")
else:
    print(f"  SKIP: could not load test image")

# ── Save ───────────────────────────────────────────────────────────
# Build clean state dict from mapped
clean_state = expE_net.state_dict()
for k, v in mapped.items():
    if k in clean_state:
        clean_state[k] = v

torch.save(clean_state, OUTPUT_CKP)
print(f"\n  Saved: {OUTPUT_CKP}")
print(f"  Size: {OUTPUT_CKP.stat().st_size / 1e6:.1f} MB")
print(f"  Done.")
