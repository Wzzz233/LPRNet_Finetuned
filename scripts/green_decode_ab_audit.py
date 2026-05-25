#!/usr/bin/env python3
"""Decode mode A/B audit: greedy vs green_ctc_beam vs family_aware_beam."""

import sys, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'training'))
sys.path.insert(0, str(ROOT / 'src' / 'evaluation'))

from load_data import CHARS, CHARS_DICT, UnifiedManifestDataset
from train_LPRNet import forward_family_logits, collate_fn
from eval_lpr_detailed import decode_logits, greedy_decode_logits
from LPRNet_multihead import build_lprnet_multihead

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

OCR_PARAMS = dict(ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
                  ocr_preproc='none', ocr_channel_order='bgr', ocr_quad_pad_ratio=0.0)

MODELS = {
    'old_green': (ROOT / 'experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth', 'expD'),
    'v3_best': (ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/best_LPRNet_model.pth', 'expD'),
    'v4b_clean': (ROOT / 'experiments/green_ccpd2019_tilt_db_challenge_cvreplace_v4b_clean_20260509/best_LPRNet_model.pth', 'expD'),
    'v4c_A': (ROOT / 'experiments/green_ccpd2019_cvr_v4c_probe_A_wan15_20260509/best_LPRNet_model.pth', 'expD'),
}

MANIFESTS = {
    'cvr_val': ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv',
    'green_simple': ROOT / 'manifests_rebased/curriculum_gray3/test_green_simple.csv',
    'green_hard': ROOT / 'manifests_rebased/curriculum_gray3/test_green_hard.csv',
    'green_val': ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv',
}

DECODE_MODES = ['greedy', 'green_ctc_beam', 'family_aware_beam']

PROVS = set('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
def get_prov(t): return t[0] if t and t[0] in PROVS else '?'

def eval_decode_mode(net, manifest_path, decode_mode, beam_size=30, beam_topk=15, max_samples=500):
    ds = UnifiedManifestDataset(str(manifest_path), [94,24], 8, split_filter='test',
                                 dataset_root=str(ROOT), **OCR_PARAMS)
    if len(ds) == 0: return None
    # Use subset for beam modes (slow), full for greedy
    if decode_mode != 'greedy' and max_samples > 0 and len(ds) > max_samples:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(max_samples)))
    # fix labels path for Subset
    from torch.utils.data import DataLoader
    ld = DataLoader(ds, batch_size=120, shuffle=False, num_workers=4, collate_fn=collate_fn)
    total = exact = char_c = char_t = short_c = long_c = len_m = 0
    prov_c = prov_t = 0
    wan_exact = wan_total = 0
    nonwan_exact = nonwan_total = 0
    greedy_wrong_beam_right = 0  # only for beam modes
    
    with torch.no_grad():
        for images, labels, lengths, families in ld:
            images = images.to(device)
            prebs = forward_family_logits(net, images, sample_families=list(families)).detach().cpu().numpy()
            decoded = decode_logits(prebs, decode_mode, beam_size, beam_topk, sample_families=list(families))
            off = 0
            for bi in range(len(decoded)):
                pred_ids = decoded[bi]
                gt_len = int(lengths[bi])
                gt_ids = [int(labels[off + i]) for i in range(gt_len)]
                gt_text = ''.join(CHARS[i] for i in gt_ids if 0 <= i < len(CHARS))
                pred_text = ''.join(CHARS[i] for i in pred_ids if 0 <= i < len(CHARS))
                off += gt_len
                total += 1
                if pred_text == gt_text: exact += 1
                for p, g in zip(pred_text, gt_text): char_c += (p == g)
                char_t += len(gt_text)
                if len(pred_text) < len(gt_text): short_c += 1
                elif len(pred_text) > len(gt_text): long_c += 1
                else: len_m += 1
                if gt_text and gt_text[0] in PROVS:
                    prov_t += 1
                    if pred_text and pred_text[0] == gt_text[0]: prov_c += 1
                    if gt_text[0] == '皖':
                        wan_total += 1
                        if pred_text == gt_text: wan_exact += 1
                    else:
                        nonwan_total += 1
                        if pred_text == gt_text: nonwan_exact += 1
    r = {'n': total, 'exact': exact/total, 'char': char_c/max(char_t,1),
         'short': short_c/max(total,1), 'long': long_c/max(total,1), 'lenmatch': len_m/max(total,1),
         'prov1st': prov_c/max(prov_t,1),
         'wan_exact': wan_exact/max(wan_total,1), 'nonwan_exact': nonwan_exact/max(nonwan_total,1)}
    return r

results = {}
for mname, (mpath, head_type) in MODELS.items():
    if not mpath.exists(): continue
    print(f"\n=== {mname} ===", flush=True)
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS),
                                  dropout_rate=0.5, enhanced_green_head=head_type, pos0_head_cols=0)
    net.load_state_dict(torch.load(str(mpath), map_location='cpu'), strict=False)
    net.to(device); net.eval()
    results[mname] = {}
    for ename, epath in MANIFESTS.items():
        results[mname][ename] = {}
        for dm in DECODE_MODES:
            r = eval_decode_mode(net, epath, dm)
            if r: results[mname][ename][dm] = r
    del net; torch.cuda.empty_cache()

# Print comparison tables
for ename in MANIFESTS:
    print(f"\n--- {ename} ---")
    print(f"{'Model':15s} {'Mode':20s} {'Exact':>8s} {'Prov1st':>8s} {'Char':>8s} {'Short':>7s} {'LenM':>6s} {'gv_wan':>8s} {'gv_nwan':>8s}")
    print('-' * 90)
    for mname in MODELS:
        if mname not in results: continue
        if ename not in results[mname]: continue
        for dm in DECODE_MODES:
            r = results[mname][ename].get(dm)
            if r:
                gv_w = r.get('wan_exact', 0) * 100
                gv_nw = r.get('nonwan_exact', 0) * 100
                print(f"{mname:15s} {dm:20s} {r['exact']*100:7.2f}% {r['prov1st']*100:7.2f}% {r['char']*100:7.2f}% "
                      f"{r['short']*100:6.2f}% {r['lenmatch']*100:5.2f}% {gv_w:7.2f}% {gv_nw:7.2f}%")

# Greedy-wrong-beam-right analysis
print(f"\n\n--- Greedy-Wrong/Beam-Right Analysis ---")
for ename in ['cvr_val']:
    for mname in ['v3_best', 'v4b_clean', 'v4c_A']:
        if mname not in results or ename not in results[mname]: continue
        gr = results[mname][ename].get('greedy', {})
        br = results[mname][ename].get('family_aware_beam', {})
        if gr and br:
            diff = (br['exact'] - gr['exact']) * 100
            print(f"  {mname:15s} on {ename}: greedy={gr['exact']*100:.1f}% beam={br['exact']*100:.1f}% gap={diff:+.1f}pp")

out = ROOT / 'experiments' / 'green_ccpd2019_eval_protocol_audit_20260508' / 'decode_ab_comparison.json'
json.dump(results, open(out, 'w'), ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")
