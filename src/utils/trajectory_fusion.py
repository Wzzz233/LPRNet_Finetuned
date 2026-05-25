#!/usr/bin/env python3
"""Trajectory-level OCR fusion for board dump evaluation.
Callable as module: from trajectory_fusion import fuse_trajectory, analyze_frame
Also runnable: python3 trajectory_fusion.py --weights <path> --dump_dir <dir> --gt <text>

Outputs per-trajectory:
  - Best single-frame result
  - Fused trajectory result
  - Province reliability status
  - Suffix stability
  - Evidence frame count"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'src' / 'training'), str(_SRC_DIR / 'src' / 'utils')):
    if _p not in sys.path: sys.path.insert(0, _p)

from src.LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from src.load_data import CHARS
from src.training.train_LPRNet import forward_family_logits
from src.evaluation.test_LPRNet import greedy_decode_logits

ALL_PROVS = list('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')

def load_model(weights_path, device='cuda:0'):
    state = torch.load(weights_path, map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device).eval()
    return net

def analyze_frame(model, img, device='cuda:0'):
    """Run model on a single 94x24 ocrin frame. Returns prediction + province logits."""
    x = img.astype(np.float32); x -= 127.5; x *= 0.0078125
    xt = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_dict = forward_family_logits(model, xt, return_all=True)
    green8 = logits_dict['green8']
    
    prebs = green8.detach().cpu().numpy()
    pred = ''.join(CHARS[int(c)] for c in greedy_decode_logits(prebs)[0])
    
    logits_t0 = green8[0, :, 0]
    prov_probs = torch.softmax(logits_t0[:31], dim=0)
    blank_prob = torch.softmax(logits_t0, dim=0)[len(CHARS)-1].item()
    top5_vals, top5_idx = torch.topk(prov_probs, 5)
    top5 = [(CHARS[i.item()], v.item()) for v, i in zip(top5_vals, top5_idx)]
    
    return {
        'pred': pred,
        'top5_prov': top5,
        'prov_conf': top5[0][1],
        'blank_prob': blank_prob,
        'brightness': float(img.mean()),
    }

def fuse_trajectory(frames_data, gt_prov=None):
    """Quality-weighted temporal fusion.
    
    Key principles:
    - Province evidence dominates weight; suffix is a gate, not a scorer.
    - Frames with strong province logits but CTC-swallowed province still count.
    - Segments scored by quality, not length; short high-quality window can win.
    - If province has visual evidence across any contiguous high-quality window, use it.
    """
    # ── Frame-level features ──
    for f in frames_data:
        pred = f['pred']
        f['suffix'] = pred[1:] if len(pred) >= 2 else ''
        f['has_province'] = len(pred) > 0 and pred[0] in ALL_PROVS
        f['prov_top1'] = f['top5_prov'][0][0] if f['top5_prov'] else ''
        f['prov_top3'] = [p for p, _ in f['top5_prov'][:3]]
        f['prov_top1_conf'] = f['top5_prov'][0][1] if f['top5_prov'] else 0
        # Province evidence strength: product of top1 confidence and brightness (capped)
        f['prov_evidence'] = f['prov_top1_conf'] * min(1.0, f['brightness'] / 100.0)
    
    # ── Suffix across trajectory (overall best suffix) ──
    suffix_counts = Counter(f['suffix'] for f in frames_data if f['suffix'])
    best_suffix = suffix_counts.most_common(1)[0][0] if suffix_counts else ''
    suffix_stable = any(c / len(frames_data) > 0.15 for c in suffix_counts.values())
    
    # ── GT check ──
    gt_in_top5 = False
    if gt_prov:
        for f in frames_data:
            if any(gt_prov == p for p, _ in f['top5_prov']):
                gt_in_top5 = True
                break
    
    # ── Find best contiguous segment by province evidence ──
    # Score every segment of length 4+ by: 
    #   avg province evidence for the dominant province in the segment
    #   × sqrt(segment length) (mild bonus for longer, but not dominant)
    #   × bonus if segment's dominant province matches the suffix's most common province
    best_seg = None
    best_seg_prov = None
    best_seg_score = -1
    
    for seg_len in range(min(len(frames_data), 30), 3, -1):
        for start in range(len(frames_data) - seg_len + 1):
            seg = frames_data[start:start+seg_len]
            
            # Find dominant province in this segment
            prov_evidence = {}
            for f in seg:
                p = f['prov_top1']
                prov_evidence[p] = prov_evidence.get(p, 0) + f['prov_evidence']
            
            dom_prov = max(prov_evidence, key=prov_evidence.get)
            dom_evidence = prov_evidence[dom_prov]
            
            # Also check how many frames have this province in top-3
            frames_with_prov_in_top3 = sum(1 for f in seg if dom_prov in f['prov_top3'])
            top3_ratio = frames_with_prov_in_top3 / seg_len
            
            # Score = average province evidence × top3 ratio × sqrt(length)
            avg_evidence = dom_evidence / seg_len
            seg_score = avg_evidence * top3_ratio * (seg_len ** 0.3)
            
            # Bonus: does this segment's province dominate consistently?
            # (penalize segments where province jumps around)
            top1_ratio = max(Counter(f['prov_top1'] for f in seg).values()) / seg_len
            seg_score *= top1_ratio
            
            if seg_score > best_seg_score:
                best_seg_score = seg_score
                best_seg = (start, start + seg_len - 1)
                best_seg_prov = dom_prov
    
    # ── Province decision ──
    # If GT never appears in any frame's top-5 → unreliable
    if gt_prov and not gt_in_top5:
        province_status = 'unreliable'
        best_prov = '?'
        fused_text = f"?{best_suffix}"
    elif best_seg_prov and best_seg_score > 0.05:
        # Only accept segment if its evidence is meaningful
        province_status = 'temporal_segment'
        fused_text = f"{best_seg_prov}{best_suffix}"
        
        # If fused result equals GT, mark as recovered
        if gt_prov and fused_text[0] == gt_prov:
            province_status = 'recovered'
    else:
        province_status = 'weak'
        fused_text = f"?{best_suffix}"
    
    # ── Best single frame ──
    best_single_f = max(frames_data, key=lambda f: f['prov_evidence']) if frames_data else {}
    
    return {
        'fused_text': fused_text,
        'best_single_frame': best_single_f.get('pred', ''),
        'best_suffix': best_suffix,
        'suffix_stability': max(suffix_counts.values()) / len(frames_data) if suffix_counts else 0,
        'best_province': best_seg_prov if province_status not in ('unreliable', 'weak') else '?',
        'province_status': province_status,
        'gt_in_top5_ever': gt_in_top5,
        'best_segment': f"frames {best_seg[0]}-{best_seg[1]}" if best_seg else None,
        'best_segment_score': best_seg_score,
        'total_frames': len(frames_data),
    }
    
    # Best single frame
    best_single = max(frames_data, key=lambda f: f['prov_conf'] * f['brightness']) if frames_data else {}
    
    return {
        'fused_text': f"{best_prov}{best_suffix}" if province_status != 'unreliable' else f"?{best_suffix}",
        'best_single_frame': best_single.get('pred', ''),
        'best_suffix': best_suffix,
        'suffix_stability': suffix_conf,
        'best_province': best_prov if province_status != 'unreliable' else '?',
        'province_status': province_status,
        'gt_in_top5_ever': gt_in_top5,
        'best_segment': f"frames {best_seg[0]}-{best_seg[1]}" if best_seg else None,
        'total_frames': len(frames_data),
    }

def evaluate_dump(model, dump_dir, gt_full=None, device='cuda:0'):
    """Run trajectory fusion on an ocrin dump directory."""
    d = Path(dump_dir)
    files = sorted(d.glob('ocrin_*.ppm'))
    if not files:
        return {'error': f'No ocrin files in {dump_dir}'}
    
    frames = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None: continue
        frames.append(analyze_frame(model, img, device))
    
    result = fuse_trajectory(frames, gt_prov=gt_full[0] if gt_full else None)
    result['gt'] = gt_full or ''
    result['dump_dir'] = str(dump_dir)
    result['frame_count'] = len(frames)
    
    # Top-1 province distribution
    prov_dist = Counter()
    for f in frames:
        if f['top5_prov']:
            prov_dist[f['top5_prov'][0][0]] += 1
    result['province_distribution'] = dict(prov_dist.most_common())
    
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Trajectory fusion on board dump')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--gt', default='')
    ap.add_argument('--out_json', default='')
    args = ap.parse_args()
    
    model = load_model(args.weights)
    result = evaluate_dump(model, args.dump_dir, args.gt)
    
    print(f"\n{'='*50}")
    print(f"Dump: {result.get('dump_dir', '?')}")
    print(f"GT:   {args.gt}")
    print(f"{'='*50}")
    print(f"Fused text:    {result.get('fused_text','?')}")
    print(f"Best single:   {result.get('best_single_frame','?')}")
    print(f"Province:      {result.get('best_province','?')} ({result.get('province_status','?')})")
    print(f"Suffix:        {result.get('best_suffix','?')} (stable {result.get('suffix_stability',0)*100:.0f}%)")
    print(f"GT in top5:    {result.get('gt_in_top5_ever',False)}")
    print(f"Best segment:  {result.get('best_segment','N/A')}")
    print(f"Frames:        {result.get('frame_count',0)}")
    print(f"Province dist: {result.get('province_distribution',{})}")
    
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.out_json}")

