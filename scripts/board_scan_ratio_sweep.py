#!/usr/bin/env python3
"""Board-centric checkpoint scan for A ratio sweep checkpoints.
Evaluates each checkpoint against pos_ocr_dump (seg_B = frames 11-40 for tilt)
and pos_ocr_dump_2 (static control).
"""
import csv, json, sys, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0, str(p))
from load_data import CHARS
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BLANK = len(CHARS) - 1

DUMP_DIRS = {
    'pos_ocr_dump': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump'),
    'pos_ocr_dump_2': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2'),
}

SEG_B_START = 10  # 0-indexed frame 10 = frame 11
SEG_B_END = 40    # 0-indexed frame 39 = frame 40

# Province characters for green plates
PROVINCE_CHARS = set('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁')

GREEN_PROV_MAP = {
    '京': '北京', '津': '天津', '沪': '上海', '渝': '重庆',
    '冀': '河北', '豫': '河南', '云': '云南', '辽': '辽宁',
    '黑': '黑龙江', '湘': '湖南', '皖': '安徽', '鲁': '山东',
    '新': '新疆', '苏': '江苏', '浙': '浙江', '赣': '江西',
    '鄂': '湖北', '桂': '广西', '甘': '甘肃', '晋': '山西',
    '蒙': '内蒙古', '陕': '陕西', '吉': '吉林', '闽': '福建',
    '贵': '贵州', '粤': '广东', '川': '四川', '青': '青海',
    '藏': '西藏', '琼': '海南', '宁': '宁夏',
}

def preprocess_bgr(img_rgb):
    """RGB → BGR matching training ocr_channel_order=bgr"""
    return img_rgb[:, :, ::-1].copy()

def normalize(x):
    x = x.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))
    return x[None, ...]

def greedy_decode(logits):
    labels = []
    prev = BLANK
    for t in range(logits.shape[1]):
        c = int(np.argmax(logits[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)

def compute_metrics(predictions, ground_truths):
    """Compute pp_exact, pp_char, len_err_rate from predictions and ground truths."""
    n = len(predictions)
    exact = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    
    total_chars = 0
    correct_chars = 0
    len_err = 0
    for p, g in zip(predictions, ground_truths):
        total_chars += len(g)
        correct_chars += sum(1 for a, b in zip(p, g) if a == b)
        if len(p) != len(g):
            len_err += 1
    
    pp_exact = (exact / n * 100) if n > 0 else 0.0
    pp_char = (correct_chars / total_chars * 100) if total_chars > 0 else 0.0
    len_err_rate = (len_err / n * 100) if n > 0 else 0.0
    mean_len = np.mean([len(p) for p in predictions])
    
    # Province confusion
    prov_confusion = Counter()
    for p, g in zip(predictions, ground_truths):
        if p != g and len(p) > 0 and len(g) > 0:
            pred_prov = p[0]
            gt_prov = g[0]
            if gt_prov in PROVINCE_CHARS and pred_prov != gt_prov:
                prov_confusion[pred_prov] += 1
    
    return {
        'n': n,
        'exact': exact,
        'pp_exact': round(pp_exact, 1),
        'pp_char': round(pp_char, 1),
        'mean_len': round(float(mean_len), 2),
        'len_err_rate': round(len_err_rate, 1),
        'prov_confusion': dict(prov_confusion.most_common(10)),
    }

def load_samples(dump_dir):
    """Load index.csv from dump directory."""
    index_path = dump_dir / 'index.csv'
    if not index_path.exists():
        print(f"  SKIP: no index.csv in {dump_dir}")
        return None
    
    samples = []
    with open(index_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ocrin_rel = row['ocr_input_path'].split('/')[-1]
            ocrin_path = dump_dir / ocrin_rel
            if ocrin_path.exists():
                samples.append({
                    'id': int(row['sample_id']),
                    'gt': row['app_text'].strip(),
                    'ocrin_path': str(ocrin_path),
                })
    return samples

def evaluate_checkpoint(checkpoint_path, dump_dir, samples, segment_name='full'):
    """Evaluate a single checkpoint on all or segmented samples."""
    if not checkpoint_path.exists():
        print(f"  SKIP: checkpoint not found at {checkpoint_path}")
        return None
    
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    net, cfg = build_lprnet_multihead_from_state_dict(
        state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(DEVICE)
    net.eval()
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for s in samples:
            img_pil = Image.open(s['ocrin_path']).convert('RGB')
            img_np = np.array(img_pil, dtype=np.uint8)
            img_gray3 = preprocess_bgr(img_np)
            x = normalize(img_gray3)
            images = torch.from_numpy(x).to(DEVICE)
            
            raw = net(images)
            logits = _select_family_logits_from_dict(
                raw, sample_families=['green8']
            ).detach().cpu().numpy()[0]
            
            pred = greedy_decode(logits)
            predictions.append(pred)
            ground_truths.append(s['gt'])
    
    return predictions, ground_truths

def main():
    exp_names = ['a_ratio_r10_20260510', 'a_ratio_r20_20260510', 'a_ratio_r35_20260510']
    ckpt_names = [
        'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
        'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth', 'LPRNet__iteration_6000.pth'
    ]
    
    for exp_name in exp_names:
        print(f"\n{'='*70}")
        print(f"  Scanning: {exp_name}")
        print(f"{'='*70}")
        
        exp_dir = ROOT / 'experiments' / exp_name
        result = {}
        
        # Load samples for both dumps
        dump_samples = {}
        for dump_key, dump_dir in DUMP_DIRS.items():
            samples = load_samples(dump_dir)
            if samples is None:
                continue
            
            # Sort by sample_id
            samples.sort(key=lambda s: s['id'])
            
            # Identify dump type
            # pos_ocr_dump: sample_ids 0-49, seg_B = 10-39 (frames 11-40)
            # pos_ocr_dump_2: all frames are static control
            dump_samples[dump_key] = samples
            print(f"  {dump_key}: {len(samples)} samples")
            
            # For pos_ocr_dump, define seg_B region
            if dump_key == 'pos_ocr_dump':
                # seg_B: sample_id from SEG_B_START to SEG_B_END-1 (frames 11-40)
                seg_B_samples = [s for s in samples if SEG_B_START <= s['id'] < SEG_B_END]
                print(f"    seg_B (frames 11-40): {len(seg_B_samples)} samples (ids {SEG_B_START}-{SEG_B_END-1})")
                dump_samples[f'{dump_key}_seg_B'] = seg_B_samples
        
        # Evaluate each checkpoint
        for ckpt_name in ckpt_names:
            ckpt_path = exp_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"  SKIP {ckpt_name}: not found")
                continue
            
            print(f"\n  --- {ckpt_name} ---")
            ckpt_result = {}
            
            for dump_key in list(dump_samples.keys()):
                samples = dump_samples[dump_key]
                if not samples:
                    continue
                
                result_key = dump_key  # e.g., 'pos_ocr_dump', 'pos_ocr_dump_seg_B', 'pos_ocr_dump_2'
                predictions, ground_truths = evaluate_checkpoint(ckpt_path, DUMP_DIRS.get(dump_key.replace('_seg_B', ''), DUMP_DIRS.get(dump_key)), samples)
                if predictions is None:
                    continue
                
                metrics = compute_metrics(predictions, ground_truths)
                ckpt_result[result_key] = {
                    'metrics': metrics,
                    'predictions': predictions,
                }
                
                if result_key.endswith('_seg_B'):
                    print(f"    seg_B_tilt: pp_char={metrics['pp_char']:.1f}% len_err={metrics['len_err_rate']:.1f}% pp_exact={metrics['pp_exact']:.1f}%")
                elif dump_key == 'pos_ocr_dump_2':
                    print(f"    static_control: pp_exact={metrics['pp_exact']:.1f}% pp_char={metrics['pp_char']:.1f}%")
                    # Province bias
                    bias = metrics.get('prov_confusion', {})
                    if bias:
                        top5 = list(bias.items())[:5]
                        print(f"    province_bias: {top5}")
                elif dump_key == 'pos_ocr_dump':
                    print(f"    full: pp_exact={metrics['pp_exact']:.1f}% pp_char={metrics['pp_char']:.1f}%")
            
            result[ckpt_name] = ckpt_result
        
        # Save
        out_path = ROOT / 'experiments' / 'mix_source_audit_20260510' / f'a_ratio_board_scan_{exp_name.replace("a_ratio_", "").replace("_20260510", "")}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

if __name__ == '__main__':
    main()
