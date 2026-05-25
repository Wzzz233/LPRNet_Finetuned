#!/usr/bin/env python3
"""Check if replacement images have reversed text."""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path('/home/wzzz/LPRNet/src')))

import cv2
import numpy as np
from load_data import prepare_board_ocr_input_from_quad_bgr888

img_path = '/home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/images/train/0241276041667-100_256-258&418_462&537-462&537_262&483_258&418_457&459-1_0_5_26_27_24_26_26-111-62_posev3_沪WDK4117.jpg'

# Get pose quad from source_log
with open('/home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3/source_log.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['generated_img'] == img_path:
            pq = r['pose_quad']
            break

img = cv2.imread(img_path)
pq_arr = np.array(pq, dtype=np.float32)

# Try both quad orders
for label, q in [('correct', pq_arr), ('reversed', pq_arr[::-1].copy())]:
    g3, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, q, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    out_path = f'/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/replace_vs_real_qa/quad_check_{label}.jpg'
    g3_2x = cv2.resize(g3, (188, 48))
    # Annotate
    cv2.putText(g3_2x, f'{label} order', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    cv2.imwrite(out_path, g3_2x)
    print(f'Wrote {out_path}')

# Also check a real CCPD2020 green for comparison
real_path = '/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/00334486715625-99_283-117&109_363&374-347&458_668&377_654&104_326&226-0_0_25_25_25_25_29_25-28-47.jpg'
quad = np.array([[347, 458], [668, 377], [654, 104], [326, 226]], dtype=np.float32)
real_img = cv2.imread(real_path)
g3_real, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
    real_img, quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
    preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
g3_real_2x = cv2.resize(g3_real, (188, 48))
out_real = '/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/replace_vs_real_qa/quad_check_real.jpg'
cv2.putText(g3_real_2x, 'REAL CCPD2020', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
cv2.imwrite(out_real, g3_real_2x)
print(f'Wrote {out_real}')
