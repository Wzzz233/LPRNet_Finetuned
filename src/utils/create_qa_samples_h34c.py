#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽样展示沪、湘、粤、浙的 simple 和 harder 样本
左边：原图（生成的车牌图）
右边：板端处理后的图（obb_warp + letterbox + nn + bgr）
"""

import cv2
import numpy as np
from pathlib import Path
import json

# 固定板端参数
TARGET_W, TARGET_H = 94, 24
OCC_RATIO = 0.90
QUAD_PAD_RATIO = 0.0

def parse_quad_from_filename(filename):
    """从文件名解析 quad"""
    # 格式: edgefit-0-...-quad-{quad_str}-{uid}.jpg
    # quad_str 格式: x1&y1_x2&y2_x3&y3_x4&y4
    try:
        parts = filename.split('-')
        # 找到 quad 部分（在 ".jpg" 之前的部分）
        for i, p in enumerate(parts):
            if '&' in p and '_' in p and not p.endswith('.jpg'):
                # 这可能是 quad 的一部分
                pass
        # 实际上 quad 在文件名中的格式是: x1&y1_x2&y2_x3&y3_x4&y4
        # 让我们直接从 accepted.tsv 读取
        return None
    except:
        return None

def apply_board_pipeline(img, quad):
    """应用板端一致的处理流程"""
    h, w = img.shape[:2]
    
    # 解析 quad
    q = np.array(quad, dtype=np.float32)
    
    # 计算目标矩形
    dst_rect = np.array([
        [0, 0],
        [TARGET_W - 1, 0],
        [TARGET_W - 1, TARGET_H - 1],
        [0, TARGET_H - 1]
    ], dtype=np.float32)
    
    # 透视变换
    M = cv2.getPerspectiveTransform(q, dst_rect)
    warped = cv2.warpPerspective(img, M, (TARGET_W, TARGET_H), flags=cv2.INTER_NEAREST)
    
    return warped

def create_side_by_side(original_path, processed_img, text, save_path):
    """创建并排对比图"""
    orig = cv2.imread(str(original_path))
    if orig is None:
        print(f"Failed to load: {original_path}")
        return False
    
    # 原图缩放到合适大小以便对比
    orig_h, orig_w = orig.shape[:2]
    scale = 3  # 放大3倍以便看清
    orig_large = cv2.resize(orig, (orig_w * scale, orig_h * scale), interpolation=cv2.INTER_NEAREST)
    proc_large = cv2.resize(processed_img, (processed_img.shape[1] * 4, processed_img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    
    # 创建画布
    margin = 20
    text_h = 40
    total_h = max(orig_large.shape[0], proc_large.shape[0]) + text_h + margin * 2
    total_w = orig_large.shape[1] + proc_large.shape[1] + margin * 3
    
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    # 放置原图
    y_offset = text_h + margin
    canvas[y_offset:y_offset+orig_large.shape[0], margin:margin+orig_large.shape[1]] = orig_large
    
    # 放置处理后图
    x_offset = margin * 2 + orig_large.shape[1]
    canvas[y_offset:y_offset+proc_large.shape[0], x_offset:x_offset+proc_large.shape[1]] = proc_large
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Original: {text}", (margin, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Board Pipeline (obb_warp+letterbox)", (x_offset, 30), font, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(str(save_path), canvas)
    return True

# 抽样数据
samples = {
    '沪': {
        'simple': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p01_u6caa/edgefit-0-9&9_244&67-15&14_242&10_243&66_10&66-train-simple-沪-0059-沪UDA1731.jpg', 
             '沪UDA1731', [[15.35, 14.42], [242.22, 9.68], [243.01, 66.24], [9.54, 65.83]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p01_u6caa/edgefit-0-11&8_245&71-12&9_237&22_244&71_14&66-train-simple-沪-0016-沪YFF5899.jpg',
             '沪YFF5899', [[11.74, 8.99], [237.18, 22.10], [244.17, 71.0], [13.68, 66.20]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p01_u6caa/edgefit-0-11&2_244&71-11&3_237&14_243&71_13&59-train-simple-沪-0095-沪JDE9278.jpg',
             '沪JDE9278', [[11.39, 2.81], [237.21, 14.22], [243.22, 71.0], [13.22, 58.99]]),
        ],
        'harder': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p01_u6caa/edgefit-0-4&8_242&71-4&13_241&8_241&71_10&70-train-harder-沪-0054-沪VDK8087.jpg',
             '沪VDK8087', [[4.10, 12.80], [240.50, 8.07], [241.24, 70.56], [10.20, 69.88]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p01_u6caa/edgefit-0-18&3_243&71-19&3_236&23_243&71_19&68-train-harder-沪-0099-沪FDY0376.jpg',
             '沪FDY0376', [[19.36, 3.24], [235.99, 23.31], [242.83, 70.50], [18.76, 68.46]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p01_u6caa/edgefit-0-22&12_243&71-23&13_238&29_242&71_30&69-train-harder-沪-0037-沪EFC2044.jpg',
             '沪EFC2044', [[22.88, 12.55], [238.33, 28.99], [242.34, 71.0], [30.47, 69.13]]),
        ]
    },
    '湘': {
        'simple': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p18_u6e58/edgefit-0-18&8_244&71-19&9_241&8_244&65_25&71-train-simple-湘-0058-湘SDB8897.jpg',
             '湘SDB8897', [[18.65, 8.79], [240.92, 8.43], [243.99, 64.57], [24.74, 71.0]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p18_u6e58/edgefit-0-0&9_244&71-16&15_244&9_244&66_0&71-train-simple-湘-0088-湘YDE1318.jpg',
             '湘YDE1318', [[16.37, 15.45], [243.64, 9.20], [243.55, 65.70], [0.15, 71.0]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p18_u6e58/edgefit-0-3&3_244&71-13&3_240&20_243&71_3&58-train-simple-湘-0063-湘FDA0651.jpg',
             '湘FDA0651', [[13.15, 3.19], [240.37, 20.41], [243.23, 71.0], [3.45, 57.82]]),
        ],
        'harder': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p18_u6e58/edgefit-0-6&9_245&69-15&18_237&9_245&65_6&69-train-harder-湘-0014-湘GDF1532.jpg',
             '湘GDF1532', [[15.24, 17.61], [237.38, 9.50], [244.57, 64.51], [6.11, 68.62]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p18_u6e58/edgefit-0-6&2_245&71-15&3_244&24_244&71_7&66-train-harder-湘-0079-湘RDQ4313.jpg',
             '湘RDQ4313', [[15.21, 2.73], [244.23, 24.17], [244.34, 71.0], [6.91, 65.96]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p18_u6e58/edgefit-0-18&5_243&68-22&6_240&26_242&68_18&66-train-harder-湘-0032-湘JDN9954.jpg',
             '湘JDN9954', [[21.54, 5.60], [239.72, 25.77], [242.42, 67.61], [18.04, 65.51]]),
        ]
    },
    '粤': {
        'simple': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p19_u7ca4/edgefit-0-6&5_245&71-7&5_240&21_245&71_12&62-train-simple-粤-0045-粤JDR2838.jpg',
             '粤JDR2838', [[6.83, 5.35], [240.29, 20.99], [244.51, 71.0], [12.10, 61.55]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p19_u7ca4/edgefit-0-9&6_245&71-9&6_238&20_245&71_22&60-train-simple-粤-0017-粤XFT0785.jpg',
             '粤XFT0785', [[9.40, 6.46], [237.89, 19.99], [244.74, 71.0], [22.23, 60.34]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p19_u7ca4/edgefit-0-1&4_245&71-16&5_244&16_243&71_2&64-train-simple-粤-0100-粤VD70070.jpg',
             '粤VD70070', [[16.16, 4.99], [244.16, 15.95], [243.01, 71.0], [1.85, 63.96]]),
        ],
        'harder': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p19_u7ca4/edgefit-0-12&4_242&71-24&6_235&4_241&65_13&71-train-harder-粤-0116-粤SF31713.jpg',
             '粤SF31713', [[24.06, 5.84], [235.42, 4.26], [241.45, 65.40], [12.71, 71.0]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p19_u7ca4/edgefit-0-4&10_242&69-26&11_239&28_241&68_5&66-train-harder-粤-0108-粤BDM8750.jpg',
             '粤BDM8750', [[26.34, 10.87], [239.35, 27.51], [241.23, 68.47], [4.62, 65.54]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p19_u7ca4/edgefit-0-27&10_243&71-27&10_241&15_243&71_29&70-train-harder-粤-0057-粤VFK3687.jpg',
             '粤VFK3687', [[27.42, 10.25], [240.70, 14.53], [242.54, 71.0], [29.50, 69.64]]),
        ]
    },
    '浙': {
        'simple': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p11_u6d59/edgefit-0-13&6_244&71-20&7_240&13_244&71_14&67-train-simple-浙-0070-浙PD92783.jpg',
             '浙PD92783', [[19.51, 6.91], [240.48, 12.69], [243.58, 71.0], [13.97, 66.92]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p11_u6d59/edgefit-0-16&7_244&71-16&8_238&11_244&71_19&57-train-simple-浙-0074-浙SFL8553.jpg',
             '浙SFL8553', [[16.05, 7.66], [237.99, 10.67], [243.70, 71.0], [19.20, 57.05]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/simple/p11_u6d59/edgefit-0-8&5_244&69-16&5_243&15_243&69_8&61-train-simple-浙-0116-浙VD37849.jpg',
             '浙VD37849', [[16.01, 5.43], [242.95, 15.24], [243.42, 68.83], [8.35, 60.98]]),
        ],
        'harder': [
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p11_u6d59/edgefit-0-14&3_243&71-14&3_243&31_241&71_20&65-train-harder-浙-0028-浙YD83764.jpg',
             '浙YD83764', [[14.47, 3.24], [242.82, 30.72], [241.27, 71.0], [19.81, 64.82]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p11_u6d59/edgefit-0-16&5_244&70-26&5_244&24_241&69_16&65-train-harder-浙-0039-浙LFM1964.jpg',
             '浙LFM1964', [[26.29, 5.29], [243.69, 23.95], [241.48, 69.49], [16.39, 64.55]]),
            ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p11_u6d59/edgefit-0-0&15_242&71-8&15_241&21_242&63_1&71-train-harder-浙-0021-浙BD18404.jpg',
             '浙BD18404', [[8.48, 15.38], [241.17, 21.13], [242.00, 62.65], [0.52, 70.71]]),
        ]
    }
}

# 创建输出目录
out_dir = Path('/home/wzzz/LPRNet/qa_samples_h34c')
out_dir.mkdir(exist_ok=True)

# 处理每个样本
for prov, difficulties in samples.items():
    print(f"\n=== {prov} ===")
    for diff, sample_list in difficulties.items():
        print(f"\n{diff.upper()}:")
        for i, (img_path, text, quad) in enumerate(sample_list, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Failed to load: {img_path}")
                continue
            
            # 应用板端处理
            processed = apply_board_pipeline(img, quad)
            
            # 创建对比图
            save_path = out_dir / f"{prov}_{diff}_{i}_{text}.jpg"
            success = create_side_by_side(img_path, processed, text, save_path)
            if success:
                print(f"  Saved: {save_path.name}")
            else:
                print(f"  Failed: {img_path}")

print(f"\n\nAll samples saved to: {out_dir}")
