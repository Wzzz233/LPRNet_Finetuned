#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成 QA 样本，准确展示：
1. 原图 + quad 框
2. 透视变换后的图像（warpPerspective 直接输出）
3. 经过 letterbox resize 后的最终输入（含黑边）
"""

import cv2
import numpy as np
import json
from pathlib import Path

def warp_quad_to_rect_debug(image, pts, dst_w=None, dst_h=None, pad_ratio=0.0):
    """透视变换，返回中间过程用于调试"""
    # 简单的 quad 裁剪（简化版）
    img_h, img_w = image.shape[:2]
    quad = np.asarray(pts, dtype=np.float32)
    
    # 计算边长
    def distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    width_top = distance(quad[0], quad[1])
    width_bottom = distance(quad[3], quad[2])
    height_left = distance(quad[0], quad[3])
    height_right = distance(quad[1], quad[2])
    
    if dst_w is None:
        dst_w = int(max(width_top, width_bottom) + 0.5)
    if dst_h is None:
        dst_h = int(max(height_left, height_right) + 0.5)
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    
    # 目标矩形
    dst = np.array([
        [0.0, 0.0],
        [dst_w - 1.0, 0.0],
        [dst_w - 1.0, dst_h - 1.0],
        [0.0, dst_h - 1.0],
    ], dtype=np.float32)
    
    # 透视变换
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)  # 用灰色边界以便观察
    )
    
    return warped, (dst_w, dst_h), matrix

def resize_bgr_letterbox_debug(src, dst_w, dst_h, pad_value=0):
    """letterbox resize，返回中间过程"""
    src_h, src_w = src.shape[:2]
    out = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    scale = min(sx, sy)
    
    scaled_w = int(src_w * scale + 0.5)
    scaled_h = int(src_h * scale + 0.5)
    scaled_w = max(1, min(dst_w, scaled_w))
    scaled_h = max(1, min(dst_h, scaled_h))
    
    off_x = (dst_w - scaled_w) // 2
    off_y = (dst_h - scaled_h) // 2
    
    # 使用最近邻插值
    resized = cv2.resize(src, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    out[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized
    
    return out, scale, (scaled_w, scaled_h), (off_x, off_y)

# 抽样数据（使用实际的 quad 值）
samples = [
    # (图像路径, 文本, quad)
    ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p01_u6caa/edgefit-0-4&8_242&71-4&13_241&8_241&71_10&70-train-harder-沪-0054-沪VDK8087.jpg', 
     '沪VDK8087 (harder)', 
     [[4.10, 12.80], [240.50, 8.07], [241.24, 70.56], [10.20, 69.88]]),
    ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p18_u6e58/edgefit-0-6&9_245&69-15&18_237&9_245&65_6&69-train-harder-湘-0014-湘GDF1532.jpg',
     '湘GDF1532 (harder)',
     [[15.24, 17.61], [237.38, 9.50], [244.57, 64.51], [6.11, 68.62]]),
    ('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore/images/train/harder/p19_u7ca4/edgefit-0-12&4_242&71-24&6_235&4_241&65_13&71-train-harder-粤-0116-粤SF31713.jpg',
     '粤SF31713 (harder)',
     [[24.06, 5.84], [235.42, 4.26], [241.45, 65.40], [12.71, 71.0]]),
]

# 创建输出目录
out_dir = Path('/home/wzzz/LPRNet/qa_samples_h34c_detailed')
out_dir.mkdir(exist_ok=True)

# 目标尺寸
TARGET_W, TARGET_H = 94, 24

for img_path, text, quad in samples:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue
    
    # 1. 原图画上 quad
    orig_with_quad = img.copy()
    quad_np = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(orig_with_quad, [quad_np], True, (0, 255, 0), 2)
    
    # 2. 透视变换
    warped, warped_size, matrix = warp_quad_to_rect_debug(img, quad)
    
    # 3. letterbox resize 到 94x24
    final_input, scale, scaled_size, offset = resize_bgr_letterbox_debug(warped, TARGET_W, TARGET_H, pad_value=0)
    
    # 创建三格对比图
    margin = 20
    text_h = 50
    
    # 放大以便观察
    scale_factor = 5
    orig_large = cv2.resize(orig_with_quad, (img.shape[1] * scale_factor // 2, img.shape[0] * scale_factor // 2), interpolation=cv2.INTER_NEAREST)
    warped_large = cv2.resize(warped, (warped.shape[1] * scale_factor, warped.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)
    final_large = cv2.resize(final_input, (TARGET_W * 8, TARGET_H * 8), interpolation=cv2.INTER_NEAREST)
    
    # 计算画布尺寸
    max_h = max(orig_large.shape[0], warped_large.shape[0], final_large.shape[0])
    total_w = orig_large.shape[1] + warped_large.shape[1] + final_large.shape[1] + margin * 4
    total_h = max_h + text_h + margin * 2
    
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    # 放置图像
    y_offset = text_h + margin
    x_positions = [margin, margin * 2 + orig_large.shape[1], margin * 3 + orig_large.shape[1] + warped_large.shape[1]]
    
    # 原图
    canvas[y_offset:y_offset+orig_large.shape[0], x_positions[0]:x_positions[0]+orig_large.shape[1]] = orig_large
    # 透视变换后
    canvas[y_offset:y_offset+warped_large.shape[0], x_positions[1]:x_positions[1]+warped_large.shape[1]] = warped_large
    # 最终输入
    canvas[y_offset:y_offset+final_large.shape[0], x_positions[2]:x_positions[2]+final_large.shape[1]] = final_large
    
    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)
    thickness = 2
    
    cv2.putText(canvas, f"1. Original + Quad: {text}", (x_positions[0], 30), font, font_scale, color, thickness)
    cv2.putText(canvas, f"2. After WarpPerspective: {warped_size[0]}x{warped_size[1]}", (x_positions[1], 30), font, font_scale, color, thickness)
    cv2.putText(canvas, f"3. After Letterbox (94x24): scale={scale:.2f}, occ={scaled_size[0]/TARGET_W:.2f}", (x_positions[2], 30), font, font_scale, color, thickness)
    
    # 添加详细信息到图像下方
    info_y = y_offset + max_h + 20
    info_text = f"Warped size: {warped_size} | Scaled size: {scaled_size} | Offset: {offset} | Final: 94x24"
    cv2.putText(canvas, info_text, (margin, info_y), font, 0.5, color, 1)
    
    # 保存
    save_name = text.replace(' ', '_').replace('(', '').replace(')', '') + '.jpg'
    save_path = out_dir / save_name
    cv2.imwrite(str(save_path), canvas)
    print(f"Saved: {save_path}")
    print(f"  Original: {img.shape}")
    print(f"  Warped: {warped.shape}")
    print(f"  Final input: {final_input.shape}, scale={scale:.2f}, occ_ratio={scaled_size[0]/TARGET_W:.2f}")
    print()

print(f"\nAll samples saved to: {out_dir}")
