#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import csv
from pathlib import Path

def warp_quad_to_rect_debug(image, pts, dst_w=None, dst_h=None):
    """透视变换"""
    img_h, img_w = image.shape[:2]
    quad = np.asarray(pts, dtype=np.float32)
    
    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])
    
    if dst_w is None:
        dst_w = int(max(width_top, width_bottom) + 0.5)
    if dst_h is None:
        dst_h = int(max(height_left, height_right) + 0.5)
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    
    dst = np.array([
        [0.0, 0.0],
        [dst_w - 1.0, 0.0],
        [dst_w - 1.0, dst_h - 1.0],
        [0.0, dst_h - 1.0],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        image, M, (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)
    )
    return warped, (dst_w, dst_h)

def resize_bgr_letterbox_debug(src, dst_w, dst_h, pad_value=0):
    """letterbox resize"""
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
    
    resized = cv2.resize(src, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    out[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized
    
    return out, scale, (scaled_w, scaled_h), (off_x, off_y)

def create_side_by_side(original_path, processed_img, text, save_path, quad, warped_size):
    """创建三格对比图"""
    orig = cv2.imread(str(original_path))
    if orig is None:
        return False
    
    # 原图画上 quad
    orig_with_quad = orig.copy()
    quad_np = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(orig_with_quad, [quad_np], True, (0, 255, 0), 2)
    
    # 透视变换
    warped, _ = warp_quad_to_rect_debug(orig, quad)
    
    # letterbox
    final_input, scale, scaled_size, offset = resize_bgr_letterbox_debug(warped, 94, 24, pad_value=0)
    
    # 放大以便观察
    scale_factor = 4
    orig_large = cv2.resize(orig_with_quad, (orig.shape[1] * scale_factor // 2, orig.shape[0] * scale_factor // 2))
    warped_large = cv2.resize(warped, (warped.shape[1] * 3, warped.shape[0] * 3))
    final_large = cv2.resize(final_input, (94 * 8, 24 * 8))
    
    # 创建画布
    margin = 15
    text_h = 60
    max_h = max(orig_large.shape[0], warped_large.shape[0], final_large.shape[0])
    total_w = orig_large.shape[1] + warped_large.shape[1] + final_large.shape[1] + margin * 4
    total_h = max_h + text_h + margin * 2
    
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    # 放置图像
    y_offset = text_h + margin
    x_positions = [margin, margin * 2 + orig_large.shape[1], margin * 3 + orig_large.shape[1] + warped_large.shape[1]]
    
    canvas[y_offset:y_offset+orig_large.shape[0], x_positions[0]:x_positions[0]+orig_large.shape[1]] = orig_large
    canvas[y_offset:y_offset+warped_large.shape[0], x_positions[1]:x_positions[1]+warped_large.shape[1]] = warped_large
    canvas[y_offset:y_offset+final_large.shape[0], x_positions[2]:x_positions[2]+final_large.shape[1]] = final_large
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (0, 0, 0)
    thickness = 2
    
    # 计算宽高比
    warped_w, warped_h = warped_size
    aspect_ratio = warped_w / warped_h if warped_h > 0 else 0
    
    cv2.putText(canvas, f"1. Original + Quad", (x_positions[0], 25), font, font_scale, color, thickness)
    cv2.putText(canvas, f"{text}", (x_positions[0], 50), font, 0.5, color, 1)
    
    cv2.putText(canvas, f"2. After WarpPerspective", (x_positions[1], 25), font, font_scale, color, thickness)
    cv2.putText(canvas, f"{warped_w:.0f}x{warped_h:.0f}, ratio={aspect_ratio:.2f}", (x_positions[1], 50), font, 0.5, color, 1)
    
    cv2.putText(canvas, f"3. Letterbox 94x24", (x_positions[2], 25), font, font_scale, color, thickness)
    cv2.putText(canvas, f"scale={scale:.2f}, occ={scaled_size[0]/94:.2f}", (x_positions[2], 50), font, 0.5, color, 1)
    
    cv2.imwrite(str(save_path), canvas)
    return True

# 读取测试数据
tsv_path = '/home/wzzz/LPRNet/green_edgefit_v2_realistic_test/details/accepted.tsv'
base_path = Path('/home/wzzz/LPRNet/green_edgefit_v2_realistic_test')

with open(tsv_path, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

# 选择沪、湘、粤、浙的 harder 样本各2张
out_dir = Path('/home/wzzz/LPRNet/qa_samples_h34d_new')
out_dir.mkdir(exist_ok=True)

print("Generating QA samples for new edgefit realistic...\n")

for prov in ['沪', '湘', '粤', '浙']:
    samples = [r for r in rows if r['province'] == prov and r['difficulty'] == 'harder'][:2]
    
    print(f"=== {prov} ===")
    for i, r in enumerate(samples, 1):
        text = r['text']
        rel_path = r['rel_path']
        quad = json.loads(r['quad'])
        img_path = base_path / rel_path
        
        # 计算宽高比
        top = np.linalg.norm(np.array(quad[1]) - np.array(quad[0]))
        bottom = np.linalg.norm(np.array(quad[2]) - np.array(quad[3]))
        left = np.linalg.norm(np.array(quad[3]) - np.array(quad[0]))
        right = np.linalg.norm(np.array(quad[2]) - np.array(quad[1]))
        warped_w = max(top, bottom)
        warped_h = max(left, right)
        aspect_ratio = warped_w / warped_h if warped_h > 0 else 0
        
        save_path = out_dir / f"{prov}_harder_{i}_{text}.jpg"
        success = create_side_by_side(img_path, None, text, save_path, quad, (warped_w, warped_h))
        
        if success:
            print(f"  {text}: ratio={aspect_ratio:.2f} -> Saved")
        else:
            print(f"  {text}: Failed to load image")
    print()

print(f"All samples saved to: {out_dir}")
