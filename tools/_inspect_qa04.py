#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# Look at the original image and the quad
img_path = '/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/303747829861111111-90_95-243&477_557&597-557&595_244&597_243&489_548&477-0_0_3_24_27_29_30_33-67-125.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Cannot read {img_path}")
else:
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # The quad from manifest
    quad = np.array([
        [236.2, 487.0],
        [554.6, 492.3],
        [552.4, 590.9],
        [236.2, 597.6],
    ], dtype=np.float32)
    
    print(f"Quad: {quad}")
    print(f"Quad X range: {quad[:,0].min():.1f} - {quad[:,0].max():.1f}")
    print(f"Quad Y range: {quad[:,1].min():.1f} - {quad[:,1].max():.1f}")
    print(f"Quad width: {quad[:,0].max() - quad[:,0].min():.1f}")
    print(f"Quad height: {quad[:,1].max() - quad[:,1].min():.1f}")
    
    # Check if quad is within image
    tol = 10
    x_ok = quad[:,0].min() >= -tol and quad[:,0].max() <= w + tol
    y_ok = quad[:,1].min() >= -tol and quad[:,1].max() <= h + tol
    print(f"Quad in bounds: X={x_ok}, Y={y_ok}")
    
    # Now run the same pipeline as the QA script
    import sys
    sys.path.insert(0, '/home/wzzz/LPRNet/src')
    from load_data import prepare_board_ocr_input_from_quad_bgr888
    
    g3, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    print(f"\nOutput shape: {g3.shape}")
    print(f"Occ ratio: {occ:.3f}")
    print(f"Warped shape: {warped.shape}")
    print(f"Gray3 value range: {g3.min():.0f} - {g3.max():.0f}")
    
    # Save the warped plate for inspection
    cv2.imwrite('/tmp/qa04_warped.jpg', warped)
    cv2.imwrite('/tmp/qa04_gray3.jpg', g3)
    
    # Also check with the GT quad (from filename parse)
    from load_data import parse_ccpd_quad_from_name, order_quad_points
    raw_quad = parse_ccpd_quad_from_name(img_path)
    gt_quad = order_quad_points(raw_quad)
    print(f"\nGT quad: {gt_quad}")
    
    g3_gt, occ_gt, warped_gt, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, gt_quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    print(f"\nGT output shape: {g3_gt.shape}")
    print(f"GT Occ ratio: {occ_gt:.3f}")
    cv2.imwrite('/tmp/qa04_gt_warped.jpg', warped_gt)
    cv2.imwrite('/tmp/qa04_gt_gray3.jpg', g3_gt)
    
    print("\nDone. Check /tmp/qa04_warped.jpg and /tmp/qa04_gt_warped.jpg")
