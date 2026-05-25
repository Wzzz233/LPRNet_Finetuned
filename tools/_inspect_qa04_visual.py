#!/usr/bin/env python3
"""Visual check of the warped outputs."""
import cv2
import numpy as np

# Check the Pose quad warped output
warped = cv2.imread('/tmp/qa04_warped.jpg')
if warped is not None:
    print(f"Pose warped: {warped.shape[1]}x{warped.shape[0]}")
    # Check for blank/black borders
    # Left column
    left_col = warped[:, 0, :]
    right_col = warped[:, -1, :]
    top_row = warped[0, :, :]
    bottom_row = warped[-1, :, :]
    print(f"  Left edge mean pixel: {left_col.mean():.1f}")
    print(f"  Right edge mean pixel: {right_col.mean():.1f}")
    print(f"  Top edge mean pixel: {top_row.mean():.1f}")
    print(f"  Bottom edge mean pixel: {bottom_row.mean():.1f}")
    
    # Count near-black pixels (>0 means content, <5 means near-black)
    total = warped.shape[0] * warped.shape[1]
    near_black = (warped.mean(axis=2) < 5).sum()
    print(f"  Near-black pixels: {near_black}/{total} ({near_black/total*100:.1f}%)")
    
    # Check if left side looks like it's missing plate content
    # Compare left 10% vs right 90%
    w = warped.shape[1]
    left_strip = warped[:, :w//10, :]
    main_area = warped[:, w//10:, :]
    print(f"  Left 10% mean: {left_strip.mean():.1f}")
    print(f"  Main area mean: {main_area.mean():.1f}")

# Check the GT warped too
gt_warped = cv2.imread('/tmp/qa04_gt_warped.jpg')
if gt_warped is not None:
    print(f"\nGT warped: {gt_warped.shape[1]}x{gt_warped.shape[0]}")
    left_strip = gt_warped[:, :gt_warped.shape[1]//10, :]
    main_area = gt_warped[:, gt_warped.shape[1]//10:, :]
    print(f"  Left 10% mean: {left_strip.mean():.1f}")
    print(f"  Main area mean: {main_area.mean():.1f}")
    
    # Ratio of left strip to main area
    ratio = left_strip.mean() / max(main_area.mean(), 1)
    print(f"  Left/Main ratio: {ratio:.2f}")

# Also check the original image around the quad
orig = cv2.imread('/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/303747829861111111-90_95-243&477_557&597-557&595_244&597_243&489_548&477-0_0_3_24_27_29_30_33-67-125.jpg')
if orig is not None:
    h, w = orig.shape[:2]
    # Crop a strip around the quad to verify
    y1, y2 = int(480), int(600)
    x1, x2 = int(230), int(560)
    plate_region = orig[y1:y2, x1:x2]
    cv2.imwrite('/tmp/qa04_plate_region.jpg', plate_region)
    print(f"\nOriginal plate region ({x1}:{x2}, {y1}:{y2}): {plate_region.shape}")
    print(f"  Mean pixel value: {plate_region.mean():.1f}")
