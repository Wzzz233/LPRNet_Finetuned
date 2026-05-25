#!/usr/bin/env python3
"""Detailed visual comparison of GT vs Pose warp."""
import cv2
import numpy as np

# Load the warped outputs
pose_warped = cv2.imread('/tmp/qa04_warped.jpg')
gt_warped = cv2.imread('/tmp/qa04_gt_warped.jpg')

# Check for missing content at top-right corner
# In the Pose quad, TR.y=492.3 vs GT TR.y=477 → Pose is 15px lower
# This means the warp might cut off the top of characters on the right side

for name, warped in [('Pose', pose_warped), ('GT', gt_warped)]:
    if warped is None:
        continue
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Divide into 4 quadrants: top-left, top-right, bottom-left, bottom-right
    mid_h, mid_w = h // 2, w // 2
    
    tl = gray[:mid_h, :mid_w]
    tr = gray[:mid_h, mid_w:]
    bl = gray[mid_h:, :mid_w]
    br = gray[mid_h:, mid_w:]
    
    print(f"=== {name} warped ({w}x{h}) ===")
    print(f"  TL quadrant mean: {tl.mean():.1f}")
    print(f"  TR quadrant mean: {tr.mean():.1f}")
    print(f"  BL quadrant mean: {bl.mean():.1f}")
    print(f"  BR quadrant mean: {br.mean():.1f}")
    print(f"  TR/TL ratio: {tr.mean()/max(tl.mean(),1):.2f}")
    
    # Check top edge for brightness drops (indicating missing content)
    top_row_means = [gray[:3, i*w//10:(i+1)*w//10].mean() for i in range(10)]
    print(f"  Top edge per-column means: {[f'{m:.0f}' for m in top_row_means]}")
    
    # Check column brightness profile
    col_means = gray.mean(axis=0)
    # Left 10 cols vs right 10 cols
    left_edge = col_means[:10].mean()
    right_edge = col_means[-10:].mean()
    print(f"  Leftmost 10 cols mean: {left_edge:.1f}")
    print(f"  Rightmost 10 cols mean: {right_edge:.1f}")
    
    # Check if there's a content gradient from left to right
    left_half = col_means[:w//2].mean()
    right_half = col_means[w//2:].mean()
    print(f"  Left half mean: {left_half:.1f}")
    print(f"  Right half mean: {right_half:.1f}")
    print()

# Check the 94x24 gray3 output
pose_g3 = cv2.imread('/tmp/qa04_gray3.jpg')
gt_g3 = cv2.imread('/tmp/qa04_gt_gray3.jpg')

for name, g3 in [('Pose', pose_g3), ('GT', gt_g3)]:
    if g3 is None:
        continue
    print(f"=== {name} gray3 94x24 ===")
    gray = cv2.cvtColor(g3, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    
    # Column means across the full width
    col_means = gray.mean(axis=0)
    
    # Check first 10% vs last 10%
    first = col_means[:w//10].mean()
    last = col_means[-w//10:].mean()
    middle = col_means[w//10:-w//10].mean()
    print(f"  First 10% cols: {first:.1f}")
    print(f"  Middle 80% cols: {middle:.1f}")
    print(f"  Last 10% cols: {last:.1f}")
    
    # Check for large uniform areas (blank = missing content)
    # A column with content varies; blank columns are uniform
    col_std = gray.std(axis=0)
    low_std_cols = (col_std < 15).sum()
    print(f"  Low-variance cols (std<15): {low_std_cols}/{w}")
    
    # Specifically check if the first character position has content
    # In 94x24, first char is roughly cols 3-14 (at 94 width)
    first_char_region = gray[:, 3:14]
    print(f"  First char region (cols 3-14) variance: {first_char_region.std():.1f}")
    print(f"  First char region mean: {first_char_region.mean():.1f}")
    print()
