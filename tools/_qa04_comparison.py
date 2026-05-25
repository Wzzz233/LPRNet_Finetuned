#!/usr/bin/env python3
"""Compare GT vs Pose quad on the original image."""
import cv2
import numpy as np

img_path = '/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/303747829861111111-90_95-243&477_557&597-557&595_244&597_243&489_548&477-0_0_3_24_27_29_30_33-67-125.jpg'
img = cv2.imread(img_path)
h, w = img.shape[:2]

gt_quad = np.array([[243., 489.], [548., 477.], [557., 595.], [244., 597.]], dtype=np.float32)
pose_quad = np.array([[236.2, 487.0], [554.6, 492.3], [552.4, 590.9], [236.2, 597.6]], dtype=np.float32)

# Draw both quads on the image
vis = img.copy()
cv2.polylines(vis, [gt_quad.round().astype(np.int32).reshape(-1,1,2)], True, (0, 255, 0), 3)
cv2.polylines(vis, [pose_quad.round().astype(np.int32).reshape(-1,1,2)], True, (255, 0, 0), 3)

# Add labels
cv2.putText(vis, 'GT (green)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(vis, 'Pose (blue)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Crop to region around plate
y1 = max(0, int(pose_quad[:,1].min()) - 20)
y2 = min(h, int(pose_quad[:,1].max()) + 20)
x1 = max(0, int(pose_quad[:,0].min()) - 20)
x2 = min(w, int(pose_quad[:,0].max()) + 20)
crop = vis[y1:y2, x1:x2]
cv2.imwrite('/tmp/qa04_quad_comparison.jpg', crop)

# Also create the QA-style visualization  
import sys
sys.path.insert(0, '/home/wzzz/LPRNet/src')
from load_data import prepare_board_ocr_input_from_quad_bgr888

# GT
g3_gt, occ_gt, warped_gt, _, _ = prepare_board_ocr_input_from_quad_bgr888(
    img, gt_quad, 94, 24,
    resize_mode='letterbox', resize_kernel='nn',
    preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)

# Pose
g3_pose, occ_pose, warped_pose, _, _ = prepare_board_ocr_input_from_quad_bgr888(
    img, pose_quad, 94, 24,
    resize_mode='letterbox', resize_kernel='nn',
    preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)

print(f"Image size: {w}x{h}")
print(f"GT quad: {gt_quad}")
print(f"Pose quad: {pose_quad}")
print(f"GT warped: {warped_gt.shape[1]}x{warped_gt.shape[0]}, occ={occ_gt:.3f}")
print(f"Pose warped: {warped_pose.shape[1]}x{warped_pose.shape[0]}, occ={occ_pose:.3f}")

# Write the 94x24 glyph for visual inspection (scaled up)
cv2.imwrite('/tmp/qa04_comparison_gt_gray3.png', cv2.resize(g3_gt, (188, 48)))
cv2.imwrite('/tmp/qa04_comparison_pose_gray3.png', cv2.resize(g3_pose, (188, 48)))

# Side-by-side
h_stack = np.hstack([
    cv2.resize(g3_gt, (188, 48)),
    np.ones((48, 10, 3), dtype=np.uint8) * 128,
    cv2.resize(g3_pose, (188, 48)),
])
cv2.imwrite('/tmp/qa04_gray3_side_by_side.png', h_stack)
print(f"\nSaved side-by-side: /tmp/qa04_gray3_side_by_side.png")
print(f"Left = GT, Right = Pose")
print()
print(f"GT gray3 first 20 cols mean: {cv2.cvtColor(g3_gt, cv2.COLOR_BGR2GRAY).mean(axis=0)[:20]}")
print(f"Pose gray3 first 20 cols mean: {cv2.cvtColor(g3_pose, cv2.COLOR_BGR2GRAY).mean(axis=0)[:20]}")
