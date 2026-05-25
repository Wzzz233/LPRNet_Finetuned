#!/usr/bin/env python3
import json
from pathlib import Path

# Find the Pose inference result for this specific image
pose_quads = '/home/wzzz/LPRNet/datasets/ccpd2020_pose_quads/pose_quads.jsonl'
failures = '/home/wzzz/LPRNet/datasets/ccpd2020_pose_quads/pose_quads_failures.jsonl'

target = '/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green/train/303747829861111111-90_95-243&477_557&597-557&595_244&597_243&489_548&477-0_0_3_24_27_29_30_33-67-125.jpg'

# Search in pose_quads
found = None
with open(pose_quads) as f:
    for line in f:
        entry = json.loads(line)
        if entry['img_path'] == target:
            found = entry
            break

if found:
    print("Pose quad entry found:")
    print(f"  pose_quad: {found['pose_quad']}")
    print(f"  confidence: {found.get('confidence', 'N/A')}")
    print(f"  bbox: {found.get('bbox', 'N/A')}")
    print(f"  score: {found.get('score', 'N/A')}")
else:
    print("Entry not found in pose_quads.jsonl")
    # Check failures
    with open(failures) as f:
        for line in f:
            entry = json.loads(line)
            if entry['img_path'] == target:
                print("Found in failures!")
                break

# Now look at the actual GT quad from the CCPD filename  
from load_data import parse_ccpd_quad_from_name, order_quad_points
import re

# Parse from filename
stem = Path(target).stem
print(f"\nFilename stem: {stem}")
print(f"CCPD quad from filename: {stem.split('-')[2]}")

raw_quad = parse_ccpd_quad_from_name(target)
ordered = order_quad_points(raw_quad)
print(f"\nParsed GT quad (raw): {raw_quad}")
print(f"Parsed GT quad (ordered): {ordered}")

# Also check: what quad was stored in the manifest?
# The manifest has pose quad - let's also check if there's a failure flag
import csv
with open('/home/wzzz/LPRNet/manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_pose_quad.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if target in row['img_path']:
            print(f"\nManifest quad_source: {row['quad_source']}")
            print(f"Manifest quad: ({row['quad_1x']},{row['quad_1y']}) ({row['quad_2x']},{row['quad_2y']}) ({row['quad_3x']},{row['quad_3y']}) ({row['quad_4x']},{row['quad_4y']})")
            break
