#!/usr/bin/env python3
import csv, json
from pathlib import Path

# Find the qa_04 sample - it's sample [4] from QA output: green8 ccpd2020 皖AD03574
rows = list(csv.DictReader(open('/home/wzzz/LPRNet/manifests/curriculum_gray3_stageb_v1_B2D_pose_quad/train_pose_quad.csv', encoding='utf-8-sig')))

found = None
for r in rows:
    if '皖AD035' in r['text']:
        found = r
        break

if found:
    print('img_path:', found['img_path'])
    print('text:', found['text'])
    print('source:', found['source'])
    print('quad_source:', found['quad_source'])
    print('quad_1x:', found['quad_1x'], 'quad_1y:', found['quad_1y'])
    print('quad_2x:', found['quad_2x'], 'quad_2y:', found['quad_2y'])
    print('quad_3x:', found['quad_3x'], 'quad_3y:', found['quad_3y'])
    print('quad_4x:', found['quad_4x'], 'quad_4y:', found['quad_4y'])
    
    # Also check the Pose quad source - is it from the pose_quads.jsonl?
    img_path = found['img_path']
    
    # Check in pose_quads.jsonl
    with open('/home/wzzz/LPRNet/datasets/ccpd2020_pose_quads/pose_quads.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            if entry['img_path'] == img_path:
                print('\nPose quads lookup:')
                print('  pose_quad:', entry['pose_quad'])
                break
        else:
            print('\n  NOT found in pose_quads.jsonl')
else:
    print('No match found')
