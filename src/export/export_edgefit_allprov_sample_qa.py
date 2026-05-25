#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, random
from pathlib import Path
import cv2, numpy as np
from data.load_data import prepare_board_ocr_input_from_quad_bgr888

ROOT = Path('/home/wzzz/LPRNet/green_edgefit_allprov_v1')
TSV = ROOT / 'details' / 'accepted.tsv'
OUT = Path('/home/wzzz/LPRNet/qa_green_edgefit_allprov_v1_board_sample_20260403')
TARGETS = [('train','simple','沪'),('train','harder','沪'),('train','simple','苏'),('train','harder','苏'),('train','harder','皖'),('val','harder','粤'),('test','simple','浙'),('test','harder','津')]

def mk(lines, width):
    canvas = np.full((36*len(lines)+20, width, 3), 255, np.uint8)
    y=34
    for line in lines:
        cv2.putText(canvas, line, (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20,20,20), 2, cv2.LINE_AA)
        y += 36
    return canvas

rows=[]
with TSV.open('r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        row['quad']=json.loads(row['quad'])
        rows.append(row)
OUT.mkdir(parents=True, exist_ok=True)
logs=[]
for idx, key in enumerate(TARGETS, 1):
    split,difficulty,province = key
    cand = [r for r in rows if r['split']==split and r['difficulty']==difficulty and r['province']==province]
    row = cand[0]
    img = cv2.imread(str(ROOT / row['rel_path']))
    quad = np.asarray(row['quad'], dtype=np.float32)
    proc, occ, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
    vis = img.copy()
    cv2.polylines(vis, [np.asarray(quad, dtype=np.int32).reshape(-1,1,2)], True, (0,255,0), 1)
    top = cv2.resize(vis, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    bot = cv2.resize(proc, (940, 240), interpolation=cv2.INTER_NEAREST)
    width = max(top.shape[1], bot.shape[1])
    if top.shape[1] < width:
        top = np.concatenate([top, np.full((top.shape[0], width-top.shape[1],3),255,np.uint8)], axis=1)
    if bot.shape[1] < width:
        bot = np.concatenate([bot, np.full((bot.shape[0], width-bot.shape[1],3),255,np.uint8)], axis=1)
    header = mk([f'idx={idx:02d} split={split} difficulty={difficulty} province={province} text={row["text"]}', f'board_occ={occ:.4f}', f'file={Path(row["rel_path"]).name}'], width)
    canvas = np.concatenate([header, top, bot], axis=0)
    out_path = OUT / f'{idx:02d}_{split}_{difficulty}_{province}_{row["text"]}.jpg'
    cv2.imwrite(str(out_path), canvas)
    logs.append(f'OK\t{out_path}')
(OUT/'index.tsv').write_text('\n'.join(logs)+'\n', encoding='utf-8')
print(str(OUT))
