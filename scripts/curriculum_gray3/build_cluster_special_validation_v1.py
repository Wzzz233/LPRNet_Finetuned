#!/usr/bin/env python3
"""Build fixed green cluster special validation set v1.

This manifest is a validation-only set for real board-domain green failures.
It must not be used for training.
"""
import csv
import json
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests/cluster_special_validation_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'cluster_special_validation_v1.csv'
SUMMARY = OUT_DIR / 'summary.json'

SOURCES = {
    'cluster1': ROOT / 'tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv',
    'cluster2': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv',
    'cluster3': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv',
    'cluster3_tail': ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv',
}

FIELDS = [
    'sample_uid','cluster_id','sub_cluster','split','gt_text','family','source','failure_subtype','frame_id','sample_id','ts_us',
    'app_text','app_conf','app_occ_ratio','local_ocrin_path','local_crop_path','img_path','crop_path','note',
    'use_for_training','validation_purpose'
]


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def exists(path):
    return bool(path) and Path(path).exists()


def main():
    rows=[]
    for cluster,path in SOURCES.items():
        if not path.exists():
            continue
        for i,r in enumerate(read_csv(path)):
            if cluster == 'cluster1':
                ocr = r.get('img_path','')
                crop = ''
                gt = r.get('text','')
                frame = Path(ocr).stem.replace('ocrin_','')
                sid = str(i)
                app_text = ''
                app_conf = ''
                occ = ''
                source = r.get('source','board_dump_benchmark_cluster1')
                fail = 'aa0_slot_alignment'
                note = 'cluster1 陕AA02222 AA0/slot alignment board-native benchmark'
            else:
                ocr = r.get('local_ocrin_path') or r.get('ocr_input_path') or r.get('img_path') or ''
                crop = r.get('local_crop_path') or r.get('crop_path') or ''
                gt = r.get('gt_text') or r.get('text') or ''
                frame = r.get('frame_id','')
                sid = r.get('sample_id',str(i))
                app_text = r.get('app_text','')
                app_conf = r.get('app_conf','')
                occ = r.get('app_occ_ratio','')
                source = f'board_dump_{cluster}'
                fail = r.get('failure_type','') or ('province_anchor' if cluster=='cluster2' else 'transition_slot_collapse')
                note = r.get('note','')
            if not gt or not exists(ocr):
                continue
            if cluster != 'cluster1' and crop and not exists(crop):
                crop = ''
            rows.append({
                'sample_uid': f'{cluster}_{sid}_f{frame}',
                'cluster_id': cluster,
                'sub_cluster': cluster,
                'split': 'test',
                'gt_text': gt,
                'family': 'green8',
                'source': source,
                'failure_subtype': fail,
                'frame_id': frame,
                'sample_id': sid,
                'ts_us': r.get('ts_us',''),
                'app_text': app_text,
                'app_conf': app_conf,
                'app_occ_ratio': occ,
                'local_ocrin_path': ocr,
                'local_crop_path': crop,
                'img_path': ocr,
                'crop_path': crop,
                'note': note,
                'use_for_training': '0',
                'validation_purpose': 'real_board_green_common_failure_special_validation_do_not_train',
            })
    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    by_cluster={}
    for r in rows:
        c=r['cluster_id']; by_cluster.setdefault(c,0); by_cluster[c]+=1
    summary={
        'manifest': str(OUT_CSV),
        'total': len(rows),
        'by_cluster': by_cluster,
        'sources': {k:str(v) for k,v in SOURCES.items()},
        'training_allowed': False,
        'note': 'Fixed special validation set for cluster common-failure validation. Do not train on these rows.',
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
