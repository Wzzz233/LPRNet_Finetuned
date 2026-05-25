#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

BASE = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv'
OUT = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak.csv'
TARGET_RATIO = 0.40
ROOT_SYN = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests')
TRAIN_SYN = ROOT_SYN / 'train_synthetic_labels.txt'
VAL_SYN = ROOT_SYN / 'val_synthetic_labels.txt'
TEST_SYN = ROOT_SYN / 'test_synthetic_labels.txt'

def load_label_file(path):
    out=set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel,text=line.strip().split(maxsplit=1)
        out.add((rel, text.strip().upper()))
    return out

train_syn_set = load_label_file(TRAIN_SYN)
val_syn_set = load_label_file(VAL_SYN)
test_syn_set = load_label_file(TEST_SYN)
with open(BASE,'r',encoding='utf-8',newline='') as f:
    reader=csv.DictReader(f)
    fieldnames=reader.fieldnames
    rows=list(reader)

base_train=[r for r in rows if r.get('split')=='train']
others=[r for r in rows if r.get('split')!='train']
base_train_keys=set((r['img_path'], r.get('text','')) for r in base_train)

def rel_and_text(row):
    rel=row.get('img_rel_path') or ''
    text=(row.get('text') or '').upper()
    return (rel, text)

# only allow synthetic rows that belong to synthetic train split and are not in synthetic val/test splits
safe_syn=[]
for r in rows:
    if r.get('family')!='green8' or r.get('source')!='synthetic_exact_quad':
        continue
    key=(r.get('img_path'), r.get('text',''))
    if key in base_train_keys:
        continue
    rt=rel_and_text(r)
    if rt in train_syn_set and rt not in val_syn_set and rt not in test_syn_set and not (r.get('text') or '').startswith('皖'):
        safe_syn.append(r.copy())

safe_syn.sort(key=lambda r: ((r.get('text') or '')[:1], r.get('img_path') or ''))
base_total=len(base_train)
anhui_count=sum(1 for r in base_train if (r.get('text') or '').startswith('皖'))
needed=0
while anhui_count/(base_total+needed) > TARGET_RATIO:
    needed += 1
selected=safe_syn[:needed]
for r in selected:
    r['split']='train'
new_train=list(base_train)+selected
with open(OUT,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(new_train)
    w.writerows(others)
prov=Counter((r.get('text') or '')[:1] or '__empty__' for r in new_train)
src=Counter(r.get('source') or 'unknown' for r in new_train)
report={
    'src': BASE,
    'out': OUT,
    'target_anhui_ratio': TARGET_RATIO,
    'original_train_total': len(base_train),
    'original_train_anhui': anhui_count,
    'safe_synthetic_candidates': len(safe_syn),
    'needed_for_exact_40': needed,
    'picked_safe_synthetic': len(selected),
    'new_train_total': len(new_train),
    'new_train_anhui': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')),
    'new_train_anhui_ratio': sum(1 for r in new_train if (r.get('text') or '').startswith('皖'))/len(new_train),
    'source_breakdown': dict(src),
    'top_provinces': prov.most_common(15),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
