#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

PROVINCES = {'苏', '沪'}


def load_rows(txt_path):
    rows = []
    for line in Path(txt_path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rel, text = line.split(maxsplit=1)
        rows.append({'rel_path': rel.replace('\\', '/'), 'text': text.strip()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_province', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20260403)
    args = ap.parse_args()

    train = load_rows(args.train)
    val = load_rows(args.val)
    test = load_rows(args.test)
    train_paths = {r['rel_path'] for r in train}
    train_texts = {r['text'] for r in train}

    candidates = []
    reasons = []
    for src_name, rows in [('val', val), ('test', test)]:
        for r in rows:
            p = r['text'][0] if r['text'] else ''
            if p not in PROVINCES:
                continue
            rel = r['rel_path']
            if rel in train_paths:
                reasons.append({'src': src_name, 'rel_path': rel, 'text': r['text'], 'reason': 'path_in_train'})
                continue
            if r['text'] in train_texts:
                reasons.append({'src': src_name, 'rel_path': rel, 'text': r['text'], 'reason': 'text_in_train'})
                continue
            item = dict(r)
            item['src_split'] = src_name
            candidates.append(item)

    byp = {'苏': [], '沪': []}
    for r in candidates:
        byp[r['text'][0]].append(r)
    rng = random.Random(args.seed)
    selected = []
    for p in ['苏', '沪']:
        rows = byp[p]
        rng.shuffle(rows)
        selected.extend(rows[:args.per_province])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_txt = out_dir / 'holdout_labels.txt'
    with holdout_txt.open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(f"{r['rel_path']} {r['text']}\n")

    report = {
        'candidate_counts': {k: len(v) for k, v in byp.items()},
        'selected_counts': {
            '苏': sum(1 for r in selected if r['text'].startswith('苏')),
            '沪': sum(1 for r in selected if r['text'].startswith('沪')),
        },
        'selected_total': len(selected),
        'examples': selected[:10],
        'filtered_examples': reasons[:10],
        'holdout_txt': str(holdout_txt),
    }
    (out_dir / 'holdout_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
