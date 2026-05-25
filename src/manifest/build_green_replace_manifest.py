#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_row(fieldnames, row):
    out = {k: row.get(k, '') for k in fieldnames}
    return out


def parse_list(text):
    if not text:
        return []
    return [x.strip() for x in text.split(',') if x.strip()]


def row_province(row):
    text = row.get('text') or ''
    return text[:1] if text else ''


def row_pos2(row):
    text = row.get('text') or ''
    return text[2] if len(text) > 2 else ''


def main():
    ap = argparse.ArgumentParser(description='Replace selected green8 train rows in a base manifest with generated board_dump rows.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--generated-manifest', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--replace-sources', default='board_native_e7_v2,board_native_cluster1_append_v1,v4_boardlike_edgefit,synthetic_exact_quad_edgefit_tier3_v3_su_conservative,synthetic_exact_quad')
    ap.add_argument('--priority-provinces', default='陕,苏,沪,浙,粤')
    ap.add_argument('--remove-pos2', default='D,F')
    args = ap.parse_args()

    base_path = Path(args.base_manifest)
    gen_path = Path(args.generated_manifest)
    out_path = Path(args.out_manifest)
    summary_path = Path(args.out_summary)

    base_fields, base_rows = load_csv(base_path)
    gen_fields, gen_rows = load_csv(gen_path)
    gen_rows = [normalize_row(base_fields, r) for r in gen_rows if (r.get('split') or '') == 'train']
    if not gen_rows:
        raise RuntimeError('generated manifest has no train rows')

    replace_sources = parse_list(args.replace_sources)
    priority_provinces = parse_list(args.priority_provinces)
    remove_pos2 = set(parse_list(args.remove_pos2))
    source_rank = {src: i for i, src in enumerate(replace_sources)}
    priority_set = set(priority_provinces)

    for r in gen_rows:
        if (r.get('preprocess_group') or '') != 'board_dump':
            raise RuntimeError(f'generated row is not board_dump: {r.get("img_rel_path")}')

    existing_pairs = {(r.get('img_path') or '', r.get('text') or '') for r in base_rows}
    for r in gen_rows:
        pair = (r.get('img_path') or '', r.get('text') or '')
        if pair in existing_pairs:
            raise RuntimeError(f'generated row duplicates base row: {pair}')

    candidate_indices = []
    for idx, row in enumerate(base_rows):
        if (row.get('split') or '') != 'train':
            continue
        if (row.get('family') or '') != 'green8':
            continue
        if (row.get('source') or '') not in source_rank:
            continue
        if row_pos2(row) not in remove_pos2:
            continue
        candidate_indices.append(idx)

    def sort_key(idx):
        row = base_rows[idx]
        prov = row_province(row)
        src = row.get('source') or ''
        return (
            0 if prov in priority_set else 1,
            source_rank.get(src, 999),
            prov,
            row.get('img_rel_path') or '',
        )

    candidate_indices.sort(key=sort_key)
    remove_count = len(gen_rows)
    if len(candidate_indices) < remove_count:
        raise RuntimeError(f'Not enough removable rows: need {remove_count}, have {len(candidate_indices)}')

    remove_indices = set(candidate_indices[:remove_count])
    removed_rows = [base_rows[i] for i in sorted(remove_indices)]
    kept_rows = [row for i, row in enumerate(base_rows) if i not in remove_indices]
    new_rows = kept_rows + gen_rows
    write_csv(out_path, base_fields, new_rows)

    summary = {
        'mode': 'replace',
        'base_manifest': str(base_path),
        'generated_manifest': str(gen_path),
        'out_manifest': str(out_path),
        'base_total': len(base_rows),
        'generated_total': len(gen_rows),
        'removed_total': len(removed_rows),
        'new_total': len(new_rows),
        'replace_sources': replace_sources,
        'priority_provinces': priority_provinces,
        'remove_pos2': sorted(remove_pos2),
        'removed_source_counts': dict(Counter(r.get('source') or '' for r in removed_rows)),
        'removed_province_counts': dict(Counter(row_province(r) for r in removed_rows)),
        'removed_pos2_counts': dict(Counter(row_pos2(r) for r in removed_rows)),
        'generated_source_counts': dict(Counter(r.get('source') or '' for r in gen_rows)),
        'generated_province_counts': dict(Counter(row_province(r) for r in gen_rows)),
        'generated_pos2_counts': dict(Counter(row_pos2(r) for r in gen_rows)),
        'removed_examples': [
            {
                'img_rel_path': r.get('img_rel_path'),
                'text': r.get('text'),
                'source': r.get('source'),
            }
            for r in removed_rows[:20]
        ],
        'generated_examples': [
            {
                'img_rel_path': r.get('img_rel_path'),
                'text': r.get('text'),
                'source': r.get('source'),
            }
            for r in gen_rows[:20]
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
