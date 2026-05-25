#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

KEY_PROVS = ['苏', '沪', '浙', '粤', '赣', '豫']
SYNTH_SOURCES = {
    'synthetic_exact_quad',
    'synthetic_exact_quad_edgefit_tier3_v3_su_conservative',
    'v4_boardlike_edgefit',
}
HARD_BUCKETS = {'board_low_occ', 'board_extreme_tail'}


def load_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def prov_of(row):
    text = row.get('text') or ''
    return text[:1] if text else ''


def exists_row(row):
    p = row.get('img_path') or ''
    return bool(p) and Path(p).exists()


def dedup_by_img(rows):
    seen = set()
    out = []
    for r in rows:
        p = r.get('img_path')
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(r)
    return out


def summarize(name, rows):
    prov = Counter(prov_of(r) for r in rows if prov_of(r))
    src = Counter(r.get('source', '') for r in rows)
    dataset = Counter(r.get('dataset_name', '') for r in rows)
    bucket = Counter(r.get('bucket', '') for r in rows if r.get('bucket'))
    split = Counter(r.get('split', '') for r in rows)
    real_vs = Counter('real' if r.get('source') == 'real' else 'synthetic' for r in rows)
    return {
        'name': name,
        'sample_count': len(rows),
        'split_counts': dict(split),
        'top_provinces': prov.most_common(20),
        'source_counts': dict(src),
        'dataset_counts': dict(dataset),
        'bucket_counts': dict(bucket),
        'real_vs_synth': dict(real_vs),
    }


def build_original_proxies(rows):
    test_green = [r for r in rows if r.get('split') == 'test' and r.get('family') == 'green8' and exists_row(r)]
    real = [r for r in test_green if r.get('source') == 'real']
    synth = [r for r in test_green if r.get('source') != 'real']

    p1 = list(test_green)

    by_prov_real = defaultdict(list)
    for r in real:
        by_prov_real[prov_of(r)].append(r)
    p2 = []
    anhui_cap = 1200
    p2.extend(by_prov_real.get('皖', [])[:anhui_cap])
    for prov in KEY_PROVS:
        p2.extend(by_prov_real.get(prov, []))
    for prov, items in sorted(by_prov_real.items()):
        if prov == '皖' or prov in KEY_PROVS:
            continue
        p2.extend(items)
    by_prov_synth = defaultdict(list)
    for r in synth:
        by_prov_synth[prov_of(r)].append(r)
    synth_fill_cap = 25
    for prov, items in sorted(by_prov_synth.items()):
        take = 40 if prov in KEY_PROVS else synth_fill_cap
        p2.extend(items[:take])
    p2 = dedup_by_img(p2)

    p3 = []
    key_real_target = 100
    key_synth_target = 50
    for prov in KEY_PROVS:
        p3.extend(by_prov_real.get(prov, [])[:key_real_target])
        p3.extend(by_prov_synth.get(prov, [])[:key_synth_target])
    p3 = dedup_by_img(p3)
    if len(p3) < 400:
        seen = {r.get('img_path') for r in p3}
        fill_candidates = []
        for prov in KEY_PROVS:
            fill_candidates.extend(by_prov_real.get(prov, [])[key_real_target:])
            fill_candidates.extend(by_prov_synth.get(prov, [])[key_synth_target:])
        for r in fill_candidates:
            p = r.get('img_path')
            if p in seen:
                continue
            seen.add(p)
            p3.append(r)
            if len(p3) >= 400:
                break

    aux = [r for r in test_green if r.get('source') in SYNTH_SOURCES]

    return {
        'green8_only_proxy': p1,
        'green8_balanced_proxy': p2,
        'green8_keyprov_proxy': p3,
        'green8_synth_aux_proxy': aux,
    }


def quota_fill(rows, total_target, keyprov_floor=0):
    by_prov = defaultdict(list)
    for r in rows:
        by_prov[prov_of(r)].append(r)
    out = []
    # key provinces first
    for prov in KEY_PROVS:
        items = by_prov.get(prov, [])
        if not items:
            continue
        take = min(len(items), keyprov_floor)
        out.extend(items[:take])
        by_prov[prov] = items[take:]
    if len(out) >= total_target:
        return dedup_by_img(out[:total_target])
    round_robin = True
    while round_robin and len(out) < total_target:
        round_robin = False
        for prov in sorted(by_prov.keys()):
            items = by_prov[prov]
            if not items:
                continue
            out.append(items.pop(0))
            round_robin = True
            if len(out) >= total_target:
                break
    return dedup_by_img(out[:total_target])


def to_eval_rows(v4_rows, fieldnames, lowocc_target):
    out = []
    lowocc = []
    extreme = []
    for r in v4_rows:
        if not exists_row(r):
            continue
        family = (r.get('family') or '').strip()
        if family != 'green8':
            continue
        bucket = (r.get('img_rel_path') or '').split('/')[2] if '/' in (r.get('img_rel_path') or '') else ''
        row = dict(r)
        row['split'] = 'test'
        row['dataset_name'] = row.get('dataset_name') or 'green_edgefit_v4_boardlike_a3000'
        row['source'] = 'v4_boardlike_edgefit'
        row['bucket'] = bucket
        if 'bucket' not in fieldnames:
            pass
        if bucket == 'board_low_occ':
            lowocc.append(row)
        elif bucket == 'board_extreme_tail':
            extreme.append(row)
    lowocc = quota_fill(lowocc, lowocc_target, keyprov_floor=20)
    extreme = quota_fill(extreme, len(extreme), keyprov_floor=20)
    hard = dedup_by_img(lowocc + extreme)
    return {
        'green8_v4_lowocc_proxy': lowocc,
        'green8_v4_extreme_proxy': extreme,
        'green8_v4_hard_proxy': hard,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='base unified manifest for original proxies')
    ap.add_argument('--v4_manifest', required=True, help='train manifest exported from green_edgefit_v4_boardlike_a3000')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lowocc_target', type=int, default=600)
    args = ap.parse_args()

    rows = load_rows(args.manifest)
    v4_rows = load_rows(args.v4_manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    if 'bucket' not in fieldnames:
        fieldnames = fieldnames + ['bucket']

    original = build_original_proxies(rows)
    hard = to_eval_rows(v4_rows, fieldnames, args.lowocc_target)
    proxies = {**original, **hard}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, subset in proxies.items():
        path = out_dir / f'{name}.csv'
        ready_rows = []
        for row in subset:
            new_row = {k: row.get(k, '') for k in fieldnames}
            ready_rows.append(new_row)
        write_rows(path, fieldnames, ready_rows)
        summary[name] = summarize(name, ready_rows)
        summary[name]['path'] = str(path)

    summary['meta'] = {
        'manifest': args.manifest,
        'v4_manifest': args.v4_manifest,
        'lowocc_target': args.lowocc_target,
        'hard_policy': {
            'keep_original_proxies': True,
            'new_hard_buckets': ['board_low_occ', 'board_extreme_tail'],
            'exclude_midocc_from_hard_main': True,
            'extreme_proxy_uses_all_extreme_rows': True,
        }
    }
    summary_path = out_dir / 'proxy_manifest_summary_with_v4_hard.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
