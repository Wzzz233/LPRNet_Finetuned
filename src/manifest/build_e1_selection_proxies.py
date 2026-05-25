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


def load_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def summarize(name, rows):
    prov = Counter(prov_of(r) for r in rows if prov_of(r))
    src = Counter(r.get('source', '') for r in rows)
    real_vs = Counter('real' if r.get('source') == 'real' else 'synthetic' for r in rows)
    return {
        'name': name,
        'sample_count': len(rows),
        'top_provinces': prov.most_common(20),
        'source_counts': dict(src),
        'real_vs_synth': dict(real_vs),
    }


def build_proxies(rows):
    test_green = [r for r in rows if r.get('split') == 'test' and r.get('family') == 'green8' and exists_row(r)]
    real = [r for r in test_green if r.get('source') == 'real']
    synth = [r for r in test_green if r.get('source') != 'real']

    # P1: all green8 test rows, enough samples, no extra filtering
    p1 = list(test_green)

    # P2: balanced-ish real-heavy proxy with explicit Anhui cap and sufficient key provinces.
    by_prov_real = defaultdict(list)
    for r in real:
        by_prov_real[prov_of(r)].append(r)
    p2 = []
    # keep Anhui substantial but capped to avoid domination; ensure total count remains large enough
    anhui_cap = 1200
    p2.extend(by_prov_real.get('皖', [])[:anhui_cap])
    for prov in KEY_PROVS:
        p2.extend(by_prov_real.get(prov, []))
    # add all remaining non-Anhui real provinces for coverage
    for prov, items in sorted(by_prov_real.items()):
        if prov == '皖' or prov in KEY_PROVS:
            continue
        p2.extend(items)
    # add synthetic support so balanced proxy is not too sparse and still covers all provinces
    by_prov_synth = defaultdict(list)
    for r in synth:
        by_prov_synth[prov_of(r)].append(r)
    synth_fill_cap = 25
    for prov, items in sorted(by_prov_synth.items()):
        take = synth_fill_cap
        if prov in KEY_PROVS:
            take = 40
        p2.extend(items[:take])

    # dedup by image path while preserving order
    seen = set()
    p2_dedup = []
    for r in p2:
        p = r.get('img_path')
        if p in seen:
            continue
        seen.add(p)
        p2_dedup.append(r)
    p2 = p2_dedup

    # P3: key province proxy, real-first, then synth补足；为避免样本过少，目标总量>=400
    p3 = []
    key_real_target = 100
    key_synth_target = 50
    for prov in KEY_PROVS:
        p3.extend(by_prov_real.get(prov, [])[:key_real_target])
        p3.extend(by_prov_synth.get(prov, [])[:key_synth_target])
    seen = set()
    p3_dedup = []
    for r in p3:
        p = r.get('img_path')
        if p in seen:
            continue
        seen.add(p)
        p3_dedup.append(r)
    p3 = p3_dedup
    if len(p3) < 400:
        # 用 key province 全量 green8 样本继续补足，优先 real，再 synth，直到数量足够或数据耗尽
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

    # Aux synth proxy: enough rows from each synth source to track v4 learning
    aux = [r for r in test_green if r.get('source') in SYNTH_SOURCES]

    return {
        'green8_only_proxy': p1,
        'green8_balanced_proxy': p2,
        'green8_keyprov_proxy': p3,
        'green8_synth_aux_proxy': aux,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    rows = load_rows(args.manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    proxies = build_proxies(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, subset in proxies.items():
        path = out_dir / f'{name}.csv'
        write_rows(path, fieldnames, subset)
        summary[name] = summarize(name, subset)
        summary[name]['path'] = str(path)
    summary_path = out_dir / 'proxy_manifest_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
