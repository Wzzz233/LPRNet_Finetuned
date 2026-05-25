#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'evaluation'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS, PROVINCE_COUNT
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]
TARGET_PROVINCES = ('京', '皖')
PROV_TO_LABEL = {'皖': 0, '京': 1}
LABEL_TO_PROV = {v: k for k, v in PROV_TO_LABEL.items()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_manifest(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(out)


def build_dump_manifest(input_csv: Path, out_manifest: Path):
    rows = []
    meta_rows = []
    with input_csv.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_path = row.get('local_ocrin_path') or row.get('ocr_input_path') or row.get('img_path')
            gt = (row.get('gt_text') or '').strip()
            if not img_path or not gt:
                continue
            img_path = str(Path(img_path))
            if not Path(img_path).exists():
                continue
            rows.append({
                'img_path': img_path,
                'img_rel_path': img_path,
                'dataset_name': 'dump_replay',
                'split': 'test',
                'text': gt,
                'plate_len': len(gt),
                'family': 'green8',
                'sub_type': 'green',
                'source': 'dump_replay',
                'is_real': 1,
                'need_tilt_aug': 0,
                'preprocess_group': 'dump_replay',
                'has_bbox': 0,
                'has_quad': 0,
                'can_parse_ccpd_geom': 0,
                'can_perspective': 0,
                'bbox_source': 'none',
                'quad_source': 'none',
                'ocr_channel_order': 'bgr',
                'ocr_crop_mode': 'obb_warp',
                'ocr_resize_mode': 'letterbox',
                'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'none',
                'ocr_min_occ_ratio': 0.9,
                'ocr_quad_pad_ratio': 0.0,
            })
            meta = dict(row)
            meta['_row_index'] = i
            meta['_img_path'] = img_path
            meta_rows.append(meta)
    write_manifest(out_manifest, rows, MANIFEST_FIELDS)
    return meta_rows


def make_dataset(manifest_path: Path):
    ds_full = UnifiedManifestDataset(
        manifest_path=str(manifest_path),
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter=None,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i, row in enumerate(ds_full.records) if Path(row.get('img_path', '')).exists()]
    return Subset(ds_full, idx), [ds_full.records[i] for i in idx]


def load_model(model_path: Path, device):
    state = torch.load(str(model_path), map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def extract_features(net, manifest_path: Path, batch_size: int, num_workers: int, device, first_char_time_steps: int):
    ds, records = make_dataset(manifest_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    pooled_list = []
    proxy_list = []
    logits_mean_list = []
    labels = []
    texts = []
    img_paths = []
    with torch.no_grad():
        for images, _labels, _lengths, families in loader:
            images = images.to(device)
            context = net.extract_context(images)
            pooled = F.adaptive_avg_pool2d(context, (1, 1)).flatten(1)
            logits = forward_family_logits(net, images, sample_families=list(families))
            aux_steps = max(1, min(first_char_time_steps, logits.shape[2]))
            first_proxy_logits = logits[:, :PROVINCE_COUNT, :aux_steps].mean(dim=2)
            logits_mean = logits.mean(dim=2)
            pooled_list.append(pooled.detach().cpu().numpy())
            proxy_list.append(first_proxy_logits.detach().cpu().numpy())
            logits_mean_list.append(logits_mean.detach().cpu().numpy())
    for row in records:
        text = row['text']
        labels.append(PROV_TO_LABEL[text[0]])
        texts.append(text)
        img_paths.append(row['img_path'])
    return {
        'pooled_context': np.concatenate(pooled_list, axis=0) if pooled_list else np.zeros((0, 1), dtype=np.float32),
        'first_proxy_logits': np.concatenate(proxy_list, axis=0) if proxy_list else np.zeros((0, PROVINCE_COUNT), dtype=np.float32),
        'seq_mean_logits': np.concatenate(logits_mean_list, axis=0) if logits_mean_list else np.zeros((0, len(CHARS)), dtype=np.float32),
        'labels': np.asarray(labels, dtype=np.int64),
        'texts': texts,
        'img_paths': img_paths,
    }


def balanced_pick(rows_by_class, max_per_class=None, rng=None):
    rng = rng or random.Random(0)
    available = min(len(rows_by_class[c]) for c in TARGET_PROVINCES)
    take = available if max_per_class is None else min(available, max_per_class)
    picked = []
    for c in TARGET_PROVINCES:
        pool = list(rows_by_class[c])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    rng.shuffle(picked)
    return picked, take


def build_general_bank(manifest_path: Path, work_dir: Path, max_train_per_class: int, max_val_per_class: int, seed: int):
    rows, fieldnames = read_manifest_rows(manifest_path)
    train_by_class = defaultdict(list)
    val_by_class = defaultdict(list)
    for row in rows:
        if row.get('family') != 'green8':
            continue
        text = row.get('text', '')
        if not text or text[0] not in TARGET_PROVINCES:
            continue
        split = row.get('split', '')
        if split == 'train':
            train_by_class[text[0]].append(row)
        elif split == 'val':
            val_by_class[text[0]].append(row)
    rng = random.Random(seed)
    train_rows, train_take = balanced_pick(train_by_class, max_train_per_class, rng)
    val_rows, val_take = balanced_pick(val_by_class, max_val_per_class, rng)
    train_manifest = work_dir / 'general_existing_train.csv'
    val_manifest = work_dir / 'general_existing_val.csv'
    write_manifest(train_manifest, train_rows, fieldnames)
    write_manifest(val_manifest, val_rows, fieldnames)
    return {
        'name': 'general_existing',
        'train_manifest': train_manifest,
        'val_manifest': val_manifest,
        'notes': {'train_per_class': train_take, 'val_per_class': val_take, 'source_manifest': str(manifest_path)},
    }


def build_random_split_bank(manifest_path: Path, work_dir: Path, name: str, pattern: str, max_per_class: int, seed: int):
    rows, fieldnames = read_manifest_rows(manifest_path)
    regex = re.compile(pattern)
    by_class = defaultdict(list)
    for row in rows:
        if row.get('family') != 'green8':
            continue
        text = row.get('text', '')
        if not text or text[0] not in TARGET_PROVINCES:
            continue
        if not regex.match(text):
            continue
        by_class[text[0]].append(row)
    rng = random.Random(seed)
    picked, take = balanced_pick(by_class, max_per_class, rng)
    stratified = defaultdict(list)
    for row in picked:
        stratified[row['text'][0]].append(row)
    train_rows = []
    val_rows = []
    for prov in TARGET_PROVINCES:
        cls_rows = stratified[prov]
        rng.shuffle(cls_rows)
        split_idx = max(1, int(len(cls_rows) * 0.8))
        split_idx = min(split_idx, len(cls_rows) - 1) if len(cls_rows) > 1 else 1
        train_rows.extend(cls_rows[:split_idx])
        val_rows.extend(cls_rows[split_idx:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    train_manifest = work_dir / f'{name}_train.csv'
    val_manifest = work_dir / f'{name}_val.csv'
    write_manifest(train_manifest, train_rows, fieldnames)
    write_manifest(val_manifest, val_rows, fieldnames)
    return {
        'name': name,
        'train_manifest': train_manifest,
        'val_manifest': val_manifest,
        'notes': {
            'per_class_total': take,
            'train_count': len(train_rows),
            'val_count': len(val_rows),
            'source_manifest': str(manifest_path),
            'pattern': pattern,
        },
    }


def standardize(train_x, val_x, dump_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return (train_x - mu) / sigma, (val_x - mu) / sigma, (dump_x - mu) / sigma, mu, sigma


def fit_linear_probe(train_x, train_y, val_x, val_y, dump_x, seed: int):
    set_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    dump_t = torch.tensor(dump_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y[:, None], dtype=torch.float32, device=device)
    model = nn.Linear(train_x.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_acc = -1.0
    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits = model(train_t)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_t)).squeeze(1)
            val_pred = (val_prob >= 0.5).long().cpu().numpy()
            val_acc = float((val_pred == val_y).mean())
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(val_t)).squeeze(1).cpu().numpy()
        dump_prob = torch.sigmoid(model(dump_t)).squeeze(1).cpu().numpy()
    val_pred = (val_prob >= 0.5).astype(np.int64)
    dump_pred = (dump_prob >= 0.5).astype(np.int64)
    return {
        'val_acc': float((val_pred == val_y).mean()),
        'val_pred_counts': Counter(LABEL_TO_PROV[int(x)] for x in val_pred),
        'dump_pred_counts': Counter(LABEL_TO_PROV[int(x)] for x in dump_pred),
        'dump_prob_mean_京': float(dump_prob.mean()) if len(dump_prob) else 0.0,
        'dump_prob_std_京': float(dump_prob.std()) if len(dump_prob) else 0.0,
        'dump_pred_labels': dump_pred.tolist(),
        'dump_probabilities_京': dump_prob.tolist(),
    }


def nearest_centroid(train_x, train_y, val_x, val_y, dump_x):
    centroids = {label: train_x[train_y == label].mean(axis=0) for label in np.unique(train_y)}
    for k in centroids:
        norm = np.linalg.norm(centroids[k])
        if norm > 0:
            centroids[k] = centroids[k] / norm
    def classify(x):
        x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-6, None)
        sim_0 = x_norm @ centroids[0]
        sim_1 = x_norm @ centroids[1]
        pred = (sim_1 >= sim_0).astype(np.int64)
        margin = sim_1 - sim_0
        return pred, sim_0, sim_1, margin
    val_pred, _, _, val_margin = classify(val_x)
    dump_pred, dump_sim_0, dump_sim_1, dump_margin = classify(dump_x)
    return {
        'val_acc': float((val_pred == val_y).mean()),
        'val_pred_counts': Counter(LABEL_TO_PROV[int(x)] for x in val_pred),
        'dump_pred_counts': Counter(LABEL_TO_PROV[int(x)] for x in dump_pred),
        'dump_margin_mean_京_minus_皖': float(dump_margin.mean()) if len(dump_margin) else 0.0,
        'dump_margin_std': float(dump_margin.std()) if len(dump_margin) else 0.0,
        'dump_similarity_mean': {
            '皖': float(dump_sim_0.mean()) if len(dump_sim_0) else 0.0,
            '京': float(dump_sim_1.mean()) if len(dump_sim_1) else 0.0,
        },
        'dump_pred_labels': dump_pred.tolist(),
        'dump_margin_values': dump_margin.tolist(),
    }


def knn_neighbors(train_x, train_y, train_texts, dump_x, topk=5):
    train_norm = train_x / np.clip(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-6, None)
    dump_norm = dump_x / np.clip(np.linalg.norm(dump_x, axis=1, keepdims=True), 1e-6, None)
    sims = dump_norm @ train_norm.T
    out = []
    for i in range(sims.shape[0]):
        idx = np.argsort(-sims[i])[:topk]
        neighbors = []
        counts = Counter()
        for j in idx:
            prov = LABEL_TO_PROV[int(train_y[j])]
            counts[prov] += 1
            neighbors.append({'province': prov, 'text': train_texts[j], 'cosine': float(sims[i, j])})
        out.append({'topk': neighbors, 'province_counts': dict(counts)})
    return out


def analyze_bank(net, bank, dump_feats, batch_size, num_workers, device, first_char_time_steps, seed: int):
    train_feats = extract_features(net, bank['train_manifest'], batch_size, num_workers, device, first_char_time_steps)
    val_feats = extract_features(net, bank['val_manifest'], batch_size, num_workers, device, first_char_time_steps)
    results = {'notes': bank['notes'], 'feature_results': {}}
    feature_names = ['pooled_context', 'first_proxy_logits']
    for feat_name in feature_names:
        train_x = train_feats[feat_name]
        val_x = val_feats[feat_name]
        dump_x = dump_feats[feat_name]
        train_y = train_feats['labels']
        val_y = val_feats['labels']
        x_train, x_val, x_dump, _mu, _sigma = standardize(train_x, val_x, dump_x)
        linear = fit_linear_probe(x_train, train_y, x_val, val_y, x_dump, seed)
        centroid = nearest_centroid(x_train, train_y, x_val, val_y, x_dump)
        neighbors = knn_neighbors(x_train, train_y, train_feats['texts'], x_dump, topk=5)
        results['feature_results'][feat_name] = {
            'linear_probe': linear,
            'nearest_centroid': centroid,
            'dump_knn_top5': neighbors,
        }
    return results


def summarize_results(all_results, dump_meta):
    lines = []
    for bank_name, bank_res in all_results.items():
        lines.append(f'[{bank_name}]')
        for feat_name, feat_res in bank_res['feature_results'].items():
            lp = feat_res['linear_probe']
            nc = feat_res['nearest_centroid']
            lines.append(
                f'  - {feat_name}: val_acc(linear)={lp["val_acc"]:.4f}, '
                f'val_acc(centroid)={nc["val_acc"]:.4f}, '
                f'dump_linear={dict(lp["dump_pred_counts"])}, dump_centroid={dict(nc["dump_pred_counts"])}, '
                f'dump_margin_mean(京-皖)={nc["dump_margin_mean_京_minus_皖"]:.4f}, '
                f'dump_prob_mean_京={lp["dump_prob_mean_京"]:.4f}'
            )
        lines.append('')
    lines.append('[dump_samples]')
    for idx, meta in enumerate(dump_meta):
        lines.append(f'  - #{idx:02d} gt={meta.get("gt_text", "")} img={meta.get("_img_path", "")}')
    return '\n'.join(lines).strip() + '\n'


def main():
    ap = argparse.ArgumentParser(description='Analyze whether cluster2 dump features are linearly / metrically separable as 京 vs 皖.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--dump-csv', required=True)
    ap.add_argument('--general-manifest', required=True)
    ap.add_argument('--e20a-manifest', required=True)
    ap.add_argument('--e25a-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--first-char-time-steps', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / 'manifests'
    work_dir.mkdir(parents=True, exist_ok=True)

    dump_manifest = work_dir / 'cluster2_dump_temp.csv'
    dump_meta = build_dump_manifest(Path(args.dump_csv), dump_manifest)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = load_model(Path(args.model), device)
    dump_feats = extract_features(net, dump_manifest, args.batch_size, args.num_workers, device, args.first_char_time_steps)

    banks = [
        build_general_bank(Path(args.general_manifest), work_dir, max_train_per_class=180, max_val_per_class=180, seed=args.seed),
        build_random_split_bank(Path(args.e20a_manifest), work_dir, 'e20a_ad_all', r'^[京皖]AD', max_per_class=223, seed=args.seed),
        build_random_split_bank(Path(args.e25a_manifest), work_dir, 'e25a_ad_all', r'^[京皖]AD', max_per_class=800, seed=args.seed),
        build_random_split_bank(Path(args.e25a_manifest), work_dir, 'e25a_exact_06088', r'^[京皖]AD06088$', max_per_class=300, seed=args.seed),
    ]

    all_results = {}
    for bank in banks:
        all_results[bank['name']] = analyze_bank(
            net=net,
            bank=bank,
            dump_feats=dump_feats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            first_char_time_steps=args.first_char_time_steps,
            seed=args.seed,
        )

    payload = {
        'model': args.model,
        'dump_csv': args.dump_csv,
        'general_manifest': args.general_manifest,
        'e20a_manifest': args.e20a_manifest,
        'e25a_manifest': args.e25a_manifest,
        'dump_sample_count': int(len(dump_meta)),
        'dump_gt_counts': dict(Counter(meta.get('gt_text', '') for meta in dump_meta)),
        'results': all_results,
    }
    json_path = out_dir / 'cluster2_repr_analysis.json'
    md_path = out_dir / 'cluster2_repr_analysis.md'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md_path.write_text(summarize_results(all_results, dump_meta), encoding='utf-8')
    print(str(json_path))
    print(str(md_path))


if __name__ == '__main__':
    main()
