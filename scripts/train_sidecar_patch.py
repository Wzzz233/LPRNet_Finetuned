#!/usr/bin/env python3
"""
Train + Evaluate police province sidecar patch experiments.
Reuses the dataset, model, and eval logic from train_police_province_sidecar.py.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvmodels

ROOT = Path('/home/wzzz/LPRNet')

# ── Dataset ──────────────────────────────────────────────────────────

class ProvinceSidecarDataset(Dataset):
    def __init__(self, manifest_path, preproc='gray3', resize_to=(224, 72),
                 province_map=None):
        self.preproc = preproc
        self.resize_to = resize_to
        self.records = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                p = row.get('path', '').strip()
                if not p:
                    continue
                if not os.path.isabs(p):
                    p = str(ROOT / p)
                if not os.path.exists(p):
                    continue
                self.records.append(row)
        self.province_map = province_map or {}
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec['path']
        if not os.path.isabs(path):
            path = str(ROOT / path)
        
        # Read image — handle PPM from board dump (BGR in raw bytes)
        ext = Path(path).suffix.lower()
        if ext == '.ppm':
            # Board PPM: cv2.imread does RGB→BGR, but data is BGR.
            # Solution: cv2 read then swap back to BGR.
            img = cv2.imread(path)
            if img is not None:
                img = img[:, :, ::-1].copy()  # RGB→BGR
        else:
            img = cv2.imread(path)  # normal JPEG/PNG: BGR
        
        if img is None:
            raise FileNotFoundError(f'Cannot read: {path}')
        
        # Resize
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        
        # Gray3 preproc
        if self.preproc == 'gray3':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # To tensor [0,1]
        img = torch.from_numpy(img.astype('float32') / 255.0)
        img = img.permute(2, 0, 1).contiguous()
        
        label = int(rec.get('label', -1))
        return img, label, rec
    
    def get_province_map(self):
        return self.province_map

def collate_sidecar(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    records = [b[2] for b in batch]
    return images, labels, records

# ── Model ────────────────────────────────────────────────────────────

def build_sidecar(num_classes=31):
    model = tvmodels.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_records = [], [], []
    with torch.no_grad():
        for images, labels, records in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_records.extend(records)
    return {'preds': all_preds, 'labels': all_labels, 'records': all_records}

def compute_province_acc(eval_result, province_map):
    """Compute per-province accuracy from eval result."""
    preds = eval_result['preds']
    labels = eval_result['labels']
    records = eval_result['records']
    
    idx2prov = {v: k for k, v in province_map.items()}
    total, correct = 0, 0
    per_prov_correct = {}
    per_prov_total = {}
    
    for p, l, rec in zip(preds, labels, records):
        prov = rec.get('province', idx2prov.get(l, '?'))
        total += 1
        if p == l:
            correct += 1
            per_prov_correct[prov] = per_prov_correct.get(prov, 0) + 1
        per_prov_total[prov] = per_prov_total.get(prov, 0) + 1
    
    per_prov_acc = {}
    for prov in per_prov_total:
        c = per_prov_correct.get(prov, 0)
        t = per_prov_total[prov]
        per_prov_acc[prov] = c / t if t > 0 else 0
    
    return {
        'overall_acc': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'per_province_acc': per_prov_acc,
        'preds': preds,
        'labels': labels,
    }

# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--val-real', default='', help='real meng holdout csv')
    ap.add_argument('--val-synth', default='', help='synthetic validation csv')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--warm-start', default='police', 
                    choices=['police', 'imagenet', 'random', 'none'])
    args = ap.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Province map
    with open(str(ROOT / 'keys/police_keys.txt')) as f:
        chars = [l.strip() for l in f if l.strip()]
    province_map = {c: i for i, c in enumerate(chars[:31])}
    MENG_IDX = province_map['蒙']
    
    # Datasets
    ds_train = ProvinceSidecarDataset(args.train_manifest, province_map=province_map)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_sidecar)
    
    loaders_eval = {}
    if args.val_real:
        ds_real = ProvinceSidecarDataset(args.val_real, province_map=province_map)
        loaders_eval['real_meng_holdout'] = DataLoader(
            ds_real, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_sidecar)
    if args.val_synth:
        ds_synth = ProvinceSidecarDataset(args.val_synth, province_map=province_map)
        loaders_eval['synth_val_all31'] = DataLoader(
            ds_synth, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_sidecar)
    
    # Model
    model = build_sidecar(num_classes=31)
    
    # Warm start
    if args.warm_start == 'police':
        old_ckpt = ROOT / 'experiments/police_province_sidecar_20260528/resnet18_gray3_pretrained/best.pt'
        if old_ckpt.exists():
            state = torch.load(str(old_ckpt), map_location='cpu')
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f'[WarmStart] Loaded police sidecar from {old_ckpt}')
        else:
            print(f'[WarmStart] Police sidecar not found at {old_ckpt}, using random init')
    elif args.warm_start == 'imagenet':
        model = tvmodels.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 31)
    
    model = model.to(device)
    
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Train log
    log_path = save_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    
    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()
    
    log(f'Config: epochs={args.epochs} lr={args.lr} wd={args.weight_decay} bs={args.batch_size}')
    log(f'Warm start: {args.warm_start}')
    log(f'Train: {len(ds_train)} samples')
    for name, ld in loaders_eval.items():
        log(f'Eval {name}: {len(ld.dataset)} samples')
    log(f'Device: {device}')
    
    best_score = -1.0
    best_state = None
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            preds = logits.argmax(dim=1)
            train_total += labels.numel()
            train_correct += int((preds == labels).sum().item())
            train_loss += float(loss.item()) * labels.numel()
        
        train_acc = train_correct / max(1, train_total)
        train_loss_avg = train_loss / max(1, train_total)
        
        # Evaluate
        epoch_results = {'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss_avg}
        for name, ld in loaders_eval.items():
            eval_res = evaluate(model, ld, device)
            stats = compute_province_acc(eval_res, province_map)
            epoch_results[name + '_acc'] = stats['overall_acc']
            epoch_results[name + '_correct'] = stats['correct']
            epoch_results[name + '_total'] = stats['total']
            
            # Real meng holdout: top1 meng
            if 'real' in name:
                meng_correct = sum(1 for p, l in zip(eval_res['preds'], eval_res['labels'])
                                   if l == MENG_IDX and p == MENG_IDX)
                meng_total = sum(1 for l in eval_res['labels'] if l == MENG_IDX)
                epoch_results['real_meng_top1'] = meng_correct / max(1, meng_total)
                epoch_results['real_meng_top1_count'] = meng_correct
                epoch_results['real_meng_total'] = meng_total
                
                # Top3 recall for meng
                model.eval()
                meng_rank3 = 0
                with torch.no_grad():
                    for images, labels, _ in ld:
                        images = images.to(device)
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                        top3 = probs.topk(3, dim=1).indices
                        for i, l in enumerate(labels):
                            if l == MENG_IDX and MENG_IDX in top3[i]:
                                meng_rank3 += 1
                epoch_results['real_meng_top3_recall'] = meng_rank3 / max(1, meng_total)
            
            # Synth val: non-meng -> meng error rate
            if 'synth' in name:
                non_meng_to_meng = sum(1 for p, l, rec in 
                    zip(eval_res['preds'], eval_res['labels'], eval_res['records'])
                    if p == MENG_IDX and l != MENG_IDX)
                non_meng_total = sum(1 for l in eval_res['labels'] if l != MENG_IDX)
                epoch_results['non_meng_to_meng_rate'] = non_meng_to_meng / max(1, non_meng_total)
                epoch_results['non_meng_to_meng_count'] = non_meng_to_meng
        
        # Combined score for selection
        real_score = epoch_results.get('real_meng_top1', 0)
        synth_score = epoch_results.get('synth_val_all31_acc', 0)
        combined = real_score * 0.6 + synth_score * 0.4
        epoch_results['combined_score'] = combined
        
        # Log
        log_parts = [f'Epoch {epoch:2d}/{args.epochs} | train_acc={train_acc:.4f} loss={train_loss_avg:.4f}']
        for k, v in epoch_results.items():
            if k not in ('epoch',):
                if isinstance(v, float):
                    log_parts.append(f'{k}={v:.4f}')
                else:
                    log_parts.append(f'{k}={v}')
        log(' | '.join(log_parts))
        
        # Save if best
        if combined > best_score:
            best_score = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            torch.save(best_state, str(save_dir / 'best.pt'))
            log(f'  -> New best (score={combined:.4f})')
    
    log(f'\nTraining complete. Best epoch={best_epoch}, score={best_score:.4f}')
    log_f.close()
    
    # Final eval with best model
    # Use a separate eval report mechanism
    print('\nFinal evaluation with best model')
    sys.stdout.flush()
    
    # Reload model for final eval
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    
    for name, ld in loaders_eval.items():
        eval_res = evaluate(model, ld, device)
        stats = compute_province_acc(eval_res, province_map)
        
        # Write predictions
        pred_csv = save_dir / f'predictions_{name}.csv'
        with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['path', 'gt_label', 'gt_prov', 'pred_label', 'pred_prov', 'correct'])
            for p, l, rec in zip(eval_res['preds'], eval_res['labels'], eval_res['records']):
                gt_prov = rec.get('province', province_map.get(l, '?'))
                pred_prov = province_map.get(p, '?')
                w.writerow([rec.get('path', ''), l, gt_prov, p, pred_prov, 'YES' if p == l else 'NO'])
        
        # Write JSON metrics
        metrics = {
            'overall_acc': stats['overall_acc'],
            'correct': stats['correct'],
            'total': stats['total'],
            'per_province_acc': stats['per_province_acc'],
        }
        
        if 'real' in name:
            meng_correct = sum(1 for p, l in zip(eval_res['preds'], eval_res['labels'])
                               if l == MENG_IDX and p == MENG_IDX)
            meng_total = sum(1 for l in eval_res['labels'] if l == MENG_IDX)
            metrics['meng_top1'] = meng_correct / max(1, meng_total)
            metrics['meng_top1_count'] = meng_correct
            metrics['meng_total'] = meng_total
            
            # Top3 recall
            meng_rank3 = 0
            model.eval()
            with torch.no_grad():
                for images, labels, _ in ld:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    top3 = probs.topk(3, dim=1).indices
                    for i, l in enumerate(labels):
                        if l == MENG_IDX and MENG_IDX in top3[i]:
                            meng_rank3 += 1
            metrics['meng_top3_recall'] = meng_rank3 / max(1, meng_total)
            
            # Top1 distribution
            from collections import Counter
            pred_dist = Counter(eval_res['preds'])
            metrics['pred_distribution'] = {province_map.get(k, str(k)): v for k, v in pred_dist.most_common()}
            
            # Average meng rank
            meng_ranks = []
            model.eval()
            with torch.no_grad():
                for images, labels, _ in ld:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    for i, l in enumerate(labels):
                        if l == MENG_IDX:
                            sorted_idx = torch.argsort(probs[i], descending=True)
                            rank = int((sorted_idx == MENG_IDX).nonzero(as_tuple=True)[0].item()) + 1
                            meng_ranks.append(rank)
            metrics['meng_avg_rank'] = sum(meng_ranks) / max(1, len(meng_ranks)) if meng_ranks else -1
        
        if 'synth' in name:
            non_meng_to_meng = sum(1 for p, l in zip(eval_res['preds'], eval_res['labels'])
                                   if p == MENG_IDX and l != MENG_IDX)
            non_meng_total = sum(1 for l in eval_res['labels'] if l != MENG_IDX)
            metrics['non_meng_to_meng_rate'] = non_meng_to_meng / max(1, non_meng_total)
            metrics['non_meng_to_meng_count'] = non_meng_to_meng
            
            # Specific confusion pairs
            confusion_pairs = {}
            for p, l in zip(eval_res['preds'], eval_res['labels']):
                if p != l:
                    pair = f'{province_map.get(l, str(l))}->{province_map.get(p, str(p))}'
                    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            metrics['confusion_pairs'] = confusion_pairs
        
        json_path = save_dir / f'eval_{name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        log(f'{name}: acc={stats["overall_acc"]:.4f} ({stats["correct"]}/{stats["total"]})')
        if 'real' in name:
            log(f'  meng top1={metrics["meng_top1"]:.4f} top3={metrics["meng_top3_recall"]:.4f} avg_rank={metrics["meng_avg_rank"]:.1f}')
        if 'synth' in name:
            log(f'  non-meng→meng rate={metrics["non_meng_to_meng_rate"]:.4f}')
            log(f'  confusion: {json.dumps(confusion_pairs, ensure_ascii=False)}')
    
    log(f'\nResults saved to {save_dir}')

if __name__ == '__main__':
    main()
