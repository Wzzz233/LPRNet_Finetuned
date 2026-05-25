from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import QuadRefinerDataset, load_jsonl_records
from .losses import QuadRefinerLoss
from .model import QuadHeatmapRefiner


def collate_batch(samples):
    batch = {}
    tensor_keys = ['image', 'heatmaps', 'mask', 'gt_points', 'gt_points_out', 'coarse_points', 'patch_box', 'gt_quad', 'coarse_quad', 'offset_targets', 'offset_masks']
    list_keys = ['sample_id', 'image_path', 'source_name', 'text']
    for key in tensor_keys:
        if key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
    for key in list_keys:
        if key in samples[0]:
            batch[key] = [s[key] for s in samples]
    return batch


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compute_tilt_ratio(quad):
    """Compute perspective tilt ratio from ordered quad (TL, TR, BR, BL)."""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    top_w = float(np.linalg.norm(pts[1] - pts[0]))
    bot_w = float(np.linalg.norm(pts[2] - pts[3]))
    left_h = float(np.linalg.norm(pts[3] - pts[0]))
    right_h = float(np.linalg.norm(pts[2] - pts[1]))
    w_ratio = max(top_w, bot_w) / max(min(top_w, bot_w), 1.0)
    h_ratio = max(left_h, right_h) / max(min(left_h, right_h), 1.0)
    return max(w_ratio, h_ratio)


def compute_sample_weights(records, green_mult: float = 2.0,
                           hard_tilt_threshold: float = 1.10,
                           hard_min_weight_mult: float = 1.0):
    """
    Compute per-sample weight for oversampling:
    - tilt_ratio > 1.05 gets linear boost proportional to deviation
    - green plates get additional multiplier
    - samples with tilt > hard_tilt_threshold get at least hard_min_weight_mult
    """
    weights = np.ones(len(records), dtype=np.float64)
    for i, rec in enumerate(records):
        tilt = compute_tilt_ratio(rec['gt_quad'])
        tilt_dev = max(0.0, tilt - 1.05)
        w = 1.0 + tilt_dev * 20.0  # tilt=1.10→2, tilt=1.20→4
        if rec.get('family', '') == 'green8':
            w *= float(green_mult)
        # Hard-tilt minimum weight guarantee
        if tilt >= float(hard_tilt_threshold):
            w = max(w, float(hard_min_weight_mult))
        weights[i] = max(w, 0.01)
    return weights


def mean_corner_error(pred_points, gt_points):
    return torch.linalg.norm(pred_points - gt_points, dim=-1).mean()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    totals = {'loss': 0.0, 'heatmap': 0.0, 'mask': 0.0, 'coord': 0.0, 'geom': 0.0, 'coarse_err': 0.0, 'pred_err': 0.0}
    count = 0
    with torch.set_grad_enabled(train):
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch['image'])
            loss, stats, pred_points_out = criterion(outputs, batch)
            in_w = batch['image'].shape[-1]
            in_h = batch['image'].shape[-2]
            out_w = outputs['heatmaps'].shape[-1]
            out_h = outputs['heatmaps'].shape[-2]
            sx = (in_w - 1) / max(out_w - 1, 1)
            sy = (in_h - 1) / max(out_h - 1, 1)
            pred_points = pred_points_out.clone()
            pred_points[..., 0] *= sx
            pred_points[..., 1] *= sy
            pred_err = mean_corner_error(pred_points, batch['gt_points'])
            coarse_err = mean_corner_error(batch['coarse_points'], batch['gt_points'])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = batch['image'].shape[0]
            totals['loss'] += float(loss.detach().cpu()) * bs
            totals['heatmap'] += stats['heatmap'] * bs
            totals['mask'] += stats['mask'] * bs
            totals['coord'] += stats['coord'] * bs
            totals['geom'] += stats['geom'] * bs
            totals['coarse_err'] += float(coarse_err.detach().cpu()) * bs
            totals['pred_err'] += float(pred_err.detach().cpu()) * bs
            count += bs
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, history, args):
    payload = {
        'epoch': epoch,
        'best_metric': best_metric,
        'history': history,
        'args': vars(args),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    torch.save(payload, path)


def build_argparser():
    ap = argparse.ArgumentParser(description='Train OBB quad post-refiner.')
    ap.add_argument('--train-jsonl', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--input-width', type=int, default=256)
    ap.add_argument('--input-height', type=int, default=128)
    ap.add_argument('--output-width', type=int, default=64)
    ap.add_argument('--output-height', type=int, default=32)
    ap.add_argument('--pad-x', type=float, default=0.20)
    ap.add_argument('--pad-y', type=float, default=0.25)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--resume', default='')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--weighted-sampler', action='store_true',
                    help='Use tilt-aware weighted sampling (oversample hard tilt + green plates)')
    ap.add_argument('--green-weight-mult', type=float, default=2.0,
                    help='Multiplier for green plate weight (default: 2.0, used with --weighted-sampler)')
    ap.add_argument('--hard-tilt-threshold', type=float, default=1.10,
                    help='Tilt ratio threshold for "hard batch" stratification (default: 1.10)')
    ap.add_argument('--hard-min-weight-mult', type=float, default=1.0,
                    help='Minimum weight multiplier for hard tilt samples (default: 1.0 = no extra boost)')
    ap.add_argument('--enable-offset', action='store_true',
                    help='Enable offset head for sub-pixel refinement (V2b)')
    ap.add_argument('--offset-weight', type=float, default=0.5,
                    help='Weight for offset loss (default: 0.5, used with --enable-offset)')
    ap.add_argument('--warp-weight', type=float, default=0.0,
                    help='Weight for warp-aware loss (default: 0.0, V2c)')
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = QuadRefinerDataset(args.train_jsonl, input_size=(args.input_width, args.input_height),
                                  output_size=(args.output_width, args.output_height), pad_x=args.pad_x, pad_y=args.pad_y,
                                  enable_offset=args.enable_offset)
    val_ds = QuadRefinerDataset(args.val_jsonl, input_size=(args.input_width, args.input_height),
                                output_size=(args.output_width, args.output_height), pad_x=args.pad_x, pad_y=args.pad_y,
                                enable_offset=args.enable_offset)

    if args.weighted_sampler:
        train_records = load_jsonl_records(args.train_jsonl)
        sample_weights = compute_sample_weights(train_records, green_mult=args.green_weight_mult,
                                                hard_tilt_threshold=args.hard_tilt_threshold,
                                                hard_min_weight_mult=args.hard_min_weight_mult)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
        w_min, w_max, w_mean = sample_weights.min(), sample_weights.max(), sample_weights.mean()
        print(json.dumps({'weighted_sampler': True, 'weight_min': float(w_min),
                          'weight_max': float(w_max), 'weight_mean': float(w_mean),
                          'n_records': len(sample_weights)}))
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

    model = QuadHeatmapRefiner(pretrained=not args.no_pretrained, enable_offset=args.enable_offset).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    offset_w = args.offset_weight if args.enable_offset else 0.0
    criterion = QuadRefinerLoss(offset_weight=offset_w, warp_weight=args.warp_weight)

    start_epoch = 0
    best_metric = float('inf')
    history = []
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_metric = float(ckpt.get('best_metric', best_metric))
        history = list(ckpt.get('history', []))

    for epoch in range(start_epoch, args.epochs):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        row = {'epoch': epoch, 'train': train_stats, 'val': val_stats, 'lr': float(optimizer.param_groups[0]['lr'])}
        history.append(row)
        save_checkpoint(out_dir / 'last.pt', model, optimizer, scheduler, epoch, best_metric, history, args)
        if val_stats['pred_err'] < best_metric:
            best_metric = val_stats['pred_err']
            save_checkpoint(out_dir / 'best.pt', model, optimizer, scheduler, epoch, best_metric, history, args)
        print(json.dumps(row, ensure_ascii=False))

    summary = {'best_val_pred_err': best_metric, 'epochs': history}
    (out_dir / 'train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
