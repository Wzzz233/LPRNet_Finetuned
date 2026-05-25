from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


def soft_argmax_2d(logits):
    b, c, h, w = logits.shape
    flat = logits.view(b, c, -1)
    probs = torch.softmax(flat, dim=-1)
    xs = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_x = grid_x.reshape(1, 1, -1)
    grid_y = grid_y.reshape(1, 1, -1)
    x = (probs * grid_x).sum(dim=-1)
    y = (probs * grid_y).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def polygon_area(points):
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs((x * torch.roll(y, shifts=-1, dims=-1)).sum(dim=-1) -
                           (y * torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1))


def warp_quad_to_rect_diff(image, src_pts, dst_size=(94, 24)):
    """Differentiable bilinear warp from quadrilateral to rectangle.

    Maps each point in the output rectangle back to the corresponding point
    in the source quadrilateral using bilinear interpolation, then samples.

    Args:
        image: (B, 3, H, W) input tensor (256x128 refiner patch)
        src_pts: (B, 4, 2) source quad corners [TL, TR, BR, BL] in input space
        dst_size: (out_w, out_h) output rectangle dimensions

    Returns:
        warped: (B, 3, out_h, out_w) warped output
    """
    B, C, H, W = image.shape
    out_w, out_h = dst_size

    # Output grid coordinates (normalized 0-1)
    ny = torch.linspace(0, 1, out_h, device=image.device, dtype=image.dtype).view(out_h, 1)
    nx = torch.linspace(0, 1, out_w, device=image.device, dtype=image.dtype).view(1, out_w)

    # src_pts: [B, 4, 2] -> individual corners
    TL = src_pts[:, 0:1, :]   # (B, 1, 2)
    TR = src_pts[:, 1:2, :]
    BR = src_pts[:, 2:3, :]
    BL = src_pts[:, 3:4, :]

    # Bilinear interpolation within the quad:
    # top(x,y) = TL + nx * (TR - TL)
    # bottom(x,y) = BL + nx * (BR - BL)
    # point(x,y) = top + ny * (bottom - top)
    top = TL.unsqueeze(1) + (TR - TL).unsqueeze(1) * nx.unsqueeze(0).unsqueeze(-1)  # (B, 1, ow, 2)
    bottom = BL.unsqueeze(1) + (BR - BL).unsqueeze(1) * nx.unsqueeze(0).unsqueeze(-1)  # (B, 1, ow, 2)
    src_coords = top + (bottom - top) * ny.unsqueeze(0).unsqueeze(-1)  # (B, oh, ow, 2)

    # Normalize to [-1, 1] for grid_sample
    grid = src_coords.clone()
    grid[..., 0] = grid[..., 0] / max(W - 1, 1) * 2.0 - 1.0
    grid[..., 1] = grid[..., 1] / max(H - 1, 1) * 2.0 - 1.0

    warped = F.grid_sample(image, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
    return warped


class QuadRefinerLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, mask_weight=0.3, coord_weight=0.5,
                 geom_weight=0.1, offset_weight=0.0, warp_weight=0.0):
        super().__init__()
        self.heatmap_weight = float(heatmap_weight)
        self.mask_weight = float(mask_weight)
        self.coord_weight = float(coord_weight)
        self.geom_weight = float(geom_weight)
        self.offset_weight = float(offset_weight)
        self.warp_weight = float(warp_weight)

    def forward(self, outputs, batch):
        heat_logits = outputs['heatmaps']
        mask_logits = outputs['mask']
        target_heat = batch['heatmaps']
        target_mask = batch['mask']
        gt_points_out = batch['gt_points_out']

        heat_probs = torch.sigmoid(heat_logits)
        heat_loss = F.mse_loss(heat_probs, target_heat)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_dice = dice_loss(mask_logits, target_mask)
        mask_loss = mask_bce + mask_dice

        pred_points_out = soft_argmax_2d(heat_logits)
        coord_loss = F.smooth_l1_loss(pred_points_out, gt_points_out)

        pred_area = polygon_area(pred_points_out)
        gt_area = polygon_area(gt_points_out).detach()
        geom_loss = torch.relu(gt_area * 0.30 - pred_area).mean() / (gt_area.mean() + 1e-6)

        total = (
            self.heatmap_weight * heat_loss
            + self.mask_weight * mask_loss
            + self.coord_weight * coord_loss
            + self.geom_weight * geom_loss
        )

        # Offset loss (only when offset head is active)
        offset_loss = torch.tensor(0.0, device=heat_logits.device)
        if self.offset_weight > 0 and 'offset_targets' in batch and outputs.get('offsets') is not None:
            offset_pred = outputs['offsets']
            offset_target = batch['offset_targets']
            offset_mask = batch['offset_masks']
            off_mask_8 = offset_mask.repeat_interleave(2, dim=1)
            masked_loss = F.smooth_l1_loss(
                offset_pred * off_mask_8,
                offset_target * off_mask_8,
                reduction='sum',
            )
            normalizer = max(off_mask_8.sum(), 1.0)
            offset_loss = masked_loss / normalizer
            total = total + self.offset_weight * offset_loss

        # Warp-aware loss (differentiable bilinear warp)
        warp_loss = torch.tensor(0.0, device=heat_logits.device)
        if self.warp_weight > 0 and 'image' in batch:
            img = batch['image']  # (B, 3, 128, 256)
            B, _, in_h, in_w = img.shape
            out_w_out = pred_points_out.shape[-2]  # 64
            out_h_out = pred_points_out.shape[-3]  # 32  (wait, pred_points is [B, 4, 2] after permute?)
            
            # pred_points_out is (B, 4, 2) in [0, out_w-1] x [0, out_h-1] space
            # Scale to input space [0, in_w-1] x [0, in_h-1]
            sx = (in_w - 1) / max(out_w_out - 1, 1)  # out_w_out = 64
            sy = (in_h - 1) / max(out_h_out - 1, 1)  # out_h_out = 32
            
            pred_pts_in = pred_points_out.clone()
            pred_pts_in[..., 0] *= sx
            pred_pts_in[..., 1] *= sy
            
            gt_pts_in = gt_points_out.clone().detach()  # detach: don't pull GT towards prediction
            gt_pts_in[..., 0] *= sx
            gt_pts_in[..., 1] *= sy

            # Warp both to 94x24 OCR space
            warped_pred = warp_quad_to_rect_diff(img, pred_pts_in, (94, 24))
            warped_gt = warp_quad_to_rect_diff(img, gt_pts_in, (94, 24))

            warp_loss = F.mse_loss(warped_pred, warped_gt)
            total = total + self.warp_weight * warp_loss

        stats = {
            'total': float(total.detach().cpu()),
            'heatmap': float(heat_loss.detach().cpu()),
            'mask': float(mask_loss.detach().cpu()),
            'coord': float(coord_loss.detach().cpu()),
            'geom': float(geom_loss.detach().cpu()),
            'offset': float(offset_loss.detach().cpu()),
            'warp': float(warp_loss.detach().cpu()),
        }
        return total, stats, pred_points_out
