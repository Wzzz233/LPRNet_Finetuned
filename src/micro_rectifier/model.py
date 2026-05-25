from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_localization_head(in_channels: int, width_mult: float = 1.0):
    c1 = max(8, int(16 * width_mult))
    c2 = max(8, int(24 * width_mult))
    c3 = max(8, int(32 * width_mult))
    return nn.Sequential(
        nn.Conv2d(in_channels, c1, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c1, c2, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c2, c3, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    ), c3


class MicroRectifier(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        self.loc, feat_dim = _make_localization_head(in_channels, width_mult)
        self.fc = nn.Linear(feat_dim, 5)
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32))

    def forward(self, x):
        feat = self.loc(x).flatten(1)
        params = self.fc(feat)
        theta = self._params_to_theta(params)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rectified = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return {'params': params, 'rectified': rectified}

    @staticmethod
    def _params_to_theta(params):
        dx, dy, sx, sy, shx = [params[:, i] for i in range(5)]
        tx = dx / 80.0
        ty = dy / 24.0
        theta = torch.zeros((params.shape[0], 2, 3), dtype=params.dtype, device=params.device)
        theta[:, 0, 0] = sx
        theta[:, 0, 1] = shx
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = sy
        theta[:, 1, 2] = ty
        return theta
