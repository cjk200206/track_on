"""Causal trajectory-conditioned patch scores, without explicit timestamps."""

import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryHistoryScorer(nn.Module):
    def __init__(self, dim, input_size, history_size=8, hidden_dim=32, max_bias=0.5):
        super().__init__()
        if history_size < 1 or max_bias <= 0:
            raise ValueError("history_size and max_bias must be positive")
        self.history_size = history_size
        self.max_bias = max_bias
        # Ordered slots: relative xy, adjacent displacement, visibility,
        # uncertainty, and validity. Slot order supplies relative frame order.
        self.history_encoder = nn.Sequential(
            nn.Linear(history_size * 7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.query_projections = nn.ModuleList([nn.Linear(dim, hidden_dim) for _ in range(2)])
        self.score_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))
            for _ in range(2)
        ])
        for head in self.score_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        height, width = input_size
        yy, xx = torch.meshgrid((torch.arange(height // 4) * 4 + 2) / height,
                                (torch.arange(width // 4) * 4 + 2) / width, indexing="ij")
        basis = torch.stack((torch.ones_like(xx), xx, yy, xx.square(), yy.square(), xx * yy))
        self.register_buffer("patch_basis", basis.flatten(1), persistent=False)

    def forward(self, correlation, query, history, mask, stage):
        history = history[:, -self.history_size:].detach().float()
        valid = ~mask[:, -self.history_size:]
        if history.shape[1] != self.history_size:
            raise ValueError("Point memory must contain at least history_size slots")
        positions = history[..., :2]
        last = positions[:, -1]
        displacement = torch.zeros_like(positions)
        displacement[:, 1:] = (positions[:, 1:] - positions[:, :-1]) * (
            valid[:, 1:] & valid[:, :-1])[..., None]
        encoded = torch.cat(((positions - last[:, None]) * valid[..., None], displacement,
                             history[..., 2:] * valid[..., None], valid[..., None]), dim=-1)
        context = self.history_encoder(encoded.flatten(1))
        current = self.query_projections[stage](F.layer_norm(query[0], (query.shape[-1],)))
        coefficients = self.score_heads[stage](torch.cat((context, current), dim=-1)).float()
        # A learned quadratic in patch-minus-last-position coordinates. Expand
        # algebraically to avoid materializing an N x patches x hidden tensor.
        a, bx, by, bxx, byy, bxy = coefficients.unbind(-1)
        px, py = last.unbind(-1)
        weights = torch.stack((a - bx * px - by * py + bxx * px.square()
                               + byy * py.square() + bxy * px * py,
                               bx - 2 * bxx * px - bxy * py,
                               by - 2 * byy * py - bxy * px, bxx, byy, bxy), dim=-1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            bias = self.max_bias * torch.tanh(weights @ self.patch_basis.float())
        bias = bias * valid.any(-1, keepdim=True)
        return correlation + bias[None].to(correlation.dtype)


def trajectory_entry(position, visibility, uncertainty, size):
    """xy normalized by width/height, followed by continuous probabilities."""
    height, width = size
    scale = position.new_tensor([width, height])
    return torch.cat((position / scale, visibility.sigmoid()[..., None],
                      uncertainty.sigmoid()[..., None]), dim=-1).detach()
