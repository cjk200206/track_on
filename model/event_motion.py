"""Causal event motion frontends with an RGB anchor and caller-owned state."""

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_fusion import EventFeatureFusion


@dataclass
class MotionEntry:
    feature: torch.Tensor
    warp: torch.Tensor
    valid: torch.Tensor


@dataclass
class MotionState:
    """Separate from TrackOn's point memory; never saved in a checkpoint."""

    anchor: tuple | None = None
    warp: torch.Tensor | None = None
    active: torch.Tensor | None = None
    history: list = field(default_factory=list)


def resize_warp(warp, size):
    """Resize a target-to-source displacement measured in feature pixels."""
    height, width = warp.shape[-2:]
    scale = warp.new_tensor([size[1] / width, size[0] / height])[None, :, None, None]
    return F.interpolate(warp, size=size, mode="bilinear", align_corners=False) * scale


def warp_features(feature, warp):
    """Backward sampling on pixel centers, matching align_corners=False."""
    height, width = feature.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=torch.float32),
        torch.arange(width, device=feature.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1)[None] + warp.float().permute(0, 2, 3, 1)
    scale = grid.new_tensor([width, height])
    grid = (grid + 0.5) * (2.0 / scale) - 1.0
    sampled = F.grid_sample(feature.float(), grid, padding_mode="border", align_corners=False)
    return sampled.to(feature.dtype)


class MotionReader(nn.Module):
    """One learned spatial sample per history entry, followed by local scoring."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(2 * dim + 2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, current, entry):
        offset = self.offset(torch.cat((current, entry.feature, entry.warp), dim=1))
        aligned = warp_features(entry.feature, offset)
        score = (self.query(current).float() * self.key(aligned).float()).sum(1, keepdim=True)
        score = score / math.sqrt(current.shape[1])
        return aligned, score


class MotionEventFusion(EventFeatureFusion):
    """Current-event residual plus an RGB anchor warped to the current grid.

    motion_pool stores independently encoded event features; motion_state stores
    one updated motion feature. Both predict an anchor-relative field directly.
    """

    def __init__(self, event_channels, dim, mode, motion_dim=64, history_size=8, encoder=None):
        super().__init__(event_channels, dim, encoder=encoder)
        if mode not in {"motion_pool", "motion_state"}:
            raise ValueError(f"Unknown motion fusion mode: {mode}")
        if motion_dim < 1 or history_size < 1:
            raise ValueError("motion_dim and history_size must be positive")
        self.mode = mode
        self.history_size = history_size if mode == "motion_pool" else 1
        self.motion_encoder = nn.Sequential(nn.Conv2d(dim, motion_dim, 3, stride=2, padding=1), nn.GELU())
        self.frame_projection = nn.Conv2d(dim, motion_dim, 1)
        self.reader = MotionReader(motion_dim)
        self.history_gate = nn.Conv2d(2 * motion_dim, motion_dim, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(2 * motion_dim + 2, motion_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(motion_dim, motion_dim, 3, padding=1),
            nn.GELU(),
        )
        self.warp_head = nn.Conv2d(motion_dim, 2, 3, padding=1)
        self.fusion_gate = nn.Conv2d(dim + motion_dim, 1, 1)
        nn.init.zeros_(self.warp_head.weight)
        nn.init.zeros_(self.warp_head.bias)

    def encode_events(self, events):
        if events.ndim != 4 or events.shape[1] != self.event_channels:
            raise ValueError("events must have shape (B, configured event_channels, H, W)")
        event4 = self.encoder(events)
        motion8 = self.motion_encoder(event4)
        activity = events.ne(0).any(dim=1, keepdim=True).to(event4.dtype)
        activity = F.adaptive_max_pool2d(activity, event4.shape[-2:])
        return event4, motion8, activity

    def _reset_anchor(self, rgb_features, current, flags, state):
        if state.anchor is None:
            if not flags.all():
                raise ValueError("The first motion step must supply a new RGB anchor")
            state.anchor = rgb_features
            state.warp = current.new_zeros(current.shape[0], 2, *current.shape[-2:])
            state.active = torch.zeros_like(flags)
            return
        if state.anchor[0].shape != rgb_features[0].shape:
            raise ValueError("Reset the frontend state before changing the batch or feature size")
        mask = flags[:, None, None, None]
        state.anchor = tuple(torch.where(mask, fresh, old) for fresh, old in zip(rgb_features, state.anchor))
        state.warp = torch.where(mask, torch.zeros_like(state.warp), state.warp)
        state.active = state.active & ~flags
        state.history = [MotionEntry(item.feature, item.warp, item.valid & ~flags) for item in state.history]
        if flags.all():
            state.history.clear()

    def _history_context(self, current, history):
        if not history:
            return torch.zeros_like(current)
        if self.mode == "motion_state":
            aligned, score = self.reader(current, history[-1])
            aligned = aligned * score.sigmoid().to(aligned.dtype)
            return torch.where(history[-1].valid[:, None, None, None], aligned, torch.zeros_like(aligned))

        # A null entry permits ignoring history and prevents all-masked softmax.
        values = [torch.zeros_like(current)]
        scores = [current.new_zeros(current.shape[0], 1, *current.shape[-2:], dtype=torch.float32)]
        for item in history:
            aligned, score = self.reader(current, item)
            score = score.masked_fill(~item.valid[:, None, None, None], float("-inf"))
            values.append(aligned)
            scores.append(score)
        weights = torch.softmax(torch.stack(scores, dim=1), dim=1).to(current.dtype)
        return (weights * torch.stack(values, dim=1)).sum(dim=1)

    def _fuse(self, anchor, event4, motion, warp, activity):
        features = []
        for i, frame in enumerate(anchor):
            size = frame.shape[-2:]
            event = F.adaptive_avg_pool2d(event4, size)
            context = F.interpolate(motion, size=size, mode="bilinear", align_corners=False)
            gate = self.fusion_gate(torch.cat((event, context), dim=1)).sigmoid()
            gate = gate * F.adaptive_max_pool2d(activity, size)
            aligned = warp_features(frame, resize_warp(warp, size))
            features.append(aligned + self.gates[i] * gate * event)
        return tuple(features)

    def step(self, rgb_features, event_features, img_ifnew, state):
        """Process one causal step; only already processed entries are readable."""
        event4, current, activity = event_features
        flags = torch.as_tensor(img_ifnew, device=current.device, dtype=torch.bool).reshape(-1)
        if flags.shape != (current.shape[0],):
            raise ValueError("img_ifnew must contain one flag per batch item")
        self._reset_anchor(rgb_features, current, flags, state)
        history = self._history_context(current, state.history)
        gate = self.history_gate(torch.cat((current, history), dim=1)).sigmoid()
        frame = self.frame_projection(state.anchor[1])
        motion = self.update(torch.cat((current + gate * history, frame, state.warp), dim=1))
        state.active = state.active | activity.flatten(1).any(dim=1)
        warp = self.warp_head(motion) * state.active[:, None, None, None]
        features = self._fuse(state.anchor, event4, motion, warp, activity)
        cached = current if self.mode == "motion_pool" else motion
        state.history.append(MotionEntry(cached, warp, torch.ones_like(flags)))
        state.history = state.history[-self.history_size:]
        state.warp = warp
        return features
