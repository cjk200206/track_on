"""TAPFormer-style content recurrence, optionally fused with RGB warping."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_encoder import build_event_encoder
from model.event_motion import MotionEventFusion, MotionState
from model.modules import MHA_Block


@dataclass
class RecurrentState(MotionState):
    fused: torch.Tensor | None = None


class RecurrentEventFusion(nn.Module):
    """Update one coarse content map and decode four current-time feature scales."""

    def __init__(self, event_channels, dim, encoder=None, with_warp=False, motion_dim=64):
        super().__init__()
        self.event_channels = event_channels
        self.encoder = encoder if encoder is not None else build_event_encoder(event_channels, dim)
        self.layers = nn.ModuleList([
            MHA_Block(dim, 8, attn_drop=0, mlp_drop=0) for _ in range(2)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * dim, dim, 3, padding=1), nn.GELU())
            for _ in range(3)
        ])
        # Only the outer module owns the event encoder and its checkpoint keys.
        self.motion = (MotionEventFusion(event_channels, dim, "motion_state",
                                        motion_dim=motion_dim, encoder=nn.Identity())
                       if with_warp else None)
        self.blend = (nn.ModuleList([nn.Conv2d(3 * dim, 1, 1) for _ in range(4)])
                      if with_warp else None)

    def encode_events(self, events):
        if events.ndim != 4 or events.shape[1] != self.event_channels:
            raise ValueError("events must have shape (B, configured event_channels, H, W)")
        event4 = self.encoder(events)
        return self.motion.encode_motion(events, event4) if self.motion is not None else (event4,)

    def step(self, rgb_features, event_features, img_ifnew, state):
        event4 = event_features[0]
        flags = torch.as_tensor(img_ifnew, device=event4.device, dtype=torch.bool).reshape(-1)
        if flags.shape != (event4.shape[0],):
            raise ValueError("img_ifnew must contain one flag per batch item")
        if state.anchor is None and not flags.all():
            raise ValueError("The first recurrent step must supply a new RGB anchor")
        if state.anchor is not None and state.anchor[0].shape != rgb_features[0].shape:
            raise ValueError("Reset the frontend state before changing the batch or feature size")
        mask = flags[:, None, None, None]
        if self.motion is not None:
            warped = self.motion.step(rgb_features, event_features, flags, state)
        else:
            state.anchor = (rgb_features if state.anchor is None else tuple(
                torch.where(mask, fresh, old) for fresh, old in zip(rgb_features, state.anchor)
            ))

        events = tuple(F.adaptive_avg_pool2d(event4, frame.shape[-2:]) for frame in state.anchor)
        base = (state.anchor[-1] if state.fused is None
                else torch.where(mask, state.anchor[-1], state.fused))
        query = base.flatten(2).transpose(1, 2)
        context = events[-1].flatten(2).transpose(1, 2)
        for layer in self.layers:
            query = layer(query, context, context)
        coarse = query.transpose(1, 2).reshape_as(base)

        features = [coarse]
        for decoder, frame, event in zip(self.decoder, reversed(state.anchor[:-1]), reversed(events[:-1])):
            skip = torch.where(mask, frame, event)
            up = F.interpolate(features[-1], size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features.append(decoder(torch.cat((up, skip), dim=1)))
        features.reverse()
        if self.motion is not None:
            features = [
                recurrent + gate(torch.cat((recurrent, warp, event), dim=1)).sigmoid() * (warp - recurrent)
                for recurrent, warp, event, gate in zip(features, warped, events, self.blend)
            ]
        # Persist final content, including the warp contribution in the hybrid.
        state.fused = features[-1]
        return tuple(features)
