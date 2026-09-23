"""Event feature extraction and residual fusion for the frame+event experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventFeatureFusion(nn.Module):
    """Add a time surface to the four projected RGB feature scales."""

    def __init__(self, event_channels, dim):
        super().__init__()
        self.event_channels = event_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(event_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        # Zero gates preserve the pretrained RGB model before event training.
        self.gates = nn.Parameter(torch.zeros(4))

    def forward(self, rgb_features, events):
        if events.ndim != 5:
            raise ValueError("events must have shape (B, T, C, H, W)")
        batch, time, channels, height, width = events.shape
        first = rgb_features[0]
        if channels != self.event_channels or batch * time != first.shape[0]:
            raise ValueError("events must match the RGB feature batch and configured channels")
        if (height, width) != (first.shape[-2] * 4, first.shape[-1] * 4):
            raise ValueError("events must match the RGB input resolution")

        event4 = self.encoder(events.reshape(batch * time, channels, height, width))
        return tuple(
            feature + self.gates[i] * F.adaptive_avg_pool2d(event4, feature.shape[-2:])
            for i, feature in enumerate(rgb_features)
        )
