"""Event encoders share the (B, C, H, W) -> (B, D, H/4, W/4) interface."""

import torch.nn as nn

from model.etap_encoder import ETAPEventEncoder


def build_event_encoder(event_channels, dim, name="simple", pretrained_path=None):
    """Add future backbone adapters here; keep fusion and tracking unchanged."""
    if name == "etap":
        encoder = ETAPEventEncoder(event_channels, dim)
        if pretrained_path is not None:
            encoder.load_pretrained(pretrained_path)
        return encoder
    if name != "simple":
        raise ValueError(f"Unknown event_encoder: {name}")
    if pretrained_path is not None:
        raise ValueError("event_pretrained_path requires event_encoder: etap")
    # Keep the original module layout and checkpoint keys.
    return nn.Sequential(
        nn.Conv2d(event_channels, 64, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
    )
