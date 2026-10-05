# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 F. Hamann (TU Berlin).
# Adapted from tub-rip/ETAP, src/model/etap/core/cotracker/blocks.py.
# Licensed under CC BY-NC 4.0; see licenses/ETAP.txt.
# Changes: fixed the official fnet dimensions, removed unused normalization
# options, and added a TrackOn projection and pretrained-loading interface.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1:
            self.norm3 = nn.InstanceNorm2d(planes)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride), self.norm3,
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        return self.relu(residual + y)


class ETAPFeatureEncoder(nn.Module):
    """Official 128-channel fnet with multiscale fusion at stride 4."""

    def __init__(self, event_channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(256)
        self.conv1 = nn.Conv2d(event_channels, 64, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 96, stride=2)
        self.layer3 = self._make_layer(96, 128, stride=2)
        self.layer4 = self._make_layer(128, 128, stride=2)
        self.conv2 = nn.Conv2d(416, 256, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 128, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _make_layer(in_planes, planes, stride):
        return nn.Sequential(ResidualBlock(in_planes, planes, stride), ResidualBlock(planes, planes))

    def forward(self, events):
        height, width = events.shape[-2:]
        x = self.relu1(self.norm1(self.conv1(events)))
        features = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            features.append(F.interpolate(x, size=(height // 4, width // 4), mode="bilinear", align_corners=True))
        x = self.relu2(self.norm2(self.conv2(torch.cat(features, dim=1))))
        return self.conv3(x)


class ETAPEventEncoder(nn.Module):
    """Pretrained ETAP fnet followed by a trainable 128 -> D projection."""

    def __init__(self, event_channels, dim):
        super().__init__()
        self.fnet = ETAPFeatureEncoder(event_channels)
        self.projection = nn.Conv2d(128, dim, 1)

    def load_pretrained(self, path):
        """Load official ETAP weights or a bare fnet state dict, strictly."""
        weights = torch.load(path, map_location="cpu", weights_only=True)
        weights = weights.get("state_dict", weights.get("model", weights))
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        fnet_weights = {key.removeprefix("fnet."): value for key, value in weights.items() if key.startswith("fnet.")}
        self.fnet.load_state_dict(fnet_weights or weights, strict=True)

    def forward(self, events):
        return self.projection(self.fnet(events))
