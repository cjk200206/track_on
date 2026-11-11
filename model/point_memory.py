"""Per-point history storage and an optional independent RGB reader."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PointMemory:
    values: torch.Tensor
    mask: torch.Tensor
    history: torch.Tensor | None = None

    @classmethod
    def empty(cls, reference, count, size, dim, with_history=False):
        values = reference.new_zeros(count, size, dim, dtype=torch.float32)
        mask = torch.ones(count, size, device=reference.device, dtype=torch.bool)
        history = values.new_zeros(count, size, 4) if with_history else None
        return cls(values, mask, history)

    def select(self, points):
        return PointMemory(self.values[points], self.mask[points],
                           None if self.history is None else self.history[points])

    def append(self, query, entry=None, active=None):
        """Return new storage; retain gradients through queries, never through scores."""
        if active is None:
            active = torch.ones(self.values.shape[0], device=query.device, dtype=torch.bool)
        value = self.values.new_zeros(self.values.shape[0], 1, self.values.shape[-1])
        value[active, 0] = query.to(value.dtype)
        values = torch.cat((self.values[:, 1:], value), dim=1)
        mask = torch.cat((self.mask[:, 1:], ~active[:, None]), dim=1)
        history = None
        if self.history is not None:
            if entry is None:
                raise ValueError("Memory reliability/history requires a matching history entry")
            latest = self.history.new_zeros(self.values.shape[0], 1, 4)
            latest[active, 0] = entry.detach().to(latest.dtype)
            history = torch.cat((self.history[:, 1:], latest), dim=1)
        return PointMemory(values, mask, history)

    def grow(self, count):
        extra = count - self.values.shape[0]
        if extra <= 0:
            return self
        padding = self.empty(self.values, extra, self.values.shape[1],
                             self.values.shape[-1], self.history is not None)
        return PointMemory(torch.cat((self.values, padding.values)),
                           torch.cat((self.mask, padding.mask)),
                           None if self.history is None else torch.cat((self.history, padding.history)))


def resize_time_embedding(embedding, size):
    """Interpolate historical slots, preserving the separate current-query slot."""
    past = F.interpolate(embedding[:, :-1].transpose(1, 2), size=size,
                         mode="linear", align_corners=True).transpose(1, 2)
    return torch.cat((past, embedding[:, -1:]), dim=1)


def read_memory(layer, query, memory, mask, embedding, bias=None, p_mask=0):
    if p_mask:
        mask = mask | (torch.rand_like(mask, dtype=torch.float32) < p_mask)
    tokens = torch.cat((memory, query.transpose(0, 1)), dim=1)
    # Keep the current token available, including for a new point with no history.
    mask = F.pad(mask, (0, 1), value=False)
    tokens = layer(tokens + embedding, tokens + embedding, tokens, mask, attention_bias=bias)
    return tokens[:, -1].unsqueeze(0), tokens[:, :-1]


class RGBMemoryReader(nn.Module):
    def __init__(self, attention, embedding, size, dim):
        super().__init__()
        self.attention = deepcopy(attention)
        self.time_embedding = nn.Parameter(resize_time_embedding(embedding.detach(), size))
        self.fusion = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in attention])
        for layer in self.fusion:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def size(self):
        return self.time_embedding.shape[1] - 1

    def initialize_from(self, attention, embedding):
        self.attention.load_state_dict(attention.state_dict())
        with torch.no_grad():
            self.time_embedding.copy_(resize_time_embedding(embedding, self.size))

    def extend(self, size):
        if size < 1:
            raise ValueError("RGB memory size must be positive")
        self.time_embedding = nn.Parameter(resize_time_embedding(self.time_embedding.detach(), size))

    def read(self, index, query, memory, mask, bias=None, p_mask=0):
        return read_memory(self.attention[index], query, memory, mask,
                           self.time_embedding, bias, p_mask)

    def fuse(self, index, recent, rgb, mask):
        residual = self.fusion[index](torch.cat((recent, rgb), dim=-1))
        available = (~mask).any(-1)[None, :, None]
        return recent + residual * available.to(residual.dtype)
