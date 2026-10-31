"""Score refined candidates using appearance correlation and ordered history."""

import torch
from torch import nn
import torch.nn.functional as F


class CandidateHistoryScorer(nn.Module):
    def __init__(self, dim, input_size, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        if hidden_dim < 1 or hidden_dim % num_heads or num_layers < 1:
            raise ValueError("Candidate scorer needs positive layers and a dimension divisible by num_heads")
        self.input_size = input_size
        self.embedding = nn.Linear(6, hidden_dim)
        self.position_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim,
                                       dropout=0.1, activation="gelu", batch_first=True)
            for _ in range(num_layers)
        ])
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, positions, features, memory, history, mask, time_embedding):
        # positions: (N, K, 2), memory: (N, M, D), history: normalized xy/v/u.
        # Only xy and appearance correlation enter the scorer; v/u are not inputs.
        height, width = self.input_size
        count, candidates, _ = positions.shape
        slots = memory.shape[1]
        valid = ~mask
        positions = positions.detach().float() / positions.new_tensor([width, height])
        grid = (positions * 2 - 1).unsqueeze(0)
        memory = F.normalize(memory.float().masked_fill(mask[..., None], 0), dim=-1)
        correlations = []
        for feature, stride in zip(features, (4, 8, 16, 32)):
            feature = feature.float().transpose(1, 2).reshape(1, -1, height // stride, width // stride)
            sampled = F.grid_sample(feature, grid, padding_mode="border", align_corners=False)
            sampled = F.normalize(sampled[0].permute(1, 2, 0), dim=-1)
            correlations.append(torch.einsum("nkd,nmd->nkm", sampled, memory))
        correlation = torch.stack(correlations, dim=-1)
        past_positions = history[..., :2].detach().float().masked_fill(mask[..., None], 0)
        displacement = positions[:, :, None] - past_positions[:, None]
        tokens = torch.cat((correlation, displacement), dim=-1)
        tokens = tokens.masked_fill(mask[:, None, :, None], 0)
        tokens = self.embedding(tokens) + self.position_projection(time_embedding[:, :slots])[:, None]
        tokens = tokens.reshape(count * candidates, slots, -1)
        padding = mask[:, None].expand(-1, candidates, -1).reshape(count * candidates, slots).clone()
        # An unused dummy slot keeps attention finite for targets without history.
        padding[padding.all(-1), -1] = False
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=padding)
        tokens = tokens.reshape(count, candidates, slots, -1)
        pooled = tokens.masked_fill(mask[:, None, :, None], 0).sum(2)
        pooled = pooled / valid.sum(-1).clamp_min(1)[:, None, None]
        return self.score(pooled).squeeze(-1)
