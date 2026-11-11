"""Optional inference policy: recent updates plus older RGB-time queries."""

from collections import deque

import torch

from model.trackon_predictor import Predictor


class RGBMemoryPredictor(Predictor):
    """Read [older RGB entries, recent entries], oldest first in each queue."""

    def __init__(self, model_args=None, checkpoint_path=None, support_grid_size=20,
                 recent_size=24, rgb_size=48):
        if type(recent_size) is not int or recent_size < 1:
            raise ValueError("recent_size must be a positive integer")
        if type(rgb_size) is not int or rgb_size < 0:
            raise ValueError("rgb_size must be a nonnegative integer")
        self.recent_size = recent_size
        self.rgb_size = rgb_size
        super().__init__(model_args, checkpoint_path, support_grid_size)
        if self.model.rgb_memory_size:
            raise ValueError("The legacy recent_rgb policy cannot wrap a dual-memory model")
        if self.model.M != recent_size + rgb_size:
            raise ValueError("recent_size + rgb_size must equal inference memory size")

    def reset(self):
        super().reset()
        self.recent_rgb = deque([False] * self.recent_size, maxlen=self.recent_size)

    @torch.no_grad()
    def forward_frame(self, frame, new_queries=None, events=None, img_ifnew=None):
        if events is not None and img_ifnew is None:
            raise ValueError("Event memory policy requires explicit img_ifnew")
        new_rgb = events is None if img_ifnew is None else bool(img_ifnew)
        saved = {}
        count = self.N
        if self.rgb_size and not self.recent_rgb[0]:
            for name in ("point_memory", "temporal_mask", "trajectory_history"):
                value = getattr(self, name)
                if value is not None:
                    saved[name] = value[:count, :self.rgb_size].clone()
        previous_t = self.t
        result = super().forward_frame(frame, new_queries, events, img_ifnew)
        # The original roll promotes the oldest recent entry into RGB memory.
        # Undo only that queue's update when the evicted entry is not from RGB.
        for name, value in saved.items():
            getattr(self, name)[:count, :self.rgb_size].copy_(value)
        if self.t != previous_t:
            self.recent_rgb.append(new_rgb)
        return result
