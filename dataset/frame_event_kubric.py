"""Paired Kubric RGB/time-surface clips for the frame+event experiment."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FrameEventKubric(Dataset):
    """One event interval per step, aligned with annotation time index i + 1."""

    def __init__(
        self,
        root,
        clip_length=24,
        num_points=128,
        crop_size=(384, 512),
        frame_interval=4,
        max_sequences=None,
        training=False,
        seed=1234,
        eval_start=0,
        sequence_paths=None,
        anchor_aligned=False,
    ):
        self.root = Path(root)
        self.clip_length = clip_length
        self.num_points = num_points
        self.crop_size = tuple(crop_size)
        self.frame_interval = frame_interval
        self.training = training
        self.seed = seed
        self.eval_start = eval_start
        self.anchor_aligned = anchor_aligned
        if sequence_paths is None and not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if clip_length < 2 or num_points < 1 or frame_interval < 1 or eval_start < 0:
            raise ValueError("clip_length must be >= 2, num_points and frame_interval positive, eval_start non-negative")
        if anchor_aligned and eval_start % frame_interval:
            raise ValueError("Motion clips must start at a real RGB anchor")
        self.sequences = (list(sequence_paths) if sequence_paths is not None else
                          sorted(p for p in self.root.iterdir() if p.is_dir()))
        if max_sequences is not None:
            self.sequences = self.sequences[:max_sequences]
        if not self.sequences:
            raise ValueError(f"No sequences found in {self.root}")

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def _read_event(path):
        try:
            import h5py
            import hdf5plugin  # noqa: F401 - registers the Blosc HDF5 filter
        except ImportError as exc:
            raise ImportError("Reading time surfaces requires h5py and hdf5plugin") from exc
        with h5py.File(path, "r") as handle:
            return handle["time_surface"][:]

    def __getitem__(self, index):
        sequence = self.sequences[index]
        annotations = np.load(sequence / "annotations.npy", allow_pickle=True).item()
        all_tracks = annotations["target_points"]
        all_occluded = annotations["occluded"]
        event_dir = sequence / "events" / "time_surfaces_v2_5"
        num_intervals = min(len(list(event_dir.glob("*.h5"))), all_tracks.shape[1] - 1)
        if num_intervals < self.clip_length:
            raise ValueError(f"{sequence}: fewer than {self.clip_length} aligned event intervals")

        with Image.open(sequence / "raw" / "rgba_blur_00001.png") as image:
            width, height = image.size
        crop_h, crop_w = self.crop_size
        if crop_h > height or crop_w > width:
            raise ValueError(f"{sequence}: crop {self.crop_size} exceeds {(height, width)}")
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        possible_starts = num_intervals - self.clip_length + 1
        stride = self.frame_interval if self.anchor_aligned else 1
        candidates_starts = np.arange(0, possible_starts, stride)
        starts = np.random.permutation(candidates_starts) if self.training else [self.eval_start]
        for start in starts:
            if start + self.clip_length > num_intervals:
                raise ValueError(f"{sequence}: clip {start}:{start + self.clip_length} exceeds {num_intervals} intervals")
            interval_indices = np.arange(start, start + self.clip_length)
            annotation_indices = interval_indices + 1
            tracks = all_tracks[:, annotation_indices].transpose(1, 0, 2).copy()
            tracks[..., 0] -= x0
            tracks[..., 1] -= y0
            visible = ~all_occluded[:, annotation_indices].T
            visible &= (tracks[..., 0] >= 0) & (tracks[..., 0] < crop_w)
            visible &= (tracks[..., 1] >= 0) & (tracks[..., 1] < crop_h)
            first_visible = visible.argmax(axis=0)
            later_visible = (visible & (np.arange(self.clip_length)[:, None] > first_visible)).any(axis=0)
            candidates = np.flatnonzero(visible.any(axis=0) & later_visible)
            if len(candidates) >= self.num_points:
                break
        else:
            raise ValueError(f"{sequence}: no {self.clip_length}-step window has {self.num_points} usable points")

        rng = np.random if self.training else np.random.default_rng(self.seed + index)
        selected = rng.choice(candidates, self.num_points, replace=False)
        frames = []
        event_maps = []
        new_frame = []
        previous_rgb_index = None
        for interval, annotation_index in zip(interval_indices, annotation_indices):
            # A new RGB observation arrives every frame_interval event steps.
            rgb_index = 1 + (int(interval) // self.frame_interval) * self.frame_interval
            if rgb_index != previous_rgb_index:
                rgb_path = sequence / "raw" / f"rgba_blur_{rgb_index:05d}.png"
                with Image.open(rgb_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                previous_rgb_index = rgb_index
            frames.append(rgb)
            event_path = event_dir / f"{int(annotation_index):03d}.h5"
            event_maps.append(self._read_event(event_path))
            new_frame.append(interval % self.frame_interval == 0)

        video = np.stack(frames)
        events = np.stack(event_maps).astype(np.float32, copy=False)
        if events.shape[-1] != 10 or video.shape[1:3] != events.shape[1:3]:
            raise ValueError(f"{sequence}: expected aligned RGB and 10-channel time surfaces")
        video = video[:, y0:y0 + crop_h, x0:x0 + crop_w]
        events = events[:, y0:y0 + crop_h, x0:x0 + crop_w]

        return {
            "video": torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))).float(),
            "events": torch.from_numpy(np.ascontiguousarray(events.transpose(0, 3, 1, 2))),
            "tracks": torch.from_numpy(np.ascontiguousarray(tracks[:, selected])).float(),
            "visibility": torch.from_numpy(np.ascontiguousarray(visible[:, selected])),
            "img_ifnew": torch.tensor(new_frame, dtype=torch.bool),
            "frame_indices": torch.from_numpy(annotation_indices.copy()),
        }
