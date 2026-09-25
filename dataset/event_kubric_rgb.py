"""RGB-only Track-On2 training on ETAP/TAPFormer EventKubric sequences."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, get_worker_info

from dataset.movi_f import Movi_F_Base


def sequence_dirs(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(Path(entry.path) for entry in os.scandir(root) if entry.is_dir())


def split_source(paths, fraction, seed):
    """Stable sequence-level split, independent of filesystem listing order."""
    if not 0 <= fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    paths = list(paths)
    if fraction == 0:
        return paths, []
    if len(paths) < 2:
        raise ValueError("A source needs at least two sequences for a train/val split")
    # Shared sequence IDs across variants must receive the same split.
    scores = [int.from_bytes(hashlib.sha256(f"{seed}:{path.name}".encode()).digest()[:8], "big") / 2**64
              for path in paths]
    val_ids = {i for i, score in enumerate(scores) if score < fraction}
    if not val_ids:
        val_ids.add(int(np.argmin(scores)))
    if len(val_ids) == len(paths):
        val_ids.remove(int(np.argmax(scores)))
    return [path for i, path in enumerate(paths) if i not in val_ids], [path for i, path in enumerate(paths) if i in val_ids]


def fixed_eval_subset(paths, limit, seed):
    if limit < 1:
        raise ValueError("Validation sequence limit must be positive")
    return sorted(paths, key=lambda path: hashlib.sha256(
        f"{seed}:eval:{path.parent.name}:{path.name}".encode()).digest())[:limit]


def build_sources(args):
    root = Path(args.event_kubric_root)
    train_paths, val_paths, group_ids, group_weights = [], [], [], []
    held_out_ids = set()
    for group_id, spec in enumerate(args.event_kubric_sources):
        name = spec["name"]
        weight = float(spec["weight"])
        if weight <= 0:
            raise ValueError(f"{name}: weight must be positive")
        paths = sequence_dirs(root / name)
        source_train, source_val = split_source(
            paths, float(spec.get("val_fraction", 0)), args.split_seed
        )
        if not source_train:
            raise ValueError(f"{name}: empty training split")
        train_paths.extend(source_train)
        group_ids.extend([group_id] * len(source_train))
        group_weights.append(weight)
        val_paths.extend(fixed_eval_subset(source_val, args.event_kubric_val_per_source, args.split_seed))
        held_out_ids.update(path.name for path in source_val)
        print(f"Kubric {name}: {len(source_train)} source-train, {len(source_val)} held-out val, weight={weight:g}")

    # The same scene ID may exist in train, exposure, blur, and scale variants.
    kept = [(path, group) for path, group in zip(train_paths, group_ids)
            if path.name not in held_out_ids]
    train_paths = [path for path, _ in kept]
    group_ids = [group for _, group in kept]
    if len(set(group_ids)) != len(group_weights):
        raise ValueError("A source became empty after removing shared validation IDs")
    base_val = sequence_dirs(args.event_kubric_val_root)
    val_paths = fixed_eval_subset(base_val, args.event_kubric_base_val_limit, args.split_seed) + val_paths
    if not val_paths:
        raise ValueError("No Kubric validation sequences")
    print(f"Kubric final training: {len(train_paths)} sequences after shared-ID exclusion")
    print(f"Kubric validation: {len(val_paths)} evaluated sequences per epoch")
    return train_paths, val_paths, group_ids, group_weights


def load_clip(path, frame_indices, rgb_variant, annotations=None):
    if annotations is None:
        annotations = np.load(path / "annotations.npy", allow_pickle=True).item()
    coords = annotations["target_points"]
    occluded = annotations["occluded"]
    if coords.ndim != 3 or coords.shape[-1] != 2 or occluded.shape != coords.shape[:2]:
        raise ValueError(f"{path}: invalid trajectory annotations")
    if frame_indices[-1] >= coords.shape[1]:
        raise ValueError(f"{path}: requested frame {frame_indices[-1]} exceeds annotations")
    prefix = "rgba_" if rgb_variant == "sharp" else "rgba_blur_"
    frames = []
    for frame_index in frame_indices:
        image_path = path / "raw" / f"{prefix}{frame_index:05d}.png"
        with Image.open(image_path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    tracks = np.ascontiguousarray(coords[:, frame_indices].transpose(1, 0, 2))
    visibility = np.ascontiguousarray(~occluded[:, frame_indices].T)
    return frames, tracks, visibility


def visible_in_crop(tracks, visibility, crop_size):
    height, width = crop_size
    visibility &= (
        (tracks[..., 0] >= 0) & (tracks[..., 0] < width)
        & (tracks[..., 1] >= 0) & (tracks[..., 1] < height)
    )
    return visibility


class EventKubricTrain(Movi_F_Base):
    def __init__(self, args, paths):
        super().__init__(args)
        self.paths = paths
        self.frame_stride = int(args.frame_sample_ratio)
        self.rgb_variant = args.event_kubric_rgb_variant

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        worker = get_worker_info()
        return (*sample, {
            "path": self._last_sample_path,
            "requested_index": int(index),
            "fallback_attempt": self._last_fallback_attempt,
            "worker_id": worker.id if worker is not None else -1,
            "first_frame": self._last_first_frame,
            "last_frame": self._last_last_frame,
        })

    def getitem_helper(self, index):
        for attempt in range(8):
            path = self.paths[(index + attempt) % len(self.paths)]
            # ETAP EventKubric sequences contain 96 annotation frames.
            annotations = np.load(path / "annotations.npy", allow_pickle=True).item()
            num_frames = annotations["target_points"].shape[1]
            span = (self.seq_len - 1) * self.frame_stride + 1
            if span > num_frames:
                raise ValueError(f"{path}: {num_frames} frames, need {span}")
            start = np.random.randint(num_frames - span + 1)
            frame_indices = np.arange(start, start + span, self.frame_stride)
            frames, tracks, visibility = load_clip(path, frame_indices, self.rgb_variant, annotations)
            frames, tracks, visibility = self.add_photometric_augs(
                frames, tracks, visibility, replace=False
            )
            frames, tracks = self.add_spatial_augs(frames, tracks, visibility, self.crop_size)
            visibility = visible_in_crop(tracks, visibility, self.crop_size)

            # Match the original Movi-F query pool: first, middle, then quarter frames.
            candidate = np.flatnonzero(visibility[0] | visibility[self.seq_len // 2])
            if len(candidate) < self.traj_per_sample:
                candidate = np.flatnonzero(
                    visibility[0] | visibility[self.seq_len // 2]
                    | visibility[self.seq_len // 4] | visibility[-self.seq_len // 4]
                )
            if len(candidate) < self.traj_per_sample // 4:
                continue
            selected = np.random.permutation(candidate)[:self.traj_per_sample]
            got_points = len(selected)
            video = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2))).float()
            # Record the sequence actually used, including the fallback attempts above.
            self._last_sample_path = str(path.resolve())
            self._last_fallback_attempt = attempt
            self._last_first_frame = int(frame_indices[0])
            self._last_last_frame = int(frame_indices[-1])
            return video, torch.from_numpy(tracks[:, selected].copy()).float(), torch.from_numpy(visibility[:, selected].copy()), got_points
        raise ValueError(f"Could not find enough visible points near {self.paths[index]}")


class EventKubricVal(Dataset):
    def __init__(self, args, paths):
        self.paths = paths
        self.seq_len = args.T
        self.stride = int(args.frame_sample_ratio)
        self.crop_size = tuple(args.input_size)
        self.num_points = args.event_kubric_eval_points
        self.rgb_variant = args.event_kubric_rgb_variant
        self.seed = args.seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        indices = np.arange(self.seq_len) * self.stride
        frames, tracks, visibility = load_clip(path, indices, self.rgb_variant)
        height, width = frames[0].shape[:2]
        crop_h, crop_w = self.crop_size
        if height < crop_h or width < crop_w:
            raise ValueError(f"{path}: frame smaller than crop {self.crop_size}")
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        video = np.stack([frame[y0:y0 + crop_h, x0:x0 + crop_w] for frame in frames])
        tracks[..., 0] -= x0
        tracks[..., 1] -= y0
        visibility = visible_in_crop(tracks, visibility, self.crop_size)
        first_visible = visibility.argmax(axis=0)
        later_visible = (visibility & (np.arange(self.seq_len)[:, None] > first_visible)).any(axis=0)
        candidates = np.flatnonzero(visibility.any(axis=0) & later_visible)
        if len(candidates) == 0:
            raise ValueError(f"{path}: no evaluable trajectories")
        generator = np.random.default_rng(self.seed + index)
        selected = generator.choice(candidates, min(self.num_points, len(candidates)), replace=False)
        tracks = tracks[:, selected].copy()
        visibility = visibility[:, selected].copy()
        query_times = visibility.argmax(axis=0)
        queries = np.stack((query_times, tracks[query_times, np.arange(len(selected)), 1],
                            tracks[query_times, np.arange(len(selected)), 0]), axis=-1)
        video = torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))).float()
        return video, torch.from_numpy(tracks).float(), torch.from_numpy(visibility), torch.from_numpy(queries).float()


class WeightedDistributedSampler(Sampler):
    """Draw a fixed number of samples per epoch with source-level weights."""
    def __init__(self, group_ids, group_weights, samples_per_epoch, rank, world_size, seed):
        if samples_per_epoch <= 0 or samples_per_epoch % world_size:
            raise ValueError("samples_per_epoch must be positive and divisible by world_size")
        counts = np.bincount(group_ids, minlength=len(group_weights))
        weights = [group_weights[group] / counts[group] for group in group_ids]
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.samples_per_epoch = samples_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.samples_per_epoch, replacement=True, generator=generator)
        return iter(indices[self.rank::self.world_size].tolist())

    def __len__(self):
        return self.samples_per_epoch // self.world_size
