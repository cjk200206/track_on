"""ERF clips with sparse RGB, cached events and offline first-frame queries."""

import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def matching_rows(available, requested, source):
    rows = np.searchsorted(available, requested)
    if (np.any(np.diff(available) <= 0) or np.any(rows >= len(available))
            or not np.array_equal(available[rows], requested)):
        raise ValueError(f"Missing or unordered native frame indices: {source}")
    return rows


def resize_labels(tracks, queries, valid, original_size, input_size):
    """Use the same pixel-center convention as teacher generation and OpenCV."""
    scale = np.asarray(input_size[::-1], dtype=np.float32) / original_size[::-1]
    tracks = ((tracks + 0.5) * scale - 0.5).astype(np.float32)
    queries = queries.copy()
    queries[:, 1:] = (queries[:, 1:] + 0.5) * scale - 0.5
    height, width = input_size
    valid = (valid & np.isfinite(tracks).all(-1)
             & (tracks[..., 0] >= 0) & (tracks[..., 0] < width)
             & (tracks[..., 1] >= 0) & (tracks[..., 1] < height))
    return tracks, queries, valid


def build_erf_index(args, sequences):
    """Screen labels once, before workers start; return a serializable index."""
    records, summary = [], []
    required_steps = math.ceil((args.T - 1) * args.min_visible_fraction)
    for name in sequences:
        sequence = Path(args.erf_root) / name
        event_path = Path(args.erf_event_root) / f"{name}.h5"
        images = sorted((sequence / "processed_images").glob("*.png"))
        if not images or [int(path.stem) for path in images] != list(range(len(images))):
            raise ValueError(f"Expected consecutive RGB filenames starting at zero: {sequence}")
        with h5py.File(event_path, "r") as cache:
            indices = cache["frame_indices"][:]
            times = cache["timestamps_us"][:]
            size = tuple(int(value) for value in cache.attrs["image_size"])
            if (Path(cache.attrs["source"]).resolve() != sequence.resolve()
                    or int(cache.attrs["native_rgb_stride"]) != args.event_native_stride
                    or not np.array_equal(indices, np.arange(0, len(images), args.event_native_stride))
                    or cache["time_surface"].shape != (len(indices), *size, 10)
                    or cache["time_surface"].dtype != np.float32
                    or times.shape != indices.shape or np.any(np.diff(times) <= 0)
                    or int(cache.attrs["n_bins"]) != 5
                    or str(cache.attrs["channel_order"]) != "bin0_p0,bin0_p1,...,bin4_p0,bin4_p1"):
                raise ValueError(f"Incompatible event cache: {event_path}")

        labels = sorted((Path(args.erf_label_root) / name).glob("*.npz"))
        if not labels:
            raise FileNotFoundError(f"No pseudo labels for {name}: {args.erf_label_root}")
        kept = 0
        for path in labels:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(data["metadata"].item())
                native = data["frame_indices"]
                tracks, visible = data["tracks"], data["pseudo_visibility"]
                valid, queries = data["label_valid"], data["queries"]
                if (metadata["schema_version"] != 1
                        or Path(metadata["sequence"]).resolve() != sequence.resolve()
                        or tuple(metadata["original_size"]) != size
                        or tracks.shape != (len(native), len(queries), 2)
                        or visible.shape != tracks.shape[:2] or valid.shape != visible.shape
                        or queries.shape != (tracks.shape[1], 3)
                        or np.any(queries[:, 0] != 0)
                        or visible.dtype != np.bool_ or valid.dtype != np.bool_
                        or native[0] != metadata["start_frame"]
                        or not np.array_equal(tracks[0], queries[:, 1:])):
                    raise ValueError(f"Incompatible first-frame pseudo labels: {path}")
                requested = native[0] + np.arange(args.T) * args.event_native_stride
                label_rows = matching_rows(native, requested, path)
                event_rows = matching_rows(indices, requested, event_path)
                if not np.array_equal(data["timestamps_us"][label_rows], times[event_rows]):
                    raise ValueError(f"Pseudo-label/event timestamp mismatch: {path}")
                tracks, queries, valid = resize_labels(
                    tracks[label_rows], queries, valid[label_rows], size, tuple(args.input_size),
                )
                usable = valid & visible[label_rows]
                points = np.flatnonzero(usable[0] & (usable[1:].sum(0) >= required_steps))
                if len(points) < args.N:
                    continue
                records.append({
                    "sequence": name, "label": str(path), "image_size": list(size),
                    "label_rows": label_rows.tolist(), "event_rows": event_rows.tolist(),
                    "frame_indices": requested.tolist(), "timestamps_us": times[event_rows].tolist(),
                    "rgb_files": [images[int(native)].name for native in requested],
                    "points": points.tolist(),
                })
                kept += 1
        summary.append({"sequence": name, "candidates": len(labels), "kept": kept,
                        "rejected": len(labels) - kept})
    return records, summary


class FrameEventERF(Dataset):
    """No augmentation; each item starts at its saved teacher query origin."""

    def __init__(self, args, records, training):
        self.args = args
        self.records = records
        self.training = training
        if not records:
            raise ValueError("No ERF clips survive trajectory screening")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        args, record = self.args, self.records[index]
        rng = np.random if self.training else np.random.default_rng(args.seed + index)
        points = rng.choice(record["points"], args.N, replace=False)
        rows = record["label_rows"]
        with np.load(record["label"], allow_pickle=False) as data:
            tracks = data["tracks"][rows][:, points]
            visibility = data["pseudo_visibility"][rows][:, points]
            valid = data["label_valid"][rows][:, points]
            queries = data["queries"][points]
        tracks, queries, valid = resize_labels(
            tracks, queries, valid, tuple(record["image_size"]), tuple(args.input_size),
        )
        # Invalid labels remain absent supervision, including for uncertainty.
        tracks = np.where(valid[..., None], tracks, 0).astype(np.float32)
        height, width = args.input_size
        video = np.empty((args.T, 3, height, width), dtype=np.float32)
        events = np.empty((args.T, 10, height, width), dtype=np.float32)
        events[0] = 0  # Discard events before this clip's RGB/query initialization.
        new_rgb = np.arange(args.T) % (args.rgb_native_interval // args.event_native_stride) == 0
        sequence = Path(args.erf_root) / record["sequence"] / "processed_images"
        event_path = Path(args.erf_event_root) / f"{record['sequence']}.h5"
        with h5py.File(event_path, "r") as cache:
            for step, event_row in enumerate(record["event_rows"]):
                if new_rgb[step]:
                    path = sequence / record["rgb_files"][step]
                    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if rgb is None or rgb.shape[:2] != tuple(record["image_size"]):
                        raise ValueError(f"Unreadable RGB or changed image size: {path}")
                    rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (width, height))
                video[step] = rgb.transpose(2, 0, 1)
                if step:
                    # Read and resize one original-resolution surface at a time.
                    surface = cv2.resize(cache["time_surface"][event_row], (width, height))
                    events[step] = surface.transpose(2, 0, 1)
        return {
            "video": torch.from_numpy(video), "events": torch.from_numpy(events),
            "tracks": torch.from_numpy(tracks), "queries": torch.from_numpy(queries),
            "visibility": torch.from_numpy(visibility.copy()), "label_valid": torch.from_numpy(valid),
            "img_ifnew": torch.from_numpy(new_rgb),
            "frame_indices": torch.tensor(record["frame_indices"]),
            "timestamps_us": torch.tensor(record["timestamps_us"]),
        }
