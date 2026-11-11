"""Real Frame-Event clips with cached events and offline first-frame queries."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.real_world_dataset import (
    _apply_brightness_contrast, _apply_gamma, _apply_saturation, _apply_hue,
    _apply_grayscale_blend, _apply_gaussian_noise, _apply_gaussian_blur,
    _apply_motion_blur, _apply_jpeg_compress,
)


def real_source_args(args, source):
    """Normalize single-frequency and probability-sampled real sources once."""
    kind = source["dataset"].lower()
    if kind not in ("erf", "dsec"):
        raise ValueError(f"Unknown real dataset: {kind}")
    options = {**vars(args), **source, "dataset": kind,
               "rgb_directory": "processed_images" if kind == "erf" else "images/left/distorted"}
    weights = source.get("event_stride_weights")
    options["stride_weights"] = {int(key): float(value) for key, value in (
        weights if weights is not None else {options["event_native_stride"]: 1.0}).items()}
    interval = options["rgb_native_interval"]
    if (not options["stride_weights"] or not isinstance(interval, int) or interval < 1
            or any(stride < 1 or interval % stride or not math.isfinite(weight) or weight <= 0
                   for stride, weight in options["stride_weights"].items())):
        raise ValueError(f"{source['name']}: invalid event frequencies or weights")
    if kind == "dsec" and set(options["stride_weights"]) != {1}:
        raise ValueError("DSEC uses native stride-1 event windows; change only the RGB interval")
    options["event_roots"] = {
        stride: str(Path(options["event_root"]) / f"stride_{stride}") if weights is not None else options["event_root"]
        for stride in options["stride_weights"]
    }
    return SimpleNamespace(**options)


def sample_photometric_augmentation():
    """Track-On-R RGB defaults; sample once per clip and reuse at RGB arrivals.

    These are the published real-world loader's probabilities and strengths.
    Its extra Gamma/noise/motion-blur operations are included; geometry and
    verifier-only trajectory perturbations are not applied here.
    """
    transforms = []
    if np.random.rand() < 0.7:
        transforms.append((_apply_brightness_contrast, (np.random.uniform(0.8, 1.25),
                                                       np.random.uniform(-32.0, 32.0))))
    if np.random.rand() < 0.5:
        transforms.append((_apply_gamma, (np.random.uniform(0.8, 1.25),)))
    if np.random.rand() < 0.6:
        transforms.append((_apply_saturation, (np.random.uniform(0.7, 1.4),)))
    if np.random.rand() < 0.4:
        transforms.append((_apply_hue, (np.random.uniform(-10.0, 10.0),)))
    if np.random.rand() < 0.15:
        transforms.append((_apply_grayscale_blend, (np.random.uniform(0.1, 0.4),)))
    if np.random.rand() < 0.3:
        transforms.append((_apply_gaussian_noise, (np.random.uniform(2.0, 8.0),)))
    if np.random.rand() < 0.25:
        transforms.append((_apply_gaussian_blur, (int(np.random.choice((3, 5))),
                                                 np.random.uniform(0.2, 1.0))))
    if np.random.rand() < 0.2:
        transforms.append((_apply_motion_blur, (int(np.random.choice((5, 7, 9))),
                                               np.random.uniform(0.0, 360.0))))
    if np.random.rand() < 0.3:
        transforms.append((_apply_jpeg_compress, (int(np.random.randint(40, 91)),)))
    return transforms


def apply_photometric_augmentation(rgb, transforms):
    """Apply one clip's transforms to a new RGB; noise is sampled per arrival."""
    image = rgb.astype(np.float32, copy=True)
    for transform, parameters in transforms:
        if transform is _apply_jpeg_compress:
            # The OpenCV JPEG helper takes BGR and returns RGB.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = transform(image, *parameters)
    return image


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


def read_event_index(args, sequence, images, stride, exposures):
    path = Path(args.event_roots[stride]) / f"{sequence.name}.h5"
    with h5py.File(path, "r") as cache:
        indices = cache["frame_indices"][:]
        times = cache["timestamps_us"][:]
        size = tuple(int(value) for value in cache.attrs["image_size"])
        if (Path(cache.attrs["source"]).resolve() != sequence.resolve()
                or int(cache.attrs["native_rgb_stride"]) != stride
                or not np.array_equal(indices, np.arange(0, len(images), stride))
                or cache["time_surface"].shape != (len(indices), *size, 10)
                or cache["time_surface"].dtype != np.float32
                or times.shape != indices.shape or np.any(np.diff(times) <= 0)
                or int(cache.attrs["n_bins"]) != 5
                or str(cache.attrs["channel_order"]) != "bin0_p0,bin0_p1,...,bin4_p0,bin4_p1"):
            raise ValueError(f"Incompatible event cache: {path}")
        if exposures is not None:
            midpoints = exposures[:, 0] + (exposures[:, 1] - exposures[:, 0]) // 2
            if (size != (480, 640) or not np.array_equal(times, midpoints)
                    or not np.array_equal(cache["exposure_timestamps_us"][:], exposures)
                    or cache.attrs["rgb_directory"] != args.rgb_directory
                    or bool(cache.attrs["rgb_timestamps_estimated"])):
                raise ValueError(f"DSEC exposure/coordinate metadata mismatch: {path}")
    return indices, times, size


def build_real_index(args, sequences):
    """Index aligned T-step clips and eligible points independently per frequency."""
    missing = [f"stride_{stride}/{name}" for stride, root in args.event_roots.items()
               for name in sequences if not (Path(root) / f"{name}.h5").is_file()]
    if missing:
        raise FileNotFoundError(f"{args.name}: {len(missing)} event caches missing; "
                                f"finish conversion first (e.g. {', '.join(missing[:5])})")
    records, summary = [], []
    for name in sequences:
        sequence = Path(args.root) / name
        images = sorted((sequence / args.rgb_directory).glob("*.png"))
        if not images or [int(path.stem) for path in images] != list(range(len(images))):
            raise ValueError(f"Expected consecutive RGB filenames starting at zero: {sequence}")
        exposures = None
        if args.dataset == "dsec":
            exposures = np.loadtxt(sequence / "images/exposure_timestamps_left.txt", delimiter=",",
                                   comments="#", dtype=np.int64, ndmin=2)
            if exposures.shape != (len(images), 2) or np.any(exposures[:, 1] <= exposures[:, 0]):
                raise ValueError(f"Invalid DSEC exposure table: {sequence}")
        caches = {stride: read_event_index(args, sequence, images, stride, exposures)
                  for stride in args.stride_weights}
        sizes = {entry[2] for entry in caches.values()}
        if len(sizes) != 1:
            raise ValueError(f"Event frequencies disagree on image size: {sequence}")
        size = sizes.pop()
        labels = sorted((Path(args.label_root) / name).glob("*.npz"))
        if not labels:
            raise FileNotFoundError(f"No pseudo labels for {name}: {args.label_root}")
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
                frequencies = {}
                for stride, (indices, times, _) in caches.items():
                    requested = native[0] + np.arange(args.T) * stride
                    if requested[-1] > native[-1] or requested[-1] > indices[-1]:
                        continue
                    label_rows = matching_rows(native, requested, path)
                    event_rows = matching_rows(indices, requested, args.event_roots[stride])
                    if not np.array_equal(data["timestamps_us"][label_rows], times[event_rows]):
                        raise ValueError(f"Pseudo-label/event timestamp mismatch: {path}, stride={stride}")
                    if exposures is not None:
                        if (metadata["dataset"] != "DSEC" or metadata["rgb_directory"] != args.rgb_directory
                                or metadata["rgb_timestamps_estimated"]
                                or not np.array_equal(data["exposure_timestamps_us"][label_rows], exposures[requested])
                                or np.any(np.diff(times[event_rows]) > 75000)):
                            raise ValueError(f"DSEC label exposure metadata or continuity mismatch: {path}")
                    _, _, scaled_valid = resize_labels(
                        tracks[label_rows], queries, valid[label_rows], size, tuple(args.input_size),
                    )
                    usable = scaled_valid & visible[label_rows]
                    supervised = usable[1:]
                    required_steps = math.ceil(len(supervised) * args.min_visible_fraction)
                    points = np.flatnonzero(usable[0] & (supervised.sum(0) >= required_steps))
                    if len(points) < args.N:
                        continue
                    frequencies[str(stride)] = {
                        "points": points.tolist(),
                        "label_rows": label_rows.tolist(), "event_rows": event_rows.tolist(),
                        "frame_indices": requested.tolist(), "timestamps_us": times[event_rows].tolist(),
                        "rgb_files": [images[int(frame)].name for frame in requested],
                    }
                if not frequencies:
                    continue
                record = {"sequence": name, "label": str(path), "image_size": list(size), "frequencies": frequencies}
                records.append(record)
                kept += 1
        summary.append({"sequence": name, "candidates": len(labels), "kept": kept,
                        "rejected": len(labels) - kept})
    return records, summary


class FrameEventReal(Dataset):
    """Fixed geometry and query origin, with training-only RGB augmentation.

    Photometric augmentation defaults to enabled for training. Set
    args.photometric_augmentation=False for the unaugmented control. Validation
    always uses clean RGB. Events, queries and label masks are unaffected.
    """

    def __init__(self, args, records, training, source_id=None, event_stride=None):
        self.args = args
        self.records = records
        self.training = training
        self.source_id = source_id
        self.strides = tuple(args.stride_weights)
        self.event_stride = event_stride
        self.photometric_augmentation = training and getattr(args, "photometric_augmentation", True)
        if not records:
            raise ValueError("No real clips survive trajectory screening")

    def __len__(self):
        return len(self.records)

    def for_stride(self, stride):
        if stride not in self.strides:
            raise ValueError(f"Unknown event stride: {stride}")
        view = copy.copy(self)
        view.event_stride = stride
        view.records = [record for record in self.records if str(stride) in record["frequencies"]]
        return view

    def __getitem__(self, index):
        args, record = self.args, self.records[index]
        rng = np.random if self.training else np.random.default_rng(args.seed + index)
        stride = self.event_stride
        if stride is None:
            available = [stride for stride in self.strides if str(stride) in record["frequencies"]]
            weights = np.asarray([args.stride_weights[stride] for stride in available], dtype=np.float64)
            stride = int(rng.choice(available, p=weights / weights.sum())) if self.training else available[0]
        timing = record["frequencies"][str(stride)]
        candidates = timing["points"]
        points = rng.choice(candidates, args.N, replace=False)
        rows = timing["label_rows"]
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
        length = len(rows)
        video = np.empty((length, 3, height, width), dtype=np.float32)
        events = np.empty((length, 10, height, width), dtype=np.float32)
        events[0] = 0  # Discard events before this clip's RGB/query initialization.
        new_rgb = np.arange(length) % (args.rgb_native_interval // stride) == 0
        sequence = Path(args.root) / record["sequence"] / args.rgb_directory
        event_path = Path(args.event_roots[stride]) / f"{record['sequence']}.h5"
        transforms = sample_photometric_augmentation() if self.photometric_augmentation else ()
        with h5py.File(event_path, "r") as cache:
            for step, event_row in enumerate(timing["event_rows"]):
                if new_rgb[step]:
                    path = sequence / timing["rgb_files"][step]
                    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if rgb is None or rgb.shape[:2] != tuple(record["image_size"]):
                        raise ValueError(f"Unreadable RGB or changed image size: {path}")
                    rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (width, height))
                    # Augment each arrival once; held RGB includes the same noise realization.
                    if self.photometric_augmentation:
                        rgb = apply_photometric_augmentation(rgb, transforms)
                video[step] = rgb.transpose(2, 0, 1)
                if step:
                    # Read and resize one original-resolution surface at a time.
                    surface = cv2.resize(cache["time_surface"][event_row], (width, height))
                    events[step] = surface.transpose(2, 0, 1)
        sample = {
            "video": torch.from_numpy(video), "events": torch.from_numpy(events),
            "tracks": torch.from_numpy(tracks), "queries": torch.from_numpy(queries),
            "visibility": torch.from_numpy(visibility.copy()), "label_valid": torch.from_numpy(valid),
            "img_ifnew": torch.from_numpy(new_rgb),
            "frame_indices": torch.tensor(timing["frame_indices"]),
            "timestamps_us": torch.tensor(timing["timestamps_us"]),
        }
        if len(self.strides) > 1:
            sample["event_stride"] = torch.tensor(stride)
        if self.source_id is not None:
            sample["source_id"] = torch.tensor(self.source_id)
        return sample
