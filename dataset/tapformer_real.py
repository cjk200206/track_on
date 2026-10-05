"""RGB frames and annotations for TAPFormer's InivTAP and DrivTAP benchmarks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class RealTAPSequence:
    name: str
    dataset: str
    image_paths: list[Path]
    tracks: np.ndarray  # (T, N, 2), original image pixels
    occluded: np.ndarray  # (T, N), True means occluded

    def frames(self):
        for path in self.image_paths:
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB"))


@dataclass
class RealTAPEventStream:
    sequence: RealTAPSequence
    event_paths: list[Path]
    rgb_indices: np.ndarray
    score_indices: list[int | None]
    dt: str

    def steps(self):
        try:
            import h5py
            import hdf5plugin  # noqa: F401 - registers the Blosc HDF5 filter
        except ImportError as exc:
            raise ImportError("Reading time surfaces requires h5py and hdf5plugin") from exc

        previous_rgb_index = -1
        for path, rgb_index, score_index in zip(self.event_paths, self.rgb_indices, self.score_indices):
            if rgb_index != previous_rgb_index:
                with Image.open(self.sequence.image_paths[rgb_index]) as image:
                    rgb = np.asarray(image.convert("RGB"))
                previous_rgb_index = rgb_index
            with h5py.File(path, "r") as handle:
                events = np.asarray(handle["time_surface"][:], dtype=np.float32)
            if events.ndim != 3 or events.shape[:2] != rgb.shape[:2]:
                raise ValueError(f"Event map does not match RGB image: {path}")
            yield rgb, events, score_index


def load_real_tap_event_stream(sequence, dt="auto"):
    """Use the latest RGB image at each event time; score only RGB timestamps."""
    event_root = sequence.image_paths[0].parent.parent / "events" / "time_surfaces_v2_5"
    if dt == "auto":
        candidates = [path for path in event_root.iterdir()
                      if path.is_dir() and path.name.replace(".", "", 1).isdigit()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one numeric event cache in {event_root}; found {len(candidates)}")
        event_dir = candidates[0]
    else:
        event_dir = event_root / str(dt)
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Event cache not found: {event_dir}")

    event_paths = sorted(event_dir.glob("*.h5"), key=lambda path: int(path.stem))
    if not event_paths:
        raise ValueError(f"No H5 event maps found in {event_dir}")
    image_times = np.asarray([int(path.stem) for path in sequence.image_paths], dtype=np.int64)
    event_times = np.asarray([int(path.stem) for path in event_paths], dtype=np.int64)
    if event_times[0] != image_times[0] or event_times[-1] != image_times[-1]:
        raise ValueError(f"Event cache does not span all RGB timestamps: {event_dir}")
    score_lookup = {int(timestamp): index for index, timestamp in enumerate(image_times)}
    if not set(image_times).issubset(event_times):
        raise ValueError(f"Event cache is missing RGB timestamps: {event_dir}")
    rgb_indices = np.searchsorted(image_times, event_times, side="right") - 1
    score_indices = [score_lookup.get(int(timestamp)) for timestamp in event_times]
    return RealTAPEventStream(sequence, event_paths, rgb_indices, score_indices, event_dir.name)


def load_real_tap_sequence(dataset_root, dataset, name, drivtap_gt_mode="downsample_even"):
    """Match each annotated frame to its RGB image using the image timestamps."""
    if dataset not in {"InivTAP", "DrivTAP"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    seq_dir = Path(dataset_root) / dataset / name
    image_dir = seq_dir / "images_corrected"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    raw_timestamps = np.atleast_1d(np.loadtxt(seq_dir / "image_timestamps.txt"))
    if not np.all(np.isfinite(raw_timestamps)) or not np.all(raw_timestamps == np.rint(raw_timestamps)):
        raise ValueError(f"Invalid image timestamps for {dataset}/{name}")
    timestamps = raw_timestamps.astype(np.int64)
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    image_times = np.asarray([int(path.stem) for path in image_paths], dtype=np.int64)
    if not np.array_equal(image_times, timestamps):
        raise ValueError(f"Image files and timestamps do not match for {dataset}/{name}")

    annotations = np.load(seq_dir / "annotations.npy", allow_pickle=True).item()
    coords = np.asarray(annotations["coords"])
    visible = np.asarray(annotations["visibility"])
    if visible.ndim == 3 and visible.shape[-1] == 1:
        visible = visible[..., 0]
    if coords.shape[:2] != visible.shape or coords.shape[-1] != 2:
        raise ValueError(f"Invalid annotation shapes for {dataset}/{name}")

    if dataset == "DrivTAP" and drivtap_gt_mode != "downsample_even":
        raise ValueError("RGB evaluation supports drivtap_gt_mode=downsample_even only")
    if dataset == "DrivTAP" and coords.shape[1] == 2 * len(image_paths):
        coords = coords[:, ::2]
        visible = visible[:, ::2]
    if coords.shape[1] != len(image_paths):
        raise ValueError(
            f"GT length {coords.shape[1]} differs from RGB length {len(image_paths)} "
            f"for {dataset}/{name}"
        )
    return RealTAPSequence(
        name=name,
        dataset=dataset,
        image_paths=image_paths,
        tracks=np.transpose(coords, (1, 0, 2)).astype(np.float32),
        occluded=np.logical_not(np.transpose(visible.astype(bool), (1, 0))),
    )
