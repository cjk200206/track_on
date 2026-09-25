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
