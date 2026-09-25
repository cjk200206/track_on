"""Original RGB frames and sparse tracks from TAPFormer's EDS/EC subsets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class FeatureTAPSequence:
    dataset: str
    name: str
    image_paths: list[Path]
    frame_times: np.ndarray  # seconds
    query_xy: np.ndarray  # (N, 2), x/y pixels in the first image
    gt_rows: np.ndarray  # (M, 4): id, time in seconds, x, y

    def frames(self):
        for path in self.image_paths:
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB"))


def load_feature_tap_sequence(dataset_root, dataset, name):
    """Load the same RGB and sparse GT files used by TAPFormer's EDS/EC tests."""
    directory = {"EDS": "eds_subseq", "EC": "ec_subseq"}.get(dataset)
    if directory is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    seq_dir = Path(dataset_root) / directory / name
    image_dir = seq_dir / "images_corrected"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two RGB frames for {dataset}/{name}")
    frame_times = np.asarray([int(path.stem) * 1e-6 for path in image_paths], dtype=np.float64)
    if not np.all(np.diff(frame_times) > 0):
        raise ValueError(f"RGB timestamps are not increasing for {dataset}/{name}")

    gt_path = seq_dir / (f"{name}.gt.txt" if dataset == "EDS" else "track.gt.txt")
    gt_rows = np.atleast_2d(np.loadtxt(gt_path, dtype=np.float64))
    if gt_rows.shape[1] != 4 or not np.all(np.isfinite(gt_rows)):
        raise ValueError(f"Expected finite [id, time, x, y] GT rows in {gt_path}")
    _, first_rows = np.unique(gt_rows[:, 0], return_index=True)
    query_xy = gt_rows[first_rows, 2:4].astype(np.float32)
    if not np.allclose(gt_rows[first_rows, 1], frame_times[0], atol=1e-5, rtol=0):
        raise ValueError(f"Query timestamps do not match the first RGB frame for {dataset}/{name}")
    return FeatureTAPSequence(dataset, name, image_paths, frame_times, query_xy, gt_rows)
