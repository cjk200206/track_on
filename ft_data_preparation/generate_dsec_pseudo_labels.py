"""Generate independent DSEC pseudo-label clips with the existing RGB teachers.

Example (absolute paths also work from outside the repository):
    /root/anaconda3/envs/track_on_r/bin/python /remote-home/share/cjk/track_on/ft_data_preparation/generate_dsec_pseudo_labels.py --gpu 0 --resume

Defaults: 48 native RGB frames (~20 Hz), starts 0/24/48/..., 512 first-frame
queries (2/3 SIFT, 1/3 frame difference, random fallback). Each clip resets the
six teachers and uses the existing verifier. RGB is resized to 384x512 for
inference; saved coordinates refer to the original event-view RGB pixels.

Input: SEQUENCE/images/left/distorted/*.png and exposure_timestamps_left.txt
under SEQUENCE/images. Times are integer left-exposure midpoints, not nominal
20 Hz estimates. Clips crossing a gap greater than 75 ms are omitted.
No event cache is required, and no events or training code are changed.
The local checkpoints/pseudo_teachers/torch cache is used when TORCH_HOME is unset.

Output defaults to DSEC/dsec_pseudo_labels/train/stride_1_t48/SEQUENCE/START.npz
beside the resolved train directory. Arrays follow the ERF label schema, with
exposure_timestamps_us added. Queries all start at local time zero; overlapping
clips remain independent, without combining query origins. label_valid describes
finite, in-image positions, independently of pseudo_visibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# Existing teacher wrappers use cuda:0; select visibility before importing torch.
if __name__ == "__main__":
    gpu_parser = argparse.ArgumentParser(add_help=False)
    gpu_parser.add_argument("--gpu", type=int)
    gpu_args, _ = gpu_parser.parse_known_args()
    if gpu_args.gpu is not None:
        if gpu_args.gpu < 0:
            gpu_parser.error("--gpu must be nonnegative")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_args.gpu)

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ft_data_preparation.generate_erf_pseudo_labels import (
    CHECKPOINT_KEYS, TEACHER_NAMES, build_models, load_config, predict_clip,
    read_clip, read_config, sample_queries, save_clip, to_original_coordinates,
)


def read_sequence(sequence):
    """Read already remapped RGB and actual left-camera exposure times."""
    images = sorted((sequence / "images/left/distorted").glob("*.png"))
    if not images or [int(path.stem) for path in images] != list(range(len(images))):
        raise ValueError(f"Expected consecutive remapped RGB indices starting at zero: {sequence}")
    exposure_path = sequence / "images/exposure_timestamps_left.txt"
    exposures = np.loadtxt(exposure_path, delimiter=",", comments="#", dtype=np.int64, ndmin=2)
    if exposures.shape != (len(images), 2) or np.any(exposures[:, 1] <= exposures[:, 0]):
        raise ValueError(f"Expected one positive exposure interval per RGB: {exposure_path}")
    times = exposures[:, 0] + (exposures[:, 1] - exposures[:, 0]) // 2
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Left-exposure midpoints must be strictly increasing: {exposure_path}")
    image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (480, 640):
        raise ValueError(f"Expected readable 640x480 event-view RGB: {images[0]}")
    return images, times, exposures, image.shape[:2]


def clip_starts(times, length, step, max_gap_us):
    """Keep the fixed start grid, excluding incomplete or discontinuous clips."""
    starts = np.arange(0, len(times) - length + 1, step, dtype=np.int64)
    gaps = np.concatenate(([0], np.cumsum(np.diff(times) > max_gap_us)))
    return starts[gaps[starts + length - 1] == gaps[starts]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/DSEC")
    parser.add_argument("--output-root", type=Path, help="Parent of stride_1_tN; default: beside resolved input split")
    parser.add_argument("--config", type=Path, default=ROOT / "ft_data_preparation/teachers.yaml")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--sequences", nargs="+", help="Sequence names, e.g. interlaken_00_c")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; otherwise respect CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--clip-length", type=int, default=48)
    parser.add_argument("--sequence-step", type=int, default=24, help="Clip start spacing in native RGB frames")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--input-size", nargs=2, type=int, default=(384, 512), metavar=("H", "W"))
    parser.add_argument("--max-frame-gap-us", type=int, default=75000)
    parser.add_argument("--max-clips", type=int, help="Limit candidate clips per sequence, including resumed clips")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Skip existing clips only if metadata and arrays match")
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("Need num-shards >= 1 and 0 <= shard-id < num-shards")
    if (args.clip_length < 16 or args.sequence_step < 1 or args.num_points < 3
            or min(args.input_size) < 16 or args.max_frame_gap_us < 1 or args.seed < 0
            or (args.max_clips is not None and args.max_clips < 1)):
        parser.error("Need length >=16, points >=3, size >=16, positive step/gap/max-clips and nonnegative seed")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = (args.output_root.expanduser().resolve() if args.output_root else
                        args.input_root.parent / "dsec_pseudo_labels" / args.input_root.name)
    return args


def main():
    args = parse_args()
    config = load_config(args)
    model_configs = {key: read_config(config[key]) for key in ("trackon2_config_path", "verifier_config_path")}
    if config.get("trackon2_vit_pretrained_path"):
        model_configs["trackon2_config_path"]["vit_pretrained_path"] = config["trackon2_vit_pretrained_path"]
    checkpoints = {}
    for name, key in CHECKPOINT_KEYS.items():
        path = Path(config[key])
        stat = path.stat()
        checkpoints[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    common = {
        "schema_version": 1, "dataset": "DSEC", "teacher_names": list(TEACHER_NAMES),
        "checkpoints": checkpoints, "model_configs": model_configs,
        "cotracker_source": "facebookresearch/co-tracker:cotracker3_online",
        "trackon_memory_size": 24, "support_grid_size": 10, "rgb_stride": 1,
        "clip_length": args.clip_length, "sequence_step": args.sequence_step,
        "num_points": args.num_points, "input_size": list(args.input_size), "seed": args.seed, "amp": args.amp,
        "max_frame_gap_us": args.max_frame_gap_us,
        "query_sampling": "first frame; 2/3 SIFT, 1/3 frame difference; distinct pixels; random fallback",
        "query_source_codes": {"random": 0, "sift": 1, "motion": 2},
        "coordinates": "original distorted event-view image pixels (x,y)",
        "initial_label": "query position, visible", "visibility_rule": "strict majority of six teachers",
        "label_valid_rule": "finite selected position within original image; independent of pseudo_visibility",
        "timestamp_source": "images/exposure_timestamps_left.txt; floor of left-exposure midpoint (us)",
        "rgb_timestamps_estimated": False,
    }
    sequences = ([args.input_root / name for name in args.sequences] if args.sequences else
                 sorted(path for path in args.input_root.iterdir() if path.is_dir()))
    sequences = sorted(set(sequences))
    if not sequences:
        raise ValueError(f"No sequences found: {args.input_root}")
    total_sequences = len(sequences)
    sequences = sequences[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(sequences)}/{total_sequences} sequences; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    jobs = []
    for sequence in sequences:
        paths, times, exposures, image_size = read_sequence(sequence)
        starts = clip_starts(times, args.clip_length, args.sequence_step, args.max_frame_gap_us)
        print(f"{sequence.name}: {len(paths)} RGB, {len(starts)} continuous clips", flush=True)
        for start in starts[:args.max_clips]:
            start = int(start)
            indices = np.arange(start, start + args.clip_length, dtype=np.int64)
            path = args.output_root / f"stride_1_t{args.clip_length}" / sequence.name / f"{start:06d}.npz"
            metadata = json.dumps({
                **common, "sequence": str(sequence.resolve()), "start_frame": start,
                "original_size": list(image_size), "rgb_directory": "images/left/distorted",
            }, sort_keys=True)
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f"Output exists; use --resume: {path}")
                with np.load(path, allow_pickle=False) as saved:
                    saved_metadata = json.loads(saved["metadata"].item())
                    # Start spacing changes the clip list, not an existing clip's labels.
                    saved_metadata["sequence_step"] = args.sequence_step
                    if (saved_metadata != json.loads(metadata)
                            or not np.array_equal(saved["frame_indices"], indices)
                            or not np.array_equal(saved["timestamps_us"], times[indices])
                            or not np.array_equal(saved["exposure_timestamps_us"], exposures[indices])
                            or saved["tracks"].shape != (args.clip_length, args.num_points, 2)
                            or saved["queries"].shape != (args.num_points, 3)
                            or saved["label_valid"].shape != (args.clip_length, args.num_points)
                            or saved["pseudo_visibility"].shape != (args.clip_length, args.num_points)):
                        raise ValueError(f"Existing labels do not match this run: {path}")
                continue
            jobs.append((paths, indices, times[indices], exposures[indices], image_size, path, metadata))
    print(f"{len(jobs)} clips to generate; output: {args.output_root}", flush=True)
    if not jobs:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The existing teacher/verifier wrappers require CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    cache = ROOT / "checkpoints/pseudo_teachers/torch"
    if (cache / "hub").is_dir():
        os.environ.setdefault("TORCH_HOME", str(cache))
    models, verifier = build_models(config)
    for number, (paths, indices, times, exposures, image_size, path, metadata) in enumerate(jobs, 1):
        started = time.monotonic()
        digest = hashlib.sha256(f"{args.seed}:{path.parent.name}:{indices[0]}".encode()).digest()
        clip_seed = int.from_bytes(digest[:4], "little")
        random.seed(clip_seed)
        torch.manual_seed(clip_seed)
        frames = read_clip(paths, indices, args.input_size, image_size)
        queries, sources = sample_queries(frames, args.num_points, np.random.default_rng(clip_seed))
        result = predict_clip(models, verifier, frames, queries, args.amp)
        result = to_original_coordinates(result, queries, args.input_size, image_size)
        save_clip(path, {**result, "query_sources": sources, "frame_indices": indices,
                         "timestamps_us": times, "exposure_timestamps_us": exposures,
                         "teacher_names": np.asarray(TEACHER_NAMES)}, metadata)
        print(f"[{number}/{len(jobs)}] Saved {path} ({time.monotonic() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
