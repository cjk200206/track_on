"""Generate independent ERF pseudo-label clips with the existing RGB teachers.

Run from the repository root after filling checkpoint paths in the teacher config:
    python ft_data_preparation/generate_erf_pseudo_labels.py --sequences 0000 --max-clips 2
    python ft_data_preparation/generate_erf_pseudo_labels.py --resume
    python ft_data_preparation/generate_erf_pseudo_labels.py --gpu 2 --num-shards 4 --shard-id 2 --resume

--gpu selects a physical GPU, exposed to this process as cuda:0. Without it,
the existing CUDA_VISIBLE_DEVICES setting is respected. Concurrent workers
should use different shard IDs with the same sequence list and shard count.

Defaults: native RGB stride 2, 48 frames, clip starts 0/48/96/..., and 256
first-frame queries (2/3 SIFT, 1/3 frame difference, random fallback). Teachers
see clean, full-view RGB resized to 384x512. All saved coordinates are mapped
back to the original image. No training code or event cache is modified.

Output: OUTPUT/stride_2_t48/SEQUENCE/START.npz. teacher_tracks uses (time, point,
teacher, xy), tracks uses (time, point, xy), and queries uses (local_time, x, y).
label_valid describes finite,
in-image positions, not calibrated confidence; pseudo_visibility is a separate
strict majority vote. Input timestamps are estimates copied from the event cache.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

# Select visibility before importing PyTorch; existing wrappers use .cuda().
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
import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT_KEYS = {
    "trackon2": "trackon2_checkpoint_path",
    "bootstapnext": "bootstapnext_checkpoint_path",
    "bootstapir": "bootstapir_checkpoint_path",
    "locotrack": "anthro_locotrack_checkpoint_path",
    "alltracker": "alltracker_checkpoint_path",
    "verifier": "verifier_checkpoint_path",
}
TEACHER_NAMES = ("trackon2", "bootstapnext", "bootstapir", "cotracker3_window", "locotrack", "alltracker")


def read_config(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def resolve_path(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(args):
    config = read_config(args.config)
    for override in args.checkpoint:
        name, separator, path = override.partition("=")
        if not separator or name not in CHECKPOINT_KEYS or not path:
            raise ValueError(f"Expected --checkpoint NAME=PATH; names: {', '.join(CHECKPOINT_KEYS)}")
        config[CHECKPOINT_KEYS[name]] = path
    keys = [*CHECKPOINT_KEYS.values(), "trackon2_config_path", "verifier_config_path"]
    for key in keys:
        if not config.get(key) or not resolve_path(config[key]).is_file():
            raise FileNotFoundError(f"Set {key} to an existing file in {args.config}, or use --checkpoint")
        config[key] = str(resolve_path(config[key]))
    backbone_path = (config.get("trackon2_vit_pretrained_path")
                     or read_config(config["trackon2_config_path"]).get("vit_pretrained_path"))
    if backbone_path:
        backbone_path = resolve_path(backbone_path)
        has_weights = (any(backbone_path.glob("*.safetensors"))
                       or any(backbone_path.glob("pytorch_model*.bin")))
        if not (backbone_path / "config.json").is_file() or not has_weights:
            raise FileNotFoundError(
                f"DINOv3 directory needs config.json and model weights: {backbone_path}. "
                f"Copy the complete model directory and set trackon2_vit_pretrained_path in {args.config}"
            )
        config["trackon2_vit_pretrained_path"] = str(backbone_path)
    return config


def build_models(config):
    # Keep models on CPU between calls; only one teacher occupies the GPU at a time.
    from model.trackon_predictor import Predictor
    from ensemble.tapnext.tapnext_predictor import TAPNextPredictor
    from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
    from ensemble.cotracker import CoTracker_Predictor
    from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
    from ensemble.alltracker.alltracker_predictor import AllTrackerPredictor
    from verifier.verifier_predictor import VerifierPredictor

    trackon_args = SimpleNamespace(**read_config(config["trackon2_config_path"]))
    if config.get("trackon2_vit_pretrained_path"):
        trackon_args.vit_pretrained_path = config["trackon2_vit_pretrained_path"]
    trackon_args.M_i = 24
    factories = {
        "trackon2": lambda: Predictor(trackon_args, config["trackon2_checkpoint_path"], support_grid_size=10),
        "bootstapnext": lambda: TAPNextPredictor(config["bootstapnext_checkpoint_path"]),
        "bootstapir": lambda: TAPIRPredictor(config["bootstapir_checkpoint_path"]),
        "cotracker3_window": lambda: CoTracker_Predictor(windowed=True),
        "locotrack": lambda: LocoTrackPredictor(config["anthro_locotrack_checkpoint_path"]),
        "alltracker": lambda: AllTrackerPredictor(checkpoint_path=config["alltracker_checkpoint_path"]),
    }
    models = {}
    for name, factory in factories.items():
        models[name] = factory().eval().requires_grad_(False).cpu()
        print(f"Loaded {name}", flush=True)
    verifier_args = SimpleNamespace(**read_config(config["verifier_config_path"]))
    verifier = VerifierPredictor(verifier_args, config["verifier_checkpoint_path"])
    return models, verifier.eval().requires_grad_(False).cpu()


def read_sequence(sequence, event_path):
    images = sorted((sequence / "processed_images").glob("*.png"))
    if not images or [int(path.stem) for path in images] != list(range(len(images))):
        raise ValueError(f"Expected consecutive RGB indices starting at zero: {sequence}")
    with h5py.File(event_path, "r") as cache:
        if Path(cache.attrs["source"]).resolve() != sequence.resolve():
            raise ValueError(f"Event cache belongs to another sequence: {event_path}")
        times = cache["rgb_timestamps_us"][:]
        image_size = tuple(int(value) for value in cache.attrs["image_size"])
        time_source = str(cache.attrs["timestamp_source"])
        estimated = bool(cache.attrs["rgb_timestamps_estimated"])
    if times.shape != (len(images),) or np.any(np.diff(times) <= 0):
        raise ValueError(f"RGB/event timestamp mismatch: {sequence}")
    return images, times, image_size, time_source, estimated


def read_clip(paths, indices, input_size, original_size):
    frames = []
    for index in indices:
        frame = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != original_size:
            raise ValueError(f"Unreadable RGB or inconsistent image size: {paths[index]}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, tuple(reversed(input_size)), interpolation=cv2.INTER_LINEAR))
    return np.stack(frames)


def sample_queries(frames, count, rng):
    """Sample distinct first-frame pixels; motion may include camera motion."""
    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(frames[1], cv2.COLOR_RGB2GRAY)
    difference = cv2.GaussianBlur(cv2.absdiff(gray, next_gray).astype(np.float32), (5, 5), 0)
    ys, xs = np.nonzero(difference > np.percentile(difference, 95))
    motion = np.column_stack((xs, ys)).astype(np.float32)
    keypoints = cv2.SIFT_create().detect(gray, None)
    sift = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
    height, width = gray.shape
    if count > (height - 2) * (width - 2):
        raise ValueError("Too many queries for distinct interior image pixels")
    points, sources, occupied = [], [], set()

    def append(candidates, limit, source):
        added = 0
        for index in rng.permutation(len(candidates)):
            x, y = candidates[index]
            pixel = (int(x), int(y))
            if pixel in occupied or not (1 <= x < width - 1 and 1 <= y < height - 1):
                continue
            occupied.add(pixel)
            points.append((x, y))
            sources.append(source)
            added += 1
            if added == limit:
                break

    append(sift, count - count // 3, 1)
    append(motion, count // 3, 2)
    while len(points) < count:
        remaining = count - len(points)
        candidates = rng.uniform((1, 1), (width - 1, height - 1), size=(remaining * 2, 2))
        append(candidates, remaining, 0)
    queries = np.column_stack((np.zeros(count), np.asarray(points))).astype(np.float32)
    return queries, np.asarray(sources, dtype=np.uint8)


@torch.inference_mode()
def predict_clip(models, verifier, frames, queries, amp):
    video = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).float()[None].cuda()
    query_tensor = torch.from_numpy(queries)[None].cuda()
    tracks, visibility = [], []
    expected = (1, len(frames), len(queries))
    for name, model in models.items():
        model.cuda()
        try:
            # LocoTrack can produce NaNs with float16; preserve the existing exception.
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and name != "locotrack"):
                positions, visible = model(video.clone(), query_tensor.clone())
            if positions.shape != (*expected, 2) or visible.shape != expected:
                raise ValueError(f"Unexpected output shape from {name}: {positions.shape}, {visible.shape}")
            if not torch.isfinite(positions).all() or not torch.isfinite(visible).all():
                raise ValueError(f"Non-finite predictions from {name}; no teacher is silently dropped")
            tracks.append(positions.float().cpu())
            visibility.append((visible > 0.5).cpu())
            del positions, visible
        finally:
            if hasattr(model, "reset"):
                model.reset()
            model.cpu()
            torch.cuda.empty_cache()

    candidates = torch.stack(tracks, dim=-2).cuda()
    teacher_visibility = torch.stack(visibility, dim=-1)
    verifier.cuda()
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            ranks = verifier.verifier.efficient_forward(
                video, candidates, query_tensor, max_frames_per_pass=16,
            )
        if not torch.isfinite(ranks).all():
            raise ValueError("Non-finite verifier ranking logits")
        selected = ranks.argmax(dim=-1)
        gather_index = selected[..., None, None].expand(-1, -1, -1, 1, 2)
        pseudo_tracks = candidates.gather(-2, gather_index).squeeze(-2)
        return {
            "teacher_tracks": candidates[0].float().cpu().numpy(),
            "teacher_visibility": teacher_visibility[0].numpy(),
            "tracks": pseudo_tracks[0].float().cpu().numpy(),
            "pseudo_visibility": (teacher_visibility[0].float().mean(dim=-1) > 0.5).numpy(),
            "selected_teacher": selected[0].to(torch.int16).cpu().numpy(),
            "ranking_logits": ranks[0].float().cpu().numpy(),
        }
    finally:
        verifier.cpu()


def to_original_coordinates(result, queries, input_size, original_size):
    scale = np.asarray(original_size[::-1], dtype=np.float32) / np.asarray(input_size[::-1])
    for name in ("tracks", "teacher_tracks"):
        result[name] = ((result[name] + 0.5) * scale - 0.5).astype(np.float32)
    queries = queries.copy()
    queries[:, 1:] = (queries[:, 1:] + 0.5) * scale - 0.5
    # The query defines the position and visibility at initialization.
    result["tracks"][0] = queries[:, 1:]
    result["pseudo_visibility"][0] = True
    positions = result["tracks"]
    height, width = original_size
    result["label_valid"] = (np.isfinite(positions).all(axis=-1)
                             & (positions[..., 0] >= 0) & (positions[..., 0] < width)
                             & (positions[..., 1] >= 0) & (positions[..., 1] < height))
    result["queries"] = queries
    sx, sy = 1 / scale
    result["original_to_input"] = np.asarray(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float32,
    )
    return result


def save_clip(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays, metadata=np.asarray(metadata))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "ft_data_preparation/teachers.yaml")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help=f"Override a teacher/verifier checkpoint: {', '.join(CHECKPOINT_KEYS)}")
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/erf_x170/train")
    parser.add_argument("--event-root", type=Path, default=ROOT / "data/erf_x170/erf_time_surfaces/train/stride_4")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/erf_x170/erf_pseudo_labels/train")
    parser.add_argument("--sequences", nargs="+", help="Sequence names, e.g. 0000 0020")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; defaults to existing CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of independent sequence shards")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard index for this process")
    parser.add_argument("--rgb-stride", type=int, choices=(2, 4), default=2)
    parser.add_argument("--clip-length", type=int, default=48)
    parser.add_argument("--sequence-step", type=int, default=48, help="Clip start spacing in native RGB intervals")
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--input-size", nargs=2, type=int, default=(384, 512), metavar=("H", "W"))
    parser.add_argument("--max-clips", type=int, help="Limit clips per sequence for a small initial run")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Skip complete clips only when their metadata matches")
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("Need num-shards >= 1 and 0 <= shard-id < num-shards")
    if (args.clip_length < 16 or args.num_points < 3 or min(args.input_size) < 16
            or args.sequence_step <= 0 or args.sequence_step % 16
            or args.seed < 0 or (args.max_clips is not None and args.max_clips < 1)):
        parser.error("Need length >=16, points >=3, size >=16, nonnegative seed, positive max-clips, and step divisible by 16")
    return args


def main():
    args = parse_args()
    config = load_config(args)
    model_configs = {key: read_config(config[key]) for key in ("trackon2_config_path", "verifier_config_path")}
    if config.get("trackon2_vit_pretrained_path"):
        model_configs["trackon2_config_path"]["vit_pretrained_path"] = config["trackon2_vit_pretrained_path"]
    checkpoint_info = {}
    for name, key in CHECKPOINT_KEYS.items():
        path = Path(config[key])
        stat = path.stat()
        checkpoint_info[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    common = {
        "schema_version": 1, "teacher_names": list(TEACHER_NAMES), "checkpoints": checkpoint_info,
        "model_configs": model_configs, "cotracker_source": "facebookresearch/co-tracker:cotracker3_online",
        "trackon_memory_size": 24, "support_grid_size": 10,
        "rgb_stride": args.rgb_stride, "clip_length": args.clip_length, "sequence_step": args.sequence_step,
        "num_points": args.num_points, "input_size": list(args.input_size), "seed": args.seed, "amp": args.amp,
        "query_sampling": "first frame; 2/3 SIFT, 1/3 frame difference; distinct pixels; random fallback",
        "query_source_codes": {"random": 0, "sift": 1, "motion": 2}, "coordinates": "original image pixels (x,y)",
        "initial_label": "query position, visible", "visibility_rule": "strict majority of six teachers",
        "label_valid_rule": "finite selected position within original image; independent of pseudo_visibility",
    }
    sequences = ([args.input_root / name for name in args.sequences] if args.sequences else
                 sorted(path for path in args.input_root.iterdir() if path.is_dir()))
    if not sequences:
        raise ValueError(f"No sequences found: {args.input_root}")
    sequences = sorted(set(sequences))
    total_sequences = len(sequences)
    sequences = sequences[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(sequences)}/{total_sequences} sequences; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    jobs = []
    for sequence in sequences:
        event_path = args.event_root / f"{sequence.name}.h5"
        paths, times, image_size, time_source, estimated = read_sequence(sequence, event_path)
        span = (args.clip_length - 1) * args.rgb_stride
        starts = list(range(0, len(paths) - span, args.sequence_step))
        if not starts:
            raise ValueError(f"Sequence too short for the requested clip: {sequence}")
        for start in starts[:args.max_clips]:
            indices = start + np.arange(args.clip_length, dtype=np.int64) * args.rgb_stride
            path = args.output_root / f"stride_{args.rgb_stride}_t{args.clip_length}" / sequence.name / f"{start:06d}.npz"
            metadata = json.dumps({
                **common, "sequence": str(sequence.resolve()), "start_frame": start,
                "original_size": list(image_size), "event_cache": str(event_path.resolve()),
                "timestamp_source": time_source, "rgb_timestamps_estimated": estimated,
            }, sort_keys=True)
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f"Output exists; use --resume: {path}")
                with np.load(path, allow_pickle=False) as saved:
                    if (saved["metadata"].item() != metadata
                            or not np.array_equal(saved["frame_indices"], indices)
                            or not np.array_equal(saved["timestamps_us"], times[indices])
                            or saved["tracks"].shape != (args.clip_length, args.num_points, 2)):
                        raise ValueError(f"Existing labels do not match this run: {path}")
                continue
            jobs.append((paths, indices, times[indices], image_size, path, metadata))
    print(f"{len(jobs)} clips to generate", flush=True)
    if not jobs:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The existing teacher/verifier wrappers require CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    models, verifier = build_models(config)
    for number, (paths, indices, times, image_size, path, metadata) in enumerate(jobs, 1):
        digest = hashlib.sha256(f"{args.seed}:{path.parent.name}:{indices[0]}".encode()).digest()
        clip_seed = int.from_bytes(digest[:4], "little")
        torch.manual_seed(clip_seed)
        frames = read_clip(paths, indices, args.input_size, image_size)
        queries, sources = sample_queries(frames, args.num_points, np.random.default_rng(clip_seed))
        result = predict_clip(models, verifier, frames, queries, args.amp)
        result = to_original_coordinates(result, queries, args.input_size, image_size)
        save_clip(path, {**result, "query_sources": sources, "frame_indices": indices,
                         "timestamps_us": times, "teacher_names": np.asarray(TEACHER_NAMES)}, metadata)
        print(f"[{number}/{len(jobs)}] Saved {path}", flush=True)


if __name__ == "__main__":
    main()
