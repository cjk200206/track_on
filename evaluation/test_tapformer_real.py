"""Evaluate RGB or frame+event Track-On2 on TAPFormer's InivTAP/DrivTAP data.

Run from the repository root:
    python -m evaluation.test_tapformer_real --config config/test_tapformer_real.yaml
    python -m evaluation.test_tapformer_real --config config/test_tapformer_frame_event.yaml
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Avoid HDF5 file locking issues on some network filesystems.

import argparse
import csv
import json
import time
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.tapformer_real import load_real_tap_event_stream, load_real_tap_sequence
from evaluation.result_export import TAPVID_CSV_COLUMNS, write_summary_csv
from evaluation.track_video import render_tracking_video
from model.rgb_memory_predictor import RGBMemoryPredictor
from model.trackon_predictor import Predictor
from utils.coord_utils import get_points_on_a_grid
from utils.eval_utils import compute_tapvid_metrics
from utils.train_utils import load_args_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Evaluation config must be a YAML mapping")
    for key in ("dataset_root", "model_config", "checkpoint", "output_dir", "sequences"):
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
    if config.get("input_mode", "rgb") not in {"rgb", "frame_event"}:
        raise ValueError("input_mode must be 'rgb' or 'frame_event'")
    return config


def local_support_points(queries, frame_size, input_size, radius):
    """Sample eight neighbors per query; radius is in model-input pixels."""
    h, w = frame_size
    offsets = queries.new_tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)
                                  if x != 0 or y != 0])
    offsets *= queries.new_tensor([radius * w / input_size[1], radius * h / input_size[0]])
    points = (queries[:, None] + offsets).reshape(-1, 2)
    valid = ((points[:, 0] >= 0) & (points[:, 0] < w)
             & (points[:, 1] >= 0) & (points[:, 1] < h))
    return points[valid]


def resolve_memory_policy(config, memory_size):
    """Keep legacy FIFO unless a valid recent/RGB split is explicitly enabled."""
    policy = config.get("memory_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("memory_policy must be a mapping")
    kind = policy.get("type", "fifo")
    if kind == "fifo":
        return dict(type=kind, recent_size=memory_size, rgb_size=0)
    if kind != "recent_rgb":
        raise ValueError("memory_policy.type must be fifo or recent_rgb")
    recent_size = policy.get("recent_size", 24)
    rgb_size = policy.get("rgb_size", 48)
    if type(recent_size) is not int or recent_size < 1:
        raise ValueError("memory_policy.recent_size must be a positive integer")
    if type(rgb_size) is not int or rgb_size < 0:
        raise ValueError("memory_policy.rgb_size must be a nonnegative integer")
    if recent_size + rgb_size != memory_size:
        raise ValueError("memory_policy sizes must sum to inference_memory_size")
    return dict(type=kind, recent_size=recent_size, rgb_size=rgb_size)


@torch.no_grad()
def predict_sequence(predictor, sequence, device, support_grid_size, amp, event_stream=None,
                     local_support_radius=0, rgb_keep_every=1, input_stats=None):
    """Stream RGB/event steps, retaining predictions at annotated RGB times."""
    predictor.reset()
    if support_grid_size < 0 or local_support_radius < 0:
        raise ValueError("Support grid size and local radius must be nonnegative")
    if type(rgb_keep_every) is not int or rgb_keep_every < 1:
        raise ValueError("rgb_keep_every must be a positive integer")
    if rgb_keep_every > 1 and event_stream is None:
        raise ValueError("RGB subsampling requires frame_event input")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    query_xy = getattr(sequence, "query_xy", None)
    if query_xy is None:
        query_xy = sequence.tracks[0]
    n = len(query_xy)
    num_frames = len(sequence.image_paths)
    pred_tracks = np.empty((num_frames, n, 2), dtype=np.float32)
    pred_visible = np.empty((num_frames, n), dtype=bool)
    scored = np.zeros(num_frames, dtype=bool)
    elapsed = 0.0
    frame_shape = None
    autocast = torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()

    steps = (event_stream.steps() if event_stream is not None else
             ((image, None, index) for index, image in enumerate(sequence.frames())))
    previous_image = None
    original_updates = 0
    retained_indices = []
    for t, (image, events, score_index) in enumerate(steps):
        h, w = image.shape[:2]
        if frame_shape is None:
            frame_shape = (h, w)
        elif frame_shape != (h, w):
            raise ValueError(f"Frame size changed in {sequence.dataset}/{sequence.name}")
        new_rgb = image is not previous_image
        previous_image = image
        if new_rgb:
            original_updates += 1
            new_rgb = (original_updates - 1) % rgb_keep_every == 0
        if new_rgb:
            frame = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
            retained_indices.append(score_index)
        event_tensor = None
        if events is not None:
            event_tensor = torch.from_numpy(np.ascontiguousarray(events.transpose(2, 0, 1)))
            event_tensor = event_tensor.unsqueeze(0).to(device)
        queries = None
        if t == 0:
            queries = torch.from_numpy(np.asarray(query_xy, dtype=np.float32).copy()).to(device)
            if local_support_radius > 0:
                local = local_support_points(queries, (h, w), predictor.model.input_size,
                                             local_support_radius)
                queries = torch.cat((queries, local), dim=0)
            if support_grid_size > 0:
                grid = get_points_on_a_grid(support_grid_size, (h, w), device).squeeze(0)
                queries = torch.cat((queries, grid), dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast:
            positions, visible = predictor.forward_frame(
                frame, new_queries=queries, events=event_tensor, img_ifnew=new_rgb,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        if score_index is not None:
            pred_tracks[score_index] = positions[:n].float().cpu().numpy()
            pred_visible[score_index] = visible[:n].cpu().numpy()
            scored[score_index] = True
    if not scored.all():
        raise ValueError(f"Missing predictions at RGB timestamps in {sequence.dataset}/{sequence.name}")
    if input_stats is not None:
        def frequency(times):
            gaps = np.diff(np.asarray(times, dtype=np.int64))
            gaps = gaps[gaps > 0]
            return float(1e6 / np.median(gaps)) if gaps.size else None

        rgb_times = np.asarray([int(path.stem) for path in sequence.image_paths], dtype=np.int64)
        event_times = ([int(path.stem) for path in event_stream.event_paths]
                       if event_stream is not None else rgb_times)
        input_stats.update(
            rgb_keep_every=rgb_keep_every, input_steps=t + 1,
            original_rgb_updates=original_updates, retained_rgb_updates=len(retained_indices),
            retained_rgb_indices=retained_indices, original_rgb_hz=frequency(rgb_times),
            retained_rgb_hz=frequency(rgb_times[retained_indices]),
            event_update_hz=frequency(event_times),
            event_cache=str(event_stream.event_paths[0].parent.resolve()) if event_stream else None,
        )
    return pred_tracks, pred_visible, elapsed / num_frames


def tracking_metrics(sequence, tracks, visible, frame_indices=None):
    # Retain the original query frame so "first" excludes only initialization.
    indices = (np.arange(len(tracks)) if frame_indices is None else
               np.r_[0, np.asarray(frame_indices)[np.asarray(frame_indices) > 0]])
    n = sequence.tracks.shape[1]
    query_points = np.concatenate(
        (np.zeros((1, n, 1), dtype=np.float32), sequence.tracks[0, :, ::-1][None]), axis=-1
    )  # TAP-Vid queries use (t, y, x).
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = compute_tapvid_metrics(
            query_points,
            sequence.occluded[indices].T[None],
            sequence.tracks[indices].transpose(1, 0, 2)[None],
            np.logical_not(visible[indices]).T[None],
            tracks[indices].transpose(1, 0, 2)[None],
            "first",
        )
    return {key: float(value[0]) for key, value in metrics.items()}


def evaluate_sequence(predictor, sequence, device, support_grid_size, amp, event_stream=None,
                      local_support_radius=0, rgb_keep_every=1, input_stats=None):
    tracks, visible, seconds_per_frame = predict_sequence(
        predictor, sequence, device, support_grid_size, amp, event_stream, local_support_radius,
        rgb_keep_every, input_stats,
    )
    return tracking_metrics(sequence, tracks, visible), seconds_per_frame, tracks, visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/test_tapformer_real.yaml")
    parser.add_argument("--sequence", help="Evaluate one named sequence from the YAML config")
    parser.add_argument("--rgb-keep-every", type=int, help="Keep every K RGB arrivals; events and GT remain")
    parser.add_argument("--output-dir", help="Override the output directory")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    rgb_keep_every = (args.rgb_keep_every if args.rgb_keep_every is not None
                      else config.get("rgb_keep_every", 1))
    if type(rgb_keep_every) is not int or rgb_keep_every < 1:
        raise ValueError("rgb_keep_every must be a positive integer")
    frequency_test = args.rgb_keep_every is not None or "rgb_keep_every" in config
    if frequency_test and config.get("input_mode", "rgb") != "frame_event":
        raise ValueError("RGB frequency tests require frame_event input")
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_config_path = resolve_path(config["model_config"])
    if model_config_path.suffix == ".json":
        with open(model_config_path, encoding="utf-8") as file:
            model_args = Namespace(**json.load(file)["args"])
    else:
        model_args = load_args_from_yaml(str(model_config_path))
    if "inference_memory_size" in config:
        model_args.M_i = int(config["inference_memory_size"])
    memory_size = getattr(model_args, "M_i", 72)
    memory_policy = resolve_memory_policy(config, memory_size)
    if config.get("input_mode", "rgb") == "frame_event" and not getattr(model_args, "event_channels", 0):
        raise ValueError("frame_event input requires model_config with event_channels")
    if getattr(model_args, "vit_pretrained_path", None):
        model_args.vit_pretrained_path = str(resolve_path(model_args.vit_pretrained_path))
    predictor_type = Predictor
    memory_kwargs = {}
    if memory_policy["type"] == "recent_rgb":
        predictor_type = RGBMemoryPredictor
        memory_kwargs = {key: memory_policy[key] for key in ("recent_size", "rgb_size")}
    predictor = predictor_type(
        model_args,
        checkpoint_path=str(resolve_path(config["checkpoint"])),
        support_grid_size=0,  # Added once at the first streamed frame below.
        **memory_kwargs,
    ).to(device).eval()
    memory_metadata = dict(memory_policy=memory_policy["type"],
                           recent_memory_size=memory_policy["recent_size"],
                           rgb_memory_size=memory_policy["rgb_size"],
                           inference_memory_size=memory_size)
    print(f"Memory policy: {memory_metadata}", flush=True)

    output_dir = resolve_path(args.output_dir or config["output_dir"])
    if frequency_test and args.output_dir is None:
        output_dir /= f"k{rgb_keep_every}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if frequency_test:
        config["rgb_keep_every"] = rgb_keep_every
        if args.sequence:
            config["sequences"] = {dataset: [name for name in names if name == args.sequence]
                                   for dataset, names in config["sequences"].items()}
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(dict(model_args=vars(model_args), visibility_threshold=predictor.delta_v,
                           memory_policy=memory_policy, checkpoint=str(resolve_path(config["checkpoint"])),
                           checkpoint_mtime_ns=resolve_path(config["checkpoint"]).stat().st_mtime_ns),
                      file, indent=2)
    results = {}
    frequency_rows = []
    selected = False
    for dataset, names in config["sequences"].items():
        rows = []
        for name in names:
            if args.sequence and name != args.sequence:
                continue
            selected = True
            sequence = load_real_tap_sequence(
                resolve_path(config["dataset_root"]), dataset, name,
                drivtap_gt_mode=config.get("drivtap_gt_mode", "downsample_even"),
            )
            event_stream = None
            if config.get("input_mode", "rgb") == "frame_event":
                dt = config.get("event_dt", {}).get(dataset, "auto")
                if isinstance(dt, dict):
                    dt = dt[name]
                event_stream = load_real_tap_event_stream(sequence, dt)
            input_stats = {} if frequency_test else None
            metrics, seconds_per_frame, tracks, visible = evaluate_sequence(
                predictor, sequence, device, int(config.get("support_grid_size", 20)),
                bool(config.get("amp", True)), event_stream,
                local_support_radius=float(config.get("local_support_radius", 0)),
                rgb_keep_every=rgb_keep_every, input_stats=input_stats,
            )
            row = {"sequence": name, "frames": len(sequence.image_paths),
                   "points": sequence.tracks.shape[1], "metrics": metrics,
                   "seconds_per_frame": seconds_per_frame}
            row.update(support_grid_size=int(config.get("support_grid_size", 20)),
                       memory_policy=dict(memory_policy), inference_memory_size=memory_size,
                       local_support_radius=float(config.get("local_support_radius", 0)),
                       support_points=predictor.N - sequence.tracks.shape[1],
                       peak_memory_mb=(torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                       if device.type == "cuda" else 0.0))
            if event_stream is not None:
                row.update(event_steps=len(event_stream.event_paths), event_dt=event_stream.dt)
            if frequency_test:
                row.update(input_stats)
                retained = np.zeros(len(tracks), dtype=bool)
                retained[input_stats["retained_rgb_indices"]] = True
                row["group_metrics"] = {}
                for segment, mask in (("full", np.ones(len(tracks), dtype=bool)),
                                      ("retained_rgb", retained), ("hidden_rgb", ~retained)):
                    indices = np.flatnonzero(mask)
                    group = metrics if segment == "full" else tracking_metrics(sequence, tracks, visible, indices)
                    row["group_metrics"][segment] = {
                        key: value if np.isfinite(value) else None for key, value in group.items()
                    }
                    scored_indices = indices[indices > 0]
                    frequency_rows.append(dict(
                        dataset=dataset, sequence=name, segment=segment,
                        **{key: value for key, value in input_stats.items() if key != "retained_rgb_indices"},
                        eval_points=len(scored_indices) * len(sequence.tracks[0]),
                        visible_points=int((~sequence.occluded[scored_indices]).sum()),
                        AJ=group["average_jaccard"], delta_avg=group["average_pts_within_thresh"],
                        OA=group["occlusion_accuracy"],
                    ))
                print(f"RGB keep_every={rgb_keep_every}: "
                      f"{input_stats['retained_rgb_updates']}/{input_stats['original_rgb_updates']} RGB, "
                      f"{input_stats['input_steps']} input steps, cache={input_stats['event_cache']}", flush=True)
                with open(output_dir / "rgb_frequency_metrics.csv", "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(frequency_rows[0]))
                    writer.writeheader()
                    writer.writerows(frequency_rows)
            vis_cfg = config.get("visualization", {})
            if vis_cfg.get("enabled", False):
                video_format = vis_cfg.get("format", "mp4")
                video_path, preview_path = render_tracking_video(
                    sequence, tracks, visible, output_dir / "videos" / f"{dataset}_{name}.{video_format}",
                    fps=vis_cfg.get("fps", 10), trail_length=vis_cfg.get("trail_length", 8),
                    max_width=vis_cfg.get("max_width", 960),
                    gt_tracks=sequence.tracks, gt_visible=~sequence.occluded,
                )
                row["video_path"] = str(video_path)
                row["preview_path"] = str(preview_path)
                print(f"Video saved to {video_path}; preview saved to {preview_path}", flush=True)
            rows.append(row)
            print(f"{dataset}/{name}: AJ={metrics['average_jaccard'] * 100:.2f} "
                  f"delta_avg={metrics['average_pts_within_thresh'] * 100:.2f} "
                  f"OA={metrics['occlusion_accuracy'] * 100:.2f}", flush=True)
            if config.get("save_predictions", False):
                np.savez_compressed(output_dir / f"{dataset}_{name}.npz",
                                    tracks=tracks, visibility=visible)
        if rows:
            averages = {key: float(np.mean([row["metrics"][key] for row in rows]))
                        for key in rows[0]["metrics"]}
            results[dataset] = {"sequences": rows, "mean_metrics": averages}
            print(f"{dataset} mean ({len(rows)} sequences): "
                  f"AJ={averages['average_jaccard'] * 100:.2f} "
                  f"delta_avg={averages['average_pts_within_thresh'] * 100:.2f} "
                  f"OA={averages['occlusion_accuracy'] * 100:.2f}", flush=True)
    if not selected:
        raise ValueError(f"Sequence {args.sequence!r} is absent from the config")
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    csv_path = output_dir / config.get("csv_filename", "summary.csv")
    write_summary_csv(results, csv_path, TAPVID_CSV_COLUMNS, metric_scale=100,
                      checkpoint=config["checkpoint"],
                      model="trackon2_frame_event" if config.get("input_mode", "rgb") == "frame_event" else "trackon2",
                      metadata=memory_metadata)
    print(f"Results saved to {summary_path} and {csv_path}")


if __name__ == "__main__":
    main()
