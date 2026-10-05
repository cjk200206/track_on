"""Evaluate RGB or frame+event Track-On2 on TAPFormer's InivTAP/DrivTAP data.

Run from the repository root:
    python -m evaluation.test_tapformer_real --config config/test_tapformer_real.yaml
    python -m evaluation.test_tapformer_real --config config/test_tapformer_frame_event.yaml
"""

import argparse
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


@torch.no_grad()
def predict_sequence(predictor, sequence, device, support_grid_size, amp, event_stream=None):
    """Stream RGB/event steps, retaining predictions at annotated RGB times."""
    predictor.reset()
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
    for t, (image, events, score_index) in enumerate(steps):
        h, w = image.shape[:2]
        if frame_shape is None:
            frame_shape = (h, w)
        elif frame_shape != (h, w):
            raise ValueError(f"Frame size changed in {sequence.dataset}/{sequence.name}")
        new_rgb = image is not previous_image
        if new_rgb:
            frame = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
            previous_image = image
        event_tensor = None
        if events is not None:
            event_tensor = torch.from_numpy(np.ascontiguousarray(events.transpose(2, 0, 1)))
            event_tensor = event_tensor.unsqueeze(0).to(device)
        queries = None
        if t == 0:
            queries = torch.from_numpy(np.asarray(query_xy, dtype=np.float32).copy()).to(device)
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
    return pred_tracks, pred_visible, elapsed / num_frames


def evaluate_sequence(predictor, sequence, device, support_grid_size, amp, event_stream=None):
    tracks, visible, seconds_per_frame = predict_sequence(
        predictor, sequence, device, support_grid_size, amp, event_stream
    )
    n = sequence.tracks.shape[1]
    query_points = np.concatenate(
        (np.zeros((1, n, 1), dtype=np.float32), sequence.tracks[0, :, ::-1][None]), axis=-1
    )  # TAP-Vid queries use (t, y, x).
    metrics = compute_tapvid_metrics(
        query_points,
        sequence.occluded.T[None],
        sequence.tracks.transpose(1, 0, 2)[None],
        np.logical_not(visible).T[None],
        tracks.transpose(1, 0, 2)[None],
        "first",
    )
    return {key: float(value[0]) for key, value in metrics.items()}, seconds_per_frame, tracks, visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/test_tapformer_real.yaml")
    parser.add_argument("--sequence", help="Evaluate one named sequence from the YAML config")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
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
    if config.get("input_mode", "rgb") == "frame_event" and not getattr(model_args, "event_channels", 0):
        raise ValueError("frame_event input requires model_config with event_channels")
    if getattr(model_args, "vit_pretrained_path", None):
        model_args.vit_pretrained_path = str(resolve_path(model_args.vit_pretrained_path))
    predictor = Predictor(
        model_args,
        checkpoint_path=str(resolve_path(config["checkpoint"])),
        support_grid_size=0,  # Added once at the first streamed frame below.
    ).to(device).eval()

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
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
                event_stream = load_real_tap_event_stream(sequence, dt)
            metrics, seconds_per_frame, tracks, visible = evaluate_sequence(
                predictor, sequence, device, int(config.get("support_grid_size", 20)),
                bool(config.get("amp", True)), event_stream,
            )
            row = {"sequence": name, "frames": len(sequence.image_paths),
                   "points": sequence.tracks.shape[1], "metrics": metrics,
                   "seconds_per_frame": seconds_per_frame}
            if event_stream is not None:
                row.update(event_steps=len(event_stream.event_paths), event_dt=event_stream.dt)
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
                      model="trackon2_frame_event" if config.get("input_mode", "rgb") == "frame_event" else "trackon2")
    print(f"Results saved to {summary_path} and {csv_path}")


if __name__ == "__main__":
    main()
