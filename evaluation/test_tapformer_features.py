"""Evaluate RGB Track-On2 on TAPFormer's EDS and EC feature-track subsets.

Run from the repository root:
    python -m evaluation.test_tapformer_features --config config/test_tapformer_features.yaml
"""

import argparse
import json
import numpy as np
import torch
import yaml

from dataset.tapformer_features import load_feature_tap_sequence
from evaluation.test_tapformer_real import predict_sequence, resolve_path
from evaluation.result_export import FEATURE_CSV_COLUMNS, write_summary_csv
from evaluation.track_video import render_tracking_video
from model.trackon_predictor import Predictor
from utils.feature_track_metrics import average_feature_age
from utils.train_utils import load_args_from_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/test_tapformer_features.yaml")
    parser.add_argument("--sequence", help="Evaluate one named sequence from the YAML config")
    args = parser.parse_args()
    with open(resolve_path(args.config), encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("dataset_root", "model_config", "checkpoint", "output_dir", "sequences"):
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_args = load_args_from_yaml(str(resolve_path(config["model_config"])))
    if getattr(model_args, "vit_pretrained_path", None):
        model_args.vit_pretrained_path = str(resolve_path(model_args.vit_pretrained_path))
    predictor = Predictor(
        model_args,
        checkpoint_path=str(resolve_path(config["checkpoint"])),
        support_grid_size=0,
    ).to(device).eval()

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = config.get("feature_thresholds", list(range(1, 32)))
    results = {}
    selected = False
    for dataset, names in config["sequences"].items():
        rows = []
        for name in names:
            if args.sequence and name != args.sequence:
                continue
            selected = True
            sequence = load_feature_tap_sequence(resolve_path(config["dataset_root"]), dataset, name)
            tracks, visibility, seconds_per_frame = predict_sequence(
                predictor, sequence, device, int(config.get("support_grid_size", 20)),
                bool(config.get("amp", True)),
            )
            metrics = average_feature_age(tracks, sequence.frame_times, sequence.gt_rows, thresholds)
            row = {"sequence": name, "frames": len(sequence.image_paths),
                   "points": len(sequence.query_xy), "metrics": metrics,
                   "seconds_per_frame": seconds_per_frame}
            vis_cfg = config.get("visualization", {})
            if vis_cfg.get("enabled", False):
                video_format = vis_cfg.get("format", "mp4")
                video_path, preview_path = render_tracking_video(
                    sequence, tracks, visibility, output_dir / "videos" / f"{dataset}_{name}.{video_format}",
                    fps=vis_cfg.get("fps", 10), trail_length=vis_cfg.get("trail_length", 8),
                    max_width=vis_cfg.get("max_width", 960),
                )
                row["video_path"] = str(video_path)
                row["preview_path"] = str(preview_path)
                print(f"Video saved to {video_path}; preview saved to {preview_path}", flush=True)
            rows.append(row)
            print(f"{dataset}/{name}: error={metrics['mean_error']:.4f} "
                  f"FA={metrics['fa']:.4f} EFA={metrics['efa']:.4f}", flush=True)
            if config.get("save_predictions", False):
                np.savez_compressed(output_dir / f"{dataset}_{name}.npz", tracks=tracks,
                                    visibility=visibility, times=sequence.frame_times)
        if rows:
            averages = {key: float(np.mean([row["metrics"][key] for row in rows]))
                        for key in rows[0]["metrics"]}
            results[dataset] = {"sequences": rows, "mean_metrics": averages}
            print(f"{dataset} mean ({len(rows)} sequences): "
                  f"FA={averages['fa']:.4f} EFA={averages['efa']:.4f}", flush=True)
    if not selected:
        raise ValueError(f"Sequence {args.sequence!r} is absent from the config")
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    csv_path = output_dir / config.get("csv_filename", "summary.csv")
    write_summary_csv(results, csv_path, FEATURE_CSV_COLUMNS,
                      checkpoint=config["checkpoint"])
    print(f"Results saved to {summary_path} and {csv_path}")


if __name__ == "__main__":
    main()
