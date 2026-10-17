"""Inspect motion Frame–Event features and tracking on InivTAP / DrivTAP.

Run from the repository root, selecting a free GPU through CUDA_VISIBLE_DEVICES:
    python visualize_frame_event_features.py --sequence ID_Fast_Fruit Day_Fast
    python visualize_frame_event_features.py --dataset DrivTAP --rgb-stride 4

The default config is the motion-state + ETAP real-data evaluation config. Model
arguments come from the checkpoint's run_config.json when available. RGB stride
1 reproduces the original input stream; larger values deliberately withhold RGB
observations while retaining every event map and annotated RGB scoring time.

Outputs: summary.json/csv (tracking), step_metrics.csv, alignment_summary.csv,
diagnostics.json, fixed PCA bases, feature videos/previews, and GT/track videos.
Warp errors use 384x512 model-input pixels (or the configured input size), while
tracking errors use original-image pixels. GT alignment uses points visible in
both the retained anchor and current annotated RGB, excluding the query frame.
Intermediate event steps without GT have no fabricated/interpolated GT metrics.
Feature cosine measures agreement with the anchor descriptor at its GT point;
it is a diagnostic of representation consistency, not a tracking accuracy score.
"""

import argparse
import csv
import json
import subprocess
import time
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset.tapformer_real import load_real_tap_event_stream, load_real_tap_sequence
from evaluation.result_export import TAPVID_CSV_COLUMNS, write_summary_csv
from evaluation.track_video import ffmpeg_command, render_tracking_video
from model.event_motion import resize_warp, warp_features
from model.trackon_predictor import Predictor
from utils.coord_utils import get_points_on_a_grid
from utils.eval_utils import compute_tapvid_metrics
from utils.train_utils import load_args_from_yaml


ROOT = Path(__file__).resolve().parent
STAGES = ("anchor", "warped", "fused")
HEADER = 30
GT_METRICS = ("identity_gt_epe_input_px", "warp_gt_epe_input_px",
              *(f"{stage}_gt_cosine" for stage in STAGES))
STEP_METRICS = ("warp_mean_input_px", "warp_p95_input_px", "warp_oob_fraction",
                "event_nonzero_fraction", "event_residual_rgb_l2_ratio")


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def capture_features(model):
    """Observe exact pre-position content and FPN output; restore both hooks."""
    captured = {}
    add_position = model.backbone._add_position

    def record(features):
        captured["content"] = features
        return add_position(features)

    handle = model.fpn.register_forward_hook(lambda module, inputs, output: captured.update(fpn=output))
    try:
        with patch.object(model.backbone, "_add_position", new=record):
            yield captured
    finally:
        handle.remove()


def sample_points(feature, xy, input_size):
    height, width = input_size
    grid = ((xy + 0.5) / xy.new_tensor([width, height]) * 2 - 1)[None, None]
    return F.grid_sample(feature.float(), grid, align_corners=False, padding_mode="border")[0, :, 0].T


class FeatureView:
    """One fixed RGB PCA basis and a separate fixed FPN basis per sequence."""

    def __init__(self, maps, fpn, query, input_size, panel_width):
        height, width = input_size
        self.size = (max(1, round(height * 64 / width)), 64)
        self.panel_size = (panel_width, max(2, round(height * panel_width / width / 2) * 2))
        self.bases = {"rgb": self.fit(maps["anchor"]), "fpn": self.fit(fpn)}
        self.query = F.normalize(sample_points(maps["anchor"], query[None], input_size)[0], dim=0)

    def reduce(self, feature):
        return F.adaptive_avg_pool2d(feature.float(), self.size)[0]

    def fit(self, feature):
        values = self.reduce(feature).flatten(1).T
        # A deterministic subset keeps the SVD small without randomized PCA.
        values = values[::max(1, len(values) // 1024)]
        mean = values.mean(0)
        _, _, vh = torch.linalg.svd(values - mean, full_matrices=False)
        return mean, vh[:3].T

    def project(self, feature, basis):
        mean, components = self.bases[basis]
        values = self.reduce(feature).permute(1, 2, 0)
        return ((values - mean) @ components).cpu().numpy()

    def snapshot(self, maps, fpn, field, image, events, anchor_rgb, row, gt_xy, pred_xy):
        projections = {name: self.project(feature, "rgb") for name, feature in maps.items()}
        projections["fpn"] = self.project(fpn, "fpn")
        similarities = {
            name: (F.normalize(self.reduce(feature), dim=0) * self.query[:, None, None]).sum(0).cpu().numpy()
            for name, feature in maps.items()
        }
        residual = (maps["fused"] - maps["warped"]).float().norm(dim=1, keepdim=True)
        residual = F.adaptive_avg_pool2d(residual, self.size)[0, 0].cpu().numpy()
        event = np.stack((events[..., 1::2].max(-1), np.zeros(events.shape[:2]),
                          events[..., ::2].max(-1)), axis=-1)
        return dict(
            projections=projections, similarities=similarities, residual=residual,
            field=F.interpolate(field, size=self.size, mode="bilinear", align_corners=False)[0].cpu().numpy(),
            anchor=cv2.resize(anchor_rgb, self.panel_size), reference=cv2.resize(image, self.panel_size),
            events=cv2.resize(np.clip(event * 255, 0, 255).astype(np.uint8), self.panel_size),
            row=row, gt_xy=gt_xy, pred_xy=pred_xy,
        )

    def save_basis(self, path, limits):
        data = {f"{name}_{key}": tensor.cpu().numpy()
                for name, values in self.bases.items()
                for key, tensor in zip(("mean", "components"), values)}
        np.savez_compressed(path, **data, **limits)


def valid_points(points, height, width):
    return (np.isfinite(points).all(-1) & (points[..., 0] >= 0) & (points[..., 0] < width)
            & (points[..., 1] >= 0) & (points[..., 1] < height))


def alignment_metrics(maps, field, sequence, score_index, anchor_index, input_size, image_size):
    result = {"gt_points": 0, **dict.fromkeys(GT_METRICS)}
    if score_index is None or score_index == 0:
        return result
    height, width = input_size
    source = sequence.tracks[anchor_index]
    target = sequence.tracks[score_index]
    image_h, image_w = image_size
    valid = (~sequence.occluded[anchor_index] & ~sequence.occluded[score_index]
             & valid_points(source, image_h, image_w) & valid_points(target, image_h, image_w))
    if not valid.any():
        return result
    scale = field.new_tensor([width / image_w, height / image_h])
    source = torch.as_tensor(source[valid], device=field.device) * scale
    target = torch.as_tensor(target[valid], device=field.device) * scale
    truth = source - target  # Backward sampling: current point -> anchor point.
    prediction = sample_points(field, target, input_size)
    reference = sample_points(maps["anchor"], source, input_size)
    result.update(gt_points=int(valid.sum()), identity_gt_epe_input_px=truth.norm(dim=-1).mean().item(),
                  warp_gt_epe_input_px=(prediction - truth).norm(dim=-1).mean().item())
    for name, feature in maps.items():
        descriptor = sample_points(feature, target, input_size)
        result[f"{name}_gt_cosine"] = F.cosine_similarity(descriptor, reference, dim=-1).mean().item()
    return result


def aggregate(rows, dataset, sequence, group):
    result = dict(dataset=dataset, sequence=sequence, group=group, steps=len(rows),
                  gt_points=sum(row["gt_points"] for row in rows))
    for name in STEP_METRICS:
        result[name] = float(np.mean([row[name] for row in rows])) if rows else None
    for name in GT_METRICS:
        pairs = [(row[name], row["gt_points"]) for row in rows if row[name] is not None]
        result[name] = (sum(value * count for value, count in pairs) / sum(count for _, count in pairs)
                        if pairs else None)
    return result


def group_metrics(rows, dataset, sequence):
    groups = {"all": rows, "new_rgb": [row for row in rows if row["new_rgb"]],
              "between_rgb": [row for row in rows if not row["new_rgb"]]}
    groups.update({f"phase_{phase}": [row for row in rows if row["phase"] == phase]
                   for phase in sorted({row["phase"] for row in rows})})
    return [aggregate(group, dataset, sequence, name) for name, group in groups.items()]


def render_features(snapshots, view, output_dir, cli):
    # Joint limits preserve the meaning of PCA colors across time and RGB stages.
    limits = {}
    for basis, names in (("rgb", STAGES), ("fpn", ("fpn",))):
        values = np.concatenate([shot["projections"][name].reshape(-1, 3)
                                 for shot in snapshots for name in names])
        limits[f"{basis}_low"], limits[f"{basis}_high"] = np.percentile(values, [1, 99], axis=0)
    warp_max = max(1e-6, float(np.percentile(np.concatenate([
        np.linalg.norm(shot["field"], axis=0).ravel() for shot in snapshots]), 99)))
    residual_max = max(1e-6, float(np.percentile(np.concatenate([
        shot["residual"].ravel() for shot in snapshots]), 99)))
    view.save_basis(output_dir / "pca_basis.npz", {**limits, "warp_max_input_px": warp_max,
                                                 "residual_max": residual_max})
    process = None
    video_path = output_dir / "features.mp4"
    preview_path = output_dir / "features_preview.png"

    def panel(rgb, label, shot, marker=False):
        rgb = cv2.resize(rgb, view.panel_size)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if marker:
            for xy, color in ((shot["gt_xy"], (0, 255, 0)),
                              (shot["pred_xy"], (255, 0, 255))):
                if xy is not None and np.isfinite(xy).all():
                    x, y = np.rint(xy * np.array(view.panel_size)).astype(int)
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 12, 2)
        image = cv2.copyMakeBorder(image, HEADER, 0, 0, 0, cv2.BORDER_CONSTANT)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        font_scale = 0.48 * min(1, (view.panel_size[0] - 14) / max(1, text_width))
        cv2.putText(image, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def heatmap(values):
        return cv2.cvtColor(cv2.applyColorMap(np.clip(values * 255, 0, 255).astype(np.uint8),
                                             cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    try:
        for index, shot in enumerate(snapshots):
            field = shot["field"]
            hsv = np.zeros((*field.shape[1:], 3), dtype=np.uint8)
            hsv[..., 0] = ((np.arctan2(field[1], field[0]) + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(np.linalg.norm(field, axis=0) / warp_max * 255, 0, 255).astype(np.uint8)
            panels = [panel(shot["anchor"], "Retained RGB anchor", shot),
                      panel(shot["events"], "Events | GT green / pred magenta", shot, True),
                      panel(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
                            f"Backward warp | scale {warp_max:.1f} input px", shot)]
            for name in (*STAGES, "fpn"):
                basis = "fpn" if name == "fpn" else "rgb"
                values = (shot["projections"][name] - limits[f"{basis}_low"]) / np.maximum(
                    limits[f"{basis}_high"] - limits[f"{basis}_low"], 1e-6)
                label = "FPN PCA (separate basis)" if name == "fpn" else f"{name} RGB content PCA"
                image = np.clip(values * 255, 0, 255).astype(np.uint8)
                if name == "fpn":
                    fpn_panel = panel(image, label, shot, True)
                else:
                    panels.append(panel(image, label, shot, True))
            panels.extend(panel(heatmap((shot["similarities"][name] + 1) / 2),
                                f"{name} query cosine [-1, 1]", shot, True) for name in STAGES)
            panels.extend((panel(heatmap(shot["residual"] / residual_max), "Event residual L2 (fixed scale)", shot),
                           fpn_panel, panel(shot["reference"], "Latest available RGB reference", shot, True)))
            canvas = np.vstack([np.hstack(panels[i:i + 3]) for i in range(0, 12, 3)])
            row = shot["row"]
            title = (f"{row['dataset']}/{row['sequence']} | event {row['step']} | phase {row['phase']} | "
                     f"RGB age {row['rgb_age_ms']:.1f} ms | point {cli.point_index} | "
                     f"GT {'yes' if row['score_index'] is not None else 'unavailable'}")
            canvas = cv2.copyMakeBorder(canvas, HEADER, 0, 0, 0, cv2.BORDER_CONSTANT)
            title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            title_scale = 0.5 * min(1, (canvas.shape[1] - 16) / max(1, title_width))
            cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, title_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if index in {0, len(snapshots) // 2, len(snapshots) - 1} or cli.save_frames:
                cv2.imwrite(str(output_dir / f"step_{row['step']:06d}.png"), canvas)
            if index == len(snapshots) // 2:
                cv2.imwrite(str(preview_path), canvas)
            if cli.video:
                if process is None:
                    process = subprocess.Popen(ffmpeg_command(video_path, canvas.shape[1], canvas.shape[0], cli.fps),
                                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                process.stdin.write(np.ascontiguousarray(canvas).tobytes())
        if process is not None:
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"Feature video encoding failed: {error}")
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        video_path.unlink(missing_ok=True)
        raise
    finally:
        if process is not None:
            process.stderr.close()
    return str(video_path) if cli.video else None, str(preview_path)


@torch.no_grad()
def inspect_sequence(predictor, sequence, stream, config, cli, device, output_dir):
    predictor.reset()
    input_size = predictor.model.input_size
    scale_index = (4, 8, 16, 32).index(cli.scale)
    step_count = min(len(stream.event_paths), cli.max_steps or len(stream.event_paths))
    selected = set(np.linspace(0, step_count - 1, min(step_count, cli.max_vis_steps), dtype=int))
    # Explicitly include a first between-RGB step even when uniform sampling aliases the RGB cadence.
    if step_count > 1:
        selected.add(1)
    rows, snapshots, predictions, visibility = [], [], [], []
    view = None
    elapsed = 0.0
    use_amp = config.get("amp", True) and device.type == "cuda"
    anchor_index = anchor_step = 0
    image_size = None
    point_count = sequence.tracks.shape[1]
    if cli.point_index >= point_count:
        raise ValueError(f"{sequence.name}: point index must be less than {point_count}")
    with capture_features(predictor.model) as captured:
        for step, (image, events, score_index) in enumerate(stream.steps()):
            if step >= step_count:
                break
            height, width = image.shape[:2]
            if step == 0:
                image_size = (height, width)
            elif image_size != (height, width):
                raise ValueError("Image size changed within a sequence")
            new_rgb = score_index is not None and score_index % cli.rgb_stride == 0
            if new_rgb:
                anchor_index, anchor_step, anchor_rgb = score_index, step, image
                frame = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().to(device)
            event_tensor = torch.from_numpy(np.ascontiguousarray(events.transpose(2, 0, 1)))[None].to(device)
            queries = None
            if step == 0:
                queries = torch.from_numpy(sequence.tracks[0].copy()).to(device)
                support = int(config.get("support_grid_size", 20))
                if support > 0:
                    queries = torch.cat((queries, get_points_on_a_grid(support, (height, width), device).squeeze(0)))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
                positions, visible = predictor.forward_frame(frame, new_queries=queries, events=event_tensor,
                                                            img_ifnew=new_rgb)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            state = predictor.frontend_state
            anchor = state.anchor[scale_index]
            field = resize_warp(state.warp.float(), input_size)
            warped = warp_features(anchor, resize_warp(state.warp, anchor.shape[-2:]))
            fused = captured["content"][scale_index]
            maps = {"anchor": anchor, "warped": warped, "fused": fused}
            magnitude = field.norm(dim=1)
            y, x = torch.meshgrid(torch.arange(input_size[0], device=device),
                                  torch.arange(input_size[1], device=device), indexing="ij")
            sample_x, sample_y = x + field[0, 0], y + field[0, 1]
            oob = (sample_x < 0) | (sample_x > input_size[1] - 1) | (sample_y < 0) | (sample_y > input_size[0] - 1)
            timestamp = int(stream.event_paths[step].stem)
            row = dict(dataset=sequence.dataset, sequence=sequence.name, step=step, timestamp_us=timestamp,
                       score_index=score_index, anchor_index=anchor_index,
                       new_rgb=bool(new_rgb), phase=step - anchor_step,
                       rgb_age_ms=(timestamp - int(sequence.image_paths[anchor_index].stem)) / 1000,
                       warp_mean_input_px=magnitude.mean().item(),
                       warp_p95_input_px=torch.quantile(magnitude.flatten(), .95).item(),
                       warp_oob_fraction=oob.float().mean().item(), event_nonzero_fraction=float(np.mean(events != 0)),
                       event_residual_rgb_l2_ratio=(
                           (fused - warped).float().norm() / warped.float().norm().clamp_min(1e-8)).item())
            row.update(alignment_metrics(maps, field, sequence, score_index, anchor_index, input_size, image_size))
            rows.append(row)
            if score_index is not None:
                if score_index != len(predictions):
                    raise ValueError("Scored RGB frames must be consecutive")
                predictions.append(positions[:point_count].float().cpu().numpy())
                visibility.append(visible[:point_count].cpu().numpy())
            if step in selected:
                tokens = captured["fpn"][0, 0]
                fpn = tokens.T.reshape(1, tokens.shape[-1], input_size[0] // 4, input_size[1] // 4)
                if view is None:
                    query = field.new_tensor(sequence.tracks[0, cli.point_index]) * field.new_tensor(
                        [input_size[1] / width, input_size[0] / height])
                    view = FeatureView(maps, fpn, query, input_size, cli.panel_width)
                gt_xy = None if score_index is None or sequence.occluded[score_index, cli.point_index] else (
                    sequence.tracks[score_index, cli.point_index] / [width, height])
                pred_xy = positions[cli.point_index].float().cpu().numpy() / [width, height]
                snapshots.append(view.snapshot(maps, fpn, field, image, events, anchor_rgb, row, gt_xy, pred_xy))
            if (step + 1) % 100 == 0:
                print(f"{sequence.dataset}/{sequence.name}: {step + 1}/{step_count} event steps", flush=True)
    tracks, visible = np.stack(predictions), np.stack(visibility)
    count = len(tracks)
    if count < 2:
        raise ValueError("At least two scored RGB frames are required; increase --max-steps")
    partial = replace(sequence, image_paths=sequence.image_paths[:count], tracks=sequence.tracks[:count],
                      occluded=sequence.occluded[:count])
    query_points = np.concatenate((np.zeros((1, point_count, 1)), partial.tracks[0, :, ::-1][None]), axis=-1)
    metrics = compute_tapvid_metrics(query_points, partial.occluded.T[None], partial.tracks.transpose(1, 0, 2)[None],
                                     (~visible).T[None], tracks.transpose(1, 0, 2)[None], "first")
    metrics = {name: float(value[0]) for name, value in metrics.items()}
    valid = ~partial.occluded
    valid[0] = False
    errors = np.linalg.norm(tracks - partial.tracks, axis=-1)[valid]
    metrics["mean_error_px"] = float(errors.mean()) if len(errors) else None
    feature_video, feature_preview = render_features(snapshots, view, output_dir, cli)
    track_video = track_preview = None
    if cli.video:
        track_video, track_preview = render_tracking_video(
            partial, tracks, visible, output_dir / "tracks.mp4", fps=cli.fps,
            gt_tracks=partial.tracks, gt_visible=~partial.occluded, max_width=cli.panel_width * 2)
    np.savez_compressed(output_dir / "predictions.npz", tracks=tracks, visibility=visible)
    result = dict(sequence=sequence.name, frames=count, event_steps=len(rows), event_dt=stream.dt,
                  points=point_count, seconds_per_frame=elapsed / count, metrics=metrics,
                  video_path=str(track_video) if track_video else "",
                  preview_path=str(track_preview) if track_preview else "")
    detail = dict(full_sequence=count == len(sequence.image_paths) and len(rows) == len(stream.event_paths),
                  scored_frames=count, available_frames=len(sequence.image_paths), event_steps=len(rows),
                  feature_video=feature_video, feature_preview=feature_preview,
                  sampled_event_steps=[shot["row"]["step"] for shot in snapshots],
                  groups=group_metrics(rows, sequence.dataset, sequence.name))
    return result, detail, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/test_tapformer_real_frame_event_motion_etap.yaml")
    parser.add_argument("--checkpoint", help="Override the evaluation checkpoint")
    parser.add_argument("--model-config", help="Override auto-discovered checkpoint run_config.json")
    parser.add_argument("--output-dir", help="Defaults to outputs/frame_event_features/TIMESTAMP")
    parser.add_argument("--dataset", nargs="+", choices=("InivTAP", "DrivTAP"))
    parser.add_argument("--sequence", nargs="+", help="Evaluate only these sequence names")
    parser.add_argument("--device", help="PyTorch device; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--rgb-stride", type=int, default=1, help="Keep every Nth RGB; all events/GT times remain")
    parser.add_argument("--scale", type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument("--point-index", type=int, default=0, help="GT query point shown on feature panels")
    parser.add_argument("--max-vis-steps", type=int, default=120, help="Uniformly sampled event steps in feature video")
    parser.add_argument("--max-steps", type=int, help="Debug prefix; results are explicitly marked partial")
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=10, help="Playback rate of sampled feature steps / RGB track frames")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action="store_true",
                        help="Save all sampled panels as PNG, not just three snapshots")
    cli = parser.parse_args()
    if min(cli.rgb_stride, cli.max_vis_steps, cli.panel_width, cli.fps) < 1 or cli.point_index < 0:
        parser.error("Strides, sizes and FPS must be positive; point-index must be nonnegative")
    if cli.panel_width % 2 or (cli.max_steps is not None and cli.max_steps < 2):
        parser.error("panel-width must be even and max-steps must be at least 2")
    with resolve_path(cli.config).open() as handle:
        config = yaml.safe_load(handle)
    checkpoint = resolve_path(cli.checkpoint or config["checkpoint"])
    snapshot = checkpoint.parent / "run_config.json"
    model_config = resolve_path(cli.model_config) if cli.model_config else (
        snapshot if snapshot.is_file() else resolve_path(config["model_config"]))
    if model_config.suffix == ".json":
        model_args = Namespace(**json.loads(model_config.read_text())["args"])
    else:
        model_args = load_args_from_yaml(str(model_config))
    if getattr(model_args, "event_fusion_type", "simple") not in {"motion_state", "motion_pool"}:
        parser.error("A motion_state or motion_pool checkpoint is required to inspect RGB warp")
    model_args.M_i = int(config.get("inference_memory_size", getattr(model_args, "M_i", 72)))
    if getattr(model_args, "vit_pretrained_path", None):
        model_args.vit_pretrained_path = str(resolve_path(model_args.vit_pretrained_path))
    if getattr(model_args, "event_pretrained_path", None):
        model_args.event_pretrained_path = str(resolve_path(model_args.event_pretrained_path))
    selected = [(dataset, name) for dataset, names in config["sequences"].items() for name in names
                if (not cli.dataset or dataset in cli.dataset) and (not cli.sequence or name in cli.sequence)]
    missing = set(cli.sequence or []) - {name for _, name in selected}
    if not selected or missing:
        parser.error(f"No matching sequences or unknown sequence names: {sorted(missing)}")
    device_name = cli.device or config.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    predictor = Predictor(model_args, str(checkpoint), support_grid_size=0).to(device).eval()
    output_dir = (resolve_path(cli.output_dir) if cli.output_dir else
                  ROOT / "outputs" / "frame_event_features" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    model_config=str(model_config), model_args=vars(model_args), evaluation_config=config,
                    options=vars(cli), input_size=model_args.input_size, visibility_threshold=model_args.delta_v,
                    alignment_error_units="model input pixels", tracking_error_units="original image pixels",
                    feature_cosine_reference="anchor descriptor at corresponding GT point",
                    feature_video_timing="uniformly sampled event steps, played at --fps",
                    gt_policy="Only native annotated RGB times; no interpolated GT; exclude frame 0")
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Checkpoint: {checkpoint}\nModel config: {model_config}\nOutput: {output_dir}", flush=True)
    results, diagnostics, all_rows = {}, {}, []
    for dataset, name in selected:
        sequence = load_real_tap_sequence(resolve_path(config["dataset_root"]), dataset, name,
                                          config.get("drivtap_gt_mode", "downsample_even"))
        stream = load_real_tap_event_stream(sequence, config.get("event_dt", {}).get(dataset, "auto"))
        destination = output_dir / dataset / name
        destination.mkdir(parents=True, exist_ok=True)
        result, detail, rows = inspect_sequence(predictor, sequence, stream, config, cli, device, destination)
        results.setdefault(dataset, {"sequences": []})["sequences"].append(result)
        diagnostics[f"{dataset}/{name}"] = detail
        all_rows.extend(rows)
        write_csv(destination / "step_metrics.csv", rows)
        print(f"{dataset}/{name}: AJ={result['metrics']['average_jaccard'] * 100:.2f} "
              f"delta_avg={result['metrics']['average_pts_within_thresh'] * 100:.2f} "
              f"OA={result['metrics']['occlusion_accuracy'] * 100:.2f}; full={detail['full_sequence']}", flush=True)
        # Save completed sequences incrementally in case a later sequence fails.
        for data in results.values():
            data["mean_metrics"] = {}
            for key in result["metrics"]:
                values = [row["metrics"][key] for row in data["sequences"] if row["metrics"][key] is not None]
                data["mean_metrics"][key] = float(np.mean(values)) if values else None
        (output_dir / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
        write_summary_csv(results, output_dir / "summary.csv", TAPVID_CSV_COLUMNS, metric_scale=100,
                          checkpoint=str(checkpoint), model=f"frame_event_rgb_stride_{cli.rgb_stride}")
        write_csv(output_dir / "step_metrics.csv", all_rows)
        groups = [row for detail in diagnostics.values() for row in detail["groups"]]
        for ds in results:
            groups.extend(group_metrics([row for row in all_rows if row["dataset"] == ds], ds, "ALL"))
        write_csv(output_dir / "alignment_summary.csv", groups)
    print(f"Saved tracking and feature diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
