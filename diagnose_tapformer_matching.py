"""Inspect actual Track-On2 candidates without changing tracking decisions.

CUDA_VISIBLE_DEVICES=4 python diagnose_tapformer_matching.py --model rgb
CUDA_VISIBLE_DEVICES=5 python diagnose_tapformer_matching.py --model frame_event

GT is used only after inference, at native annotated RGB timestamps. All point
indices and frame/step indices are zero based. Distances use model-input pixels.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from argparse import Namespace
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from dataset.tapformer_real import load_real_tap_event_stream, load_real_tap_sequence
from evaluation.track_video import ffmpeg_command
from model.event_motion import resize_warp, warp_features
from model.trackon_predictor import Predictor
from utils.coord_utils import get_points_on_a_grid
from utils.eval_utils import compute_tapvid_metrics
from visualize_frame_event_features import FeatureView, capture_features


ROOT = Path(__file__).resolve().parent
CONFIGS = {"rgb": "config/test_tapformer_real.yaml",
           "frame_event": "config/test_tapformer_real_frame_event_rgb_aligned.yaml"}
# Matched windows for both models, chosen from the existing event predictions.
CASES = {"ID_Fast_Fruit": (10, 5), "OD_Basketball": (9, 2),
         "OD_Static": (43, 4), "Day_Fast": (30, 12),
         "Night_Fast": (2, 1), "Day_Bridge": (25, 0)}


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


@contextmanager
def capture_matching(model, count):
    """Observe the two actual correlations and heads; never replace their outputs."""
    captured = {}
    correlate = model.multiscale_correlation

    def record(*args, **kwargs):
        result = correlate(*args, **kwargs)
        captured.setdefault("correlations", []).append(array(result[0, :count]))
        captured.setdefault("queries", []).append(array(args[0][0, :count]))
        return result

    def rerank(module, inputs, output):
        captured["top_xy"] = array(output[1][0, :count])
        captured["top_uncertainty"] = array(output[2][0, :count])
        captured["top_scores"] = array(output[3][0, :count])

    def prediction(module, inputs, output):
        captured["offset"] = array(output[0][-1, :count])
        captured["visibility"] = array(output[1][:count].sigmoid())
        captured["uncertainty"] = array(output[2][:count].sigmoid())

    def query_layer(name, memory=False):
        def record_layer(module, inputs, output):
            current = output[:count, -1] if memory else output[0, :count]
            captured[name] = array(current)
        return record_layer

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, "multiscale_correlation", new=record))
        handles = [model.reranking_head.register_forward_hook(rerank),
                   model.prediction_head.register_forward_hook(prediction)]
        for layer in range(model.decoder_layer_num):
            handles.append(model.feature_attention[layer].register_forward_hook(query_layer(f"feature_{layer}")))
            handles.append(model.memory_attention[layer].register_forward_hook(query_layer(f"memory_{layer}", True)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()


def stream_steps(sequence, config, event_mode):
    if event_mode:
        return load_real_tap_event_stream(sequence, config["event_dt"][sequence.dataset]).steps()
    return ((image, None, index) for index, image in enumerate(sequence.frames()))


def prepare_step(predictor, image, events, new_rgb, previous_frame, sequence, support, first, device):
    frame = (torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().to(device)
             if new_rgb else previous_frame)
    event = (None if events is None else
             torch.from_numpy(np.ascontiguousarray(events.transpose(2, 0, 1)))[None].to(device))
    queries = None
    if first:
        queries = torch.from_numpy(sequence.tracks[0].copy()).to(device)
        if support > 0:
            grid = get_points_on_a_grid(support, image.shape[:2], device).squeeze(0)
            queries = torch.cat((queries, grid))
    return frame, dict(new_queries=queries, events=event, img_ifnew=new_rgb)


@torch.no_grad()
def verify_observer(predictor, sequence, config, event_mode, device):
    """Replay identical prefixes with/without both observers, including memory."""
    results = []
    for observe in (False, True):
        predictor.reset()
        frame, previous_image, records = None, None, []
        with ExitStack() as stack:
            if observe:
                capture = stack.enter_context(capture_matching(predictor.model, len(sequence.tracks[0])))
                stack.enter_context(capture_features(predictor.model))
            for step, (image, events, _) in enumerate(stream_steps(sequence, config, event_mode)):
                if step == 4:
                    break
                new_rgb = image is not previous_image
                frame, kwargs = prepare_step(predictor, image, events, new_rgb, frame, sequence,
                                             config.get("support_grid_size", 20), step == 0, device)
                previous_image = image
                if observe:
                    capture.clear()
                with torch.autocast("cuda", dtype=torch.float16, enabled=config.get("amp", True) and device.type == "cuda"):
                    positions, visible = predictor.forward_frame(frame, **kwargs)
                records.append((array(positions), visible.cpu().numpy().copy()))
                if observe and len(capture["correlations"]) != 2:
                    raise AssertionError("Expected exactly the actual c1 and c2 calls")
        results.append((records, array(predictor.point_memory[:predictor.N]),
                        predictor.temporal_mask[:predictor.N].cpu().numpy().copy(),
                        None if predictor.frontend_state is None else array(predictor.frontend_state.warp),
                        None if predictor.trajectory_history is None else array(predictor.trajectory_history[:predictor.N])))
    max_error = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(results[0][0], results[1][0]))
    identical = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                    for a, b in zip(results[0][0], results[1][0]))
    identical &= all((a is None and b is None) or np.array_equal(a, b)
                     for a, b in zip(results[0][1:], results[1][1:]))
    if not identical:
        raise AssertionError(f"Observer changed predictions/state; max position difference={max_error}")
    return dict(steps=len(results[0][0]), predictions_and_memory_identical=True, max_position_difference_px=max_error)


def point_metrics(c1, c2, top_xy, gt, grid, width, radius=8):
    """Score the GT neighborhood and the actual candidate ordering, in one space."""
    distance = np.linalg.norm(grid - gt, axis=-1)
    candidate_distance = np.linalg.norm(top_xy - gt, axis=-1)
    result = {"candidate_min_distance": float(candidate_distance.min())}
    gt_index = int(np.floor(gt[1] / 4)) * (width // 4) + int(np.floor(gt[0] / 4))
    for r in (4, 8):
        for k in (1, 4, 8, 16):
            result[f"recall{k}_r{r}"] = bool(np.any(candidate_distance[:k] <= r))
    for name, values in (("c1", c1), ("c2", c2)):
        best = int(values.argmax())
        result[f"{name}_best_distance"] = float(distance[best])
        result[f"{name}_gt_patch_rank"] = int(np.count_nonzero(values > values[gt_index]) + 1)
        for r in (4, 8):
            nearby = distance <= r
            best_gt = float(values[nearby].max())
            best_far = float(values[~nearby].max())
            result[f"{name}_gt_region_rank_r{r}"] = int(np.count_nonzero(values > best_gt) + 1)
            result[f"{name}_margin_r{r}"] = best_gt - best_far
        result[f"{name}_correct"] = bool(distance[best] <= radius)
    hit = result["recall16_r8"]
    result["stage"] = ("candidate_miss" if not hit and not result["c2_correct"] else
                       "rescued_after_candidate_miss" if not hit else
                       "c1_correct_c2_wrong" if result["c1_correct"] and not result["c2_correct"] else
                       "candidate_present_c2_wrong" if not result["c2_correct"] else "c2_correct")
    return result


def metric_check():
    grid = np.stack(np.meshgrid(np.arange(2, 34, 4), np.arange(2, 34, 4)), -1).reshape(-1, 2)
    c1 = -np.arange(64, dtype=np.float32)
    top = grid[:16]
    c2 = c1.copy()
    c2[-1] = 10
    result = point_metrics(c1, c2, top, grid[0], grid, 32)
    assert result["stage"] == "c1_correct_c2_wrong" and result["c2_margin_r8"] < 0
    result = point_metrics(c1, c1, top, grid[-1], grid, 32)
    assert result["stage"] == "candidate_miss" and not result["recall16_r8"]


def summarize(rows, dataset, sequence, group):
    valid = [row for row in rows if row.get("gt_valid")]
    result = dict(dataset=dataset, sequence=sequence, group=group, point_steps=len(rows), gt_point_steps=len(valid))
    if not valid:
        return result
    for key in ("error_input_px", "candidate_min_distance", "recall1_r4", "recall16_r4",
                "recall1_r8", "recall4_r8", "recall8_r8", "recall16_r8", "c1_correct", "c2_correct"):
        result[key] = float(np.mean([row[key] for row in valid]))
    for stage in ("candidate_miss", "rescued_after_candidate_miss", "c1_correct_c2_wrong",
                  "candidate_present_c2_wrong", "c2_correct"):
        result[stage + "_count"] = sum(row["stage"] == stage for row in valid)
    result["jump_onsets"] = sum(row.get("jump_onset", False) for row in valid)
    result["c2_outside_top16_fraction"] = float(np.mean([not row["c2_in_top16"] for row in valid]))
    return result


def timeline(path, rows, point, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["point"] == point]
    valid = [row for row in selected if row["gt_valid"]]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for key, label in (("error_input_px", "final error"), ("candidate_min_distance", "closest c1 candidate")):
        axes[0].plot([r["time_s"] for r in valid], [r[key] for r in valid], ".-", label=label)
    axes[0].set_ylabel("Input pixels")
    for name in ("c1", "c2"):
        axes[1].plot([r["time_s"] for r in valid], [r[name + "_margin_r8"] for r in valid], ".-", label=name + " GT-vs-far margin")
    axes[1].axhline(0, color="gray", linewidth=.7)
    axes[1].set_ylabel("Score margin")
    axes[2].plot([r["time_s"] for r in selected], [r["visibility_probability"] for r in selected], label="visibility")
    axes[2].plot([r["time_s"] for r in selected], [r["uncertainty_probability"] for r in selected], label="uncertainty")
    axes[2].set_ylim(-.05, 1.05)
    axes[2].set_xlabel("Elapsed seconds; dots with GT only at RGB timestamps")
    target = next((r["time_s"] for r in selected if r["score_index"] == frame), None)
    for ax in axes:
        if target is not None:
            ax.axvline(target, color="red", linestyle="--", linewidth=.8)
        ax.legend(loc="upper right")
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_case(destination, shots, rows, point, target, input_size):
    if not shots:
        return
    destination.mkdir(parents=True, exist_ok=True)
    height, width = input_size
    panel_w, panel_h, header = 384, 288, 30
    scale = np.array([panel_w / width, panel_h / height])
    raw = dict(c1=np.stack([s["c1"] for s in shots]), c2=np.stack([s["c2"] for s in shots]),
               top_xy=np.stack([s["top_xy"] for s in shots]),
               steps=np.array([s["row"]["step"] for s in shots]),
               timestamps_us=np.array([s["row"]["timestamp_us"] for s in shots]),
               gt_xy=np.stack([s["gt"] if s["gt"] is not None else np.full(2, np.nan) for s in shots]),
               pred_xy=np.stack([s["pred"] for s in shots]),
               offset=np.stack([s["offset"] for s in shots]),
               correlation_queries=np.stack([s["queries"] for s in shots]))
    for key in shots[0]["layer_queries"]:
        raw[key] = np.stack([s["layer_queries"][key] for s in shots])
    raw["rerank_uncertainty_logits"] = np.stack([s["top_uncertainty"] for s in shots])
    raw["rerank_score_logits"] = np.stack([s["top_scores"] for s in shots])
    np.savez_compressed(destination / "matching.npz", **raw)
    write_csv(destination / "steps.csv", [s["row"] for s in shots])
    # Fixed limits within the complete window, separately for c1 and c2.
    limits = {name: (float(raw[name].min()), float(raw[name].max())) for name in ("c1", "c2")}
    pca_limits = {}
    for basis in ("rgb", "fpn"):
        values = [value.reshape(-1, 3) for shot in shots for name, value in shot["pca"].items()
                  if (name == "fpn") == (basis == "fpn")]
        if values:
            pca_limits[basis] = np.percentile(np.concatenate(values), [1, 99], axis=0)
    write_json(destination / "display_limits.json", dict(correlations=limits, pca={k: v.tolist() for k, v in pca_limits.items()}))
    warp_limit = (max(float(np.percentile(np.concatenate([
        np.linalg.norm(s["warp"], axis=0).ravel() for s in shots]), 99)), 1e-6)
        if "warp" in shots[0] else None)

    def panel(image, title, shot, markers=False, candidates=False):
        image = cv2.resize(image, (panel_w, panel_h))
        if candidates:
            for xy in shot["top_xy"]:
                cv2.circle(image, tuple(np.rint(xy * scale).astype(int)), 3, (0, 230, 255), 1)
        if markers:
            for xy, color in ((shot["gt"], (0, 255, 0)), (shot["previous"], (255, 255, 0)),
                              (shot["pred"], (255, 0, 255))):
                if xy is not None and np.isfinite(xy).all():
                    cv2.drawMarker(image, tuple(np.rint(xy * scale).astype(int)), color, cv2.MARKER_CROSS, 12, 2)
        image = cv2.copyMakeBorder(image, header, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(image, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    process = None
    try:
        for index, shot in enumerate(shots):
            row = shot["row"]
            panels = [panel(cv2.cvtColor(shot["image"], cv2.COLOR_RGB2BGR),
                            f"step {row['step']} RGB={row['score_index']} point={point}", shot, True)]
            for name in ("c1", "c2"):
                lo, hi = limits[name]
                values = np.clip((shot[name] - lo) / max(hi - lo, 1e-8), 0, 1)
                image = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                panels.append(panel(image, name + " | yellow=c1 candidates; green=GT", shot, True, True))
            for name, value in shot["pca"].items():
                lo, hi = pca_limits["fpn" if name == "fpn" else "rgb"]
                image = np.clip((value - lo) / np.maximum(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                panels.append(panel(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), name + " PCA (fixed basis)", shot, True))
            if "warp" in shot:
                field = shot["warp"]
                mag = np.linalg.norm(field, axis=0)
                hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
                hsv[..., 0] = ((np.arctan2(field[1], field[0]) + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(mag / warp_limit * 255, 0, 255).astype(np.uint8)
                panels.append(panel(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), f"Backward warp | scale {warp_limit:.1f} input px", shot))
            text_image = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            text = [f"new RGB: {row['new_rgb']} | phase: {row['phase']}",
                    f"visibility: {row['visibility_probability']:.3f}",
                    f"uncertainty: {row['uncertainty_probability']:.3f}",
                    f"c2 in c1 top16: {row['c2_in_top16']}",
                    "GT: green; prediction: magenta", "previous prediction: cyan"]
            if row["gt_valid"]:
                text += [f"error: {row['error_input_px']:.1f} input px",
                         f"closest candidate: {row['candidate_min_distance']:.1f}", row["stage"]]
            else:
                text += ["No valid GT: no localization score"]
            for line, value in enumerate(text):
                cv2.putText(text_image, value, (8, 26 + line * 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
            panels.append(panel(text_image, "Diagnostics | elapsed time is in steps.csv", shot))
            while len(panels) % 3:
                panels.append(np.zeros_like(panels[0]))
            canvas = np.concatenate([np.concatenate(panels[i:i + 3], axis=1) for i in range(0, len(panels), 3)])
            if process is None:
                process = subprocess.Popen(ffmpeg_command(destination / "matching.mp4", canvas.shape[1], canvas.shape[0], 10), stdin=subprocess.PIPE)
            process.stdin.write(canvas.tobytes())
            if index in {0, len(shots) - 1} or row["score_index"] == target:
                cv2.imwrite(str(destination / f"step_{row['step']:06d}.png"), canvas)
            if row["score_index"] == target:
                cv2.imwrite(str(destination / "preview.png"), canvas)
                if shot["gt"] is not None:
                    origin = np.clip(np.rint(shot["gt"] - 32).astype(int), 0, [width - 64, height - 64])
                    crop_panels = []
                    images = {"RGB": cv2.cvtColor(shot["image"], cv2.COLOR_RGB2BGR)}
                    for name in ("c1", "c2"):
                        lo, hi = limits[name]
                        heat = np.clip((shot[name] - lo) / max(hi - lo, 1e-8), 0, 1)
                        images[name] = cv2.resize(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                                                 (width, height), interpolation=cv2.INTER_NEAREST)
                    for name, image in images.items():
                        crop = image[origin[1]:origin[1] + 64, origin[0]:origin[0] + 64]
                        zoom = panel(crop, name + " | 64x64 input-pixel GT crop", shot)
                        for xy, color in ((shot["gt"], (0, 255, 0)), (shot["pred"], (255, 0, 255))):
                            local = xy - origin
                            if np.all(local >= 0) and np.all(local < 64):
                                position = np.rint(local * [panel_w / 64, panel_h / 64]).astype(int)
                                cv2.drawMarker(zoom, (int(position[0]), int(position[1]) + header), color, cv2.MARKER_CROSS, 15, 2)
                        crop_panels.append(zoom)
                    cv2.imwrite(str(destination / "gt_crop.png"), np.concatenate(crop_panels, axis=1))
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Failed to encode diagnostic video")
    finally:
        if process is not None and process.poll() is None:
            process.stdin.close()
            process.wait()
    timeline(destination / "timeline.png", rows, point, target)


@torch.no_grad()
def inspect_sequence(predictor, sequence, config, cli, destination, device):
    predictor.reset()
    count = sequence.tracks.shape[1]
    height_in, width_in = predictor.model.input_size
    yy, xx = np.meshgrid(np.arange(height_in // 4) * 4 + 2, np.arange(width_in // 4) * 4 + 2, indexing="ij")
    grid = np.stack((xx, yy), -1).reshape(-1, 2)
    rows, tracks, visibility, shots = [], [], [], []
    frame, previous_image, previous_prediction, previous_gt_error = None, None, None, None
    previous_gt = None
    anchor_step = anchor_index = 0
    target, selected_point = CASES.get(sequence.name, (-100, 0))
    view = None
    with capture_matching(predictor.model, count) as captured, capture_features(predictor.model) as features:
        for step, (image, events, score_index) in enumerate(stream_steps(sequence, config, cli.model == "frame_event")):
            if cli.max_steps is not None and step >= cli.max_steps:
                break
            image_h, image_w = image.shape[:2]
            factor = np.array([width_in / image_w, height_in / image_h])
            new_rgb = image is not previous_image
            if new_rgb:
                anchor_step = step
                anchor_index = score_index
            frame, kwargs = prepare_step(predictor, image, events, new_rgb, frame, sequence,
                                         config.get("support_grid_size", 20), step == 0, device)
            previous_image = image
            captured.clear()
            with torch.autocast("cuda", dtype=torch.float16, enabled=config.get("amp", True) and device.type == "cuda"):
                positions, visible = predictor.forward_frame(frame, **kwargs)
            if len(captured["correlations"]) != 2:
                raise AssertionError("Expected exactly c1 and c2")
            c1, c2 = captured["correlations"]
            pred = array(positions[:count])
            pred_input = pred * factor
            best1, best2 = c1.argmax(-1), c2.argmax(-1)
            top_xy = captured["top_xy"]
            top_indices = ((top_xy[..., 1] // 4) * (width_in // 4) + top_xy[..., 0] // 4).astype(int)
            timestamp = (int(sequence.image_paths[score_index].stem) if events is None else
                         int(sequence.event_timestamps[step]))
            # A real timestamp for every step; GT remains absent between RGB frames.
            gt = None if score_index is None else sequence.tracks[score_index] * factor
            if score_index is not None:
                if score_index != len(tracks):
                    raise ValueError("Scored RGB frames must be consecutive")
                tracks.append(pred)
                visibility.append(visible[:count].cpu().numpy())
            step_rows = []
            for point in range(count):
                in_bounds = gt is not None and 0 <= gt[point, 0] < width_in and 0 <= gt[point, 1] < height_in
                valid = bool(score_index is not None and score_index > 0 and
                             not sequence.occluded[score_index, point] and in_bounds)
                row = dict(dataset=sequence.dataset, sequence=sequence.name, step=step,
                           timestamp_us=timestamp, time_s=(timestamp - int(sequence.image_paths[0].stem)) / 1e6,
                           score_index=score_index, anchor_index=anchor_index, new_rgb=new_rgb,
                           phase=step - anchor_step, point=point, gt_valid=valid,
                           gt_x=None if gt is None else float(gt[point, 0]),
                           gt_y=None if gt is None else float(gt[point, 1]),
                           pred_x=float(pred_input[point, 0]), pred_y=float(pred_input[point, 1]),
                           visibility_probability=float(captured["visibility"][point]),
                           predicted_visible=bool(visible[point]),
                           uncertainty_probability=float(captured["uncertainty"][point]),
                           c1_x=int(grid[best1[point], 0]), c1_y=int(grid[best1[point], 1]),
                           c2_x=int(grid[best2[point], 0]), c2_y=int(grid[best2[point], 1]),
                           c2_in_top16=bool(best2[point] in top_indices[point]),
                           prediction_step_distance=None if previous_prediction is None else
                           float(np.linalg.norm(pred_input[point] - previous_prediction[point])),
                           jump_onset=False)
                if valid:
                    row.update(point_metrics(c1[point], c2[point], top_xy[point], gt[point], grid, width_in))
                    row["error_input_px"] = float(np.linalg.norm(pred_input[point] - gt[point]))
                    row["error_original_px"] = float(np.linalg.norm(pred[point] - sequence.tracks[score_index, point]))
                    if previous_gt is not None and score_index > 1 and not sequence.occluded[score_index - 1, point]:
                        row["motion_error_input_px"] = float(np.linalg.norm(
                            (pred_input[point] - previous_gt[0][point]) - (gt[point] - previous_gt[1][point])))
                        row["jump_onset"] = bool(previous_gt_error[point] <= 8 and row["error_input_px"] > 16
                                                 and row["motion_error_input_px"] > 12)
                step_rows.append(row)
            rows.extend(step_rows)
            if sequence.name in CASES and not cli.no_details:
                tokens = features["fpn"][0, 0]
                fpn = tokens.T.reshape(1, tokens.shape[-1], height_in // 4, width_in // 4)
                content = features["content"][0]
                maps = {"anchor": content, "warped": content, "fused": content}
                state = predictor.frontend_state
                if state is not None:
                    anchor = state.anchor[0]
                    maps = dict(anchor=anchor, warped=warp_features(anchor, resize_warp(state.warp, anchor.shape[-2:])), fused=content)
                if view is None:
                    query = torch.as_tensor(sequence.tracks[0, selected_point] * factor,
                                            device=content.device, dtype=torch.float32)
                    view = FeatureView(maps, fpn, query, predictor.model.input_size, 384)
                # Include every event step in the selected RGB interval, with no fabricated GT.
                if max(0, target - 5) <= anchor_index <= target + 5:
                    pca = {name: view.project(value, "rgb") for name, value in maps.items()
                           if state is not None or name == "anchor"}
                    pca["fpn"] = view.project(fpn, "fpn")
                    shot = dict(row=step_rows[selected_point], image=cv2.resize(image, (width_in, height_in)),
                                c1=c1[selected_point].reshape(height_in // 4, width_in // 4),
                                c2=c2[selected_point].reshape(height_in // 4, width_in // 4),
                                top_xy=top_xy[selected_point], offset=captured["offset"][selected_point],
                                gt=None if gt is None or sequence.occluded[score_index, selected_point] else gt[selected_point],
                                pred=pred_input[selected_point], previous=None if previous_prediction is None else previous_prediction[selected_point],
                                queries=np.stack([q[selected_point] for q in captured["queries"]]),
                                layer_queries={k: v[selected_point] for k, v in captured.items()
                                               if k.startswith(("feature_", "memory_"))},
                                top_uncertainty=captured["top_uncertainty"][selected_point],
                                top_scores=captured["top_scores"][selected_point], pca=pca)
                    if state is not None:
                        shot["warp"] = array(resize_warp(state.warp.float(), (height_in, width_in)))[0]
                    shots.append(shot)
            previous_prediction = pred_input
            if gt is not None:
                previous_gt = pred_input, gt
                previous_gt_error = np.linalg.norm(pred_input - gt, axis=-1)
            if step and step % 100 == 0:
                print(f"{cli.model} {sequence.dataset}/{sequence.name}: step {step}", flush=True)
    write_csv(destination / "point_steps.csv", rows)
    write_csv(destination / "jump_onsets.csv", [row for row in rows if row["jump_onset"]])
    tracks, visibility = np.stack(tracks), np.stack(visibility)
    np.savez_compressed(destination / "predictions.npz", tracks=tracks, visibility=visibility)
    summary = summarize(rows, sequence.dataset, sequence.name, "all")
    summary.update(frames=len(tracks), steps=rows[-1]["step"] + 1,
                   image_width=image_w, image_height=image_h, partial=len(tracks) != len(sequence.tracks))
    if len(tracks) == len(sequence.tracks):
        query = np.concatenate((np.zeros((1, count, 1)), sequence.tracks[0, :, ::-1][None]), -1)
        metrics = compute_tapvid_metrics(query, sequence.occluded.T[None], sequence.tracks.transpose(1, 0, 2)[None],
                                         (~visibility).T[None], tracks.transpose(1, 0, 2)[None], "first")
        summary.update({k: float(v[0]) for k, v in metrics.items()})
    groups = [summarize([r for r in rows if r["new_rgb"] == flag], sequence.dataset, sequence.name, label)
              for flag, label in ((True, "new_rgb"), (False, "between_rgb"))]
    write_csv(destination / "groups.csv", groups)
    write_json(destination / "summary.json", summary)
    if shots:
        detail = destination / f"point_{selected_point}_rgb_{target}"
        render_case(detail, shots, rows, selected_point, target, predictor.model.input_size)
        summary["detail"] = str(detail)
    return summary, groups


def report(output, summaries):
    lines = ["# Track-On2 candidate diagnostics", "", "Indices are zero based. Distances/radii use model-input pixels.",
             "GT diagnostics exclude frame 0, occluded and out-of-image GT. No interpolated GT.",
             "Recall uses the actual ordered c1 candidates; c2 is a new global search.", "",
             "| Dataset/sequence | GT samples | R16 @8 | c1 correct | c2 correct | candidate miss | c1 good→c2 bad | jump onsets |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['dataset']}/{s['sequence']} | {s['gt_point_steps']} | {s['recall16_r8']:.1%} | "
                     f"{s['c1_correct']:.1%} | {s['c2_correct']:.1%} | {s['candidate_miss_count']} | "
                     f"{s['c1_correct_c2_wrong_count']} | {s['jump_onsets']} |")
    for s in summaries:
        if "detail" in s:
            detail = Path(s["detail"]).relative_to(output)
            lines += ["", f"## {s['dataset']}/{s['sequence']}", "",
                      f"[Video]({detail}/matching.mp4) · [Raw correlations/query vectors]({detail}/matching.npz)", "",
                      f"![Matching]({detail}/preview.png)", "", f"![Timeline]({detail}/timeline.png)"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--config")
    parser.add_argument("--model-config", help="Explicit model arguments override the checkpoint's adjacent snapshot")
    parser.add_argument("--output-dir", default="outputs/matching_diagnostics")
    parser.add_argument("--sequence", nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    cli = parser.parse_args()
    if cli.max_steps is not None and cli.max_steps < 2:
        parser.error("--max-steps must be at least 2")
    metric_check()
    config = yaml.safe_load(resolve(cli.config or CONFIGS[cli.model]).read_text())
    checkpoint = resolve(config["checkpoint"])
    snapshot = checkpoint.parent / "run_config.json"
    model_config = (resolve(cli.model_config) if cli.model_config else
                    snapshot if snapshot.is_file() else resolve(config["model_config"]))
    saved = json.loads(model_config.read_text()) if model_config.suffix == ".json" else yaml.safe_load(model_config.read_text())
    args = Namespace(**saved["args"] if model_config.suffix == ".json" else saved)
    args.M_i = int(config.get("inference_memory_size", getattr(args, "M_i", 72)))
    for name in ("vit_pretrained_path", "event_pretrained_path"):
        if getattr(args, name, None):
            setattr(args, name, str(resolve(getattr(args, name))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(args, str(checkpoint), support_grid_size=0).to(device).eval()
    output = resolve(cli.output_dir) / cli.model
    output.mkdir(parents=True, exist_ok=True)
    selected = [(dataset, name) for dataset, names in config["sequences"].items() for name in names
                if not cli.sequence or name in cli.sequence]
    if not selected or set(cli.sequence or []) - {name for _, name in selected}:
        parser.error("Unknown or empty sequence selection")
    metadata = dict(model=cli.model, checkpoint=str(checkpoint), checkpoint_size=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns, model_config=str(model_config),
                    model_args=vars(args), evaluation_config=config, options=vars(cli),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    index_base=0, distance_units="model input pixels", radius_input_px=[4, 8],
                    gt_policy="native RGB timestamps only; exclude initialization/occlusion/out-of-bounds; Driv even GT",
                    jump_rule="previous error<=8, current error>16, motion error>12; consecutive visible RGB GT",
                    case_windows={name: list(value) for name, value in CASES.items()})
    write_json(output / "run_config.json", metadata)
    summaries, groups = [], []
    for index, (dataset, name) in enumerate(selected):
        sequence = load_real_tap_sequence(resolve(config["dataset_root"]), dataset, name,
                                          config.get("drivtap_gt_mode", "downsample_even"))
        if cli.model == "frame_event":
            stream = load_real_tap_event_stream(sequence, config["event_dt"][dataset])
            sequence.event_timestamps = np.array([int(path.stem) for path in stream.event_paths])
        if index == 0:
            metadata["observer_verification"] = verify_observer(predictor, sequence, config, cli.model == "frame_event", device)
            write_json(output / "run_config.json", metadata)
            print(f"Observer verification: {metadata['observer_verification']}", flush=True)
            if cli.verify_only:
                return
        destination = output / dataset / name
        destination.mkdir(parents=True, exist_ok=True)
        summary, sequence_groups = inspect_sequence(predictor, sequence, config, cli, destination, device)
        summaries.append(summary)
        groups.extend(sequence_groups)
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "groups.csv", groups)
        write_json(output / "summary.json", summaries)
        report(output, summaries)
        print(f"Finished {dataset}/{name}: R16@8={summary['recall16_r8']:.3f}, c2 correct={summary['c2_correct']:.3f}", flush=True)
    if checkpoint.stat().st_mtime_ns != metadata["checkpoint_mtime_ns"]:
        raise RuntimeError("Checkpoint changed during diagnosis; loaded model results are saved, but snapshot identity must be checked")
    metadata["complete"] = cli.max_steps is None
    write_json(output / "run_config.json", metadata)


if __name__ == "__main__":
    main()
