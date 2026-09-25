"""Render FE_Kubric test clips with GT, RGB-only, fused, and zero-event tracks."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dataset.frame_event_kubric import FrameEventKubric
from evaluation.track_video import ffmpeg_command
from main_frame_event import first_visible_queries, load_event_checkpoint
from model.trackon import Track_On2
from utils.coord_utils import get_points_on_a_grid
from utils.train_utils import load_args_from_yaml


HEADER = 30


def colors_for_points(count):
    values = np.linspace(0, 255, count, endpoint=False, dtype=np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(values, cv2.COLORMAP_HSV).reshape(count, 3)


def inside(xy, width, height):
    return np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height


def draw_tracks(background, tracks, visible, query_times, frame, colors, tail_length):
    canvas = background.copy()
    height, width = canvas.shape[:2]
    for point in range(tracks.shape[1]):
        start = int(query_times[point])
        if frame < start or not visible[frame, point]:
            continue
        color = tuple(int(channel) for channel in colors[point])
        for previous in range(max(start, frame - tail_length), frame):
            a, b = tracks[previous, point], tracks[previous + 1, point]
            if visible[previous, point] and visible[previous + 1, point]:
                if inside(a, width, height) and inside(b, width, height):
                    cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 2)
        xy = tracks[frame, point]
        if inside(xy, width, height):
            center = tuple(np.rint(xy).astype(int))
            cv2.circle(canvas, center, 5, (0, 0, 0), -1)
            cv2.circle(canvas, center, 3, color, -1)
    return canvas


def event_background(event):
    # Match TAPFormer's ten-channel time-surface compression for display only.
    signed = event.copy()
    signed[1::2] *= -1
    index = np.abs(signed).argmax(axis=0)
    strongest = np.take_along_axis(signed, index[None], axis=0)[0]
    gray = np.clip((strongest + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def labeled_panel(image, label):
    height, width = image.shape[:2]
    panel = np.zeros((height + HEADER, width, 3), dtype=np.uint8)
    panel[HEADER:] = image
    cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def load_display_frames(sample, sequence, crop_size, use_clear):
    if not use_clear:
        return sample["video"].byte().permute(0, 2, 3, 1).numpy()[:, :, :, ::-1].copy()
    frames = []
    crop_h, crop_w = crop_size
    for index in sample["frame_indices"].tolist():
        path = sequence / "raw" / f"rgba_{index:05d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Clear display frame missing: {path}; use --no-display-clear")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        frames.append(cv2.cvtColor(rgb[y0:y0 + crop_h, x0:x0 + crop_w], cv2.COLOR_RGB2BGR))
    return np.stack(frames)


def render_sequence(sample, sequence, queries, predictions, output_dir, fps, tail_length, display_clear, gt_mode):
    query_times = queries[0, :, 0].long().cpu().numpy()
    video = load_display_frames(sample, sequence, sample["video"].shape[-2:], display_clear)
    events = sample["events"].numpy()
    gt_tracks = sample["tracks"].numpy()
    gt_visible = sample["visibility"].numpy()
    count = queries.shape[1]
    colors = colors_for_points(count)
    if gt_mode:
        first_tracks, first_visible = gt_tracks, gt_visible
        first_label = "GT tracks on RGB"
    else:
        first_tracks = np.repeat(queries[0, None, :, 1:].cpu().numpy(), len(video), axis=0)
        first_visible = np.zeros((len(video), count), dtype=bool)
        first_visible[0] = True
        first_label = "RGB / grid queries"

    height, width = video.shape[1:3]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{sequence.name}_comparison.mp4"
    preview_path = output_dir / f"{sequence.name}_preview.png"
    process = subprocess.Popen(
        ffmpeg_command(video_path, width * 3, (height + HEADER) * 2, fps),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for frame in range(len(video)):
            rgb = video[frame]
            event = event_background(events[frame])
            first = draw_tracks(rgb, first_tracks, first_visible, query_times, frame, colors, tail_length)
            rgb_pred = draw_tracks(rgb, *predictions["rgb"], query_times, frame, colors, tail_length)
            fused_pred = draw_tracks(rgb, *predictions["fused"], query_times, frame, colors, tail_length)
            zero_pred = draw_tracks(rgb, *predictions["zero"], query_times, frame, colors, tail_length)
            event_fused = draw_tracks(event, *predictions["fused"], query_times, frame, colors, tail_length)
            step = int(sample["frame_indices"][frame])
            top = np.hstack((
                labeled_panel(first, f"{first_label} | step {step}"),
                labeled_panel(rgb_pred, "RGB-only prediction"),
                labeled_panel(fused_pred, "RGB + event prediction"),
            ))
            bottom = np.hstack((
                labeled_panel(event, "Event time surface"),
                labeled_panel(zero_pred, "Zero-event prediction"),
                labeled_panel(event_fused, "Fused tracks on events"),
            ))
            composed = np.vstack((top, bottom))
            process.stdin.write(np.ascontiguousarray(composed).tobytes())
            if frame == len(video) // 2:
                if not cv2.imwrite(str(preview_path), composed):
                    raise RuntimeError(f"Could not write preview: {preview_path}")
    finally:
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            video_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr.strip()}")
    return video_path, preview_path, gt_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained frame_event_last.pt")
    parser.add_argument("--config", default="config/train_frame_event_full_frames.yaml")
    parser.add_argument("--data-root", default="data/event_kubric/test")
    parser.add_argument("--sequence", help="Exact Kubric sequence name, e.g. 00000000")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-length", type=int, help="Defaults to config T; keep 24 for comparable validation clips")
    parser.add_argument("--query-mode", choices=("gt", "grid"), default="gt")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid width when --query-mode grid")
    parser.add_argument("--max-points", type=int, default=64, help="Number of GT query tracks to render")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tail-length", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, help="Defaults to CHECKPOINT_DIRECTORY/visualizations")
    parser.add_argument("--display-clear", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw on clear RGB while inference still uses the training RGB")
    cli = parser.parse_args()
    if not cli.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {cli.checkpoint}")
    if cli.num_sequences < 1 or cli.start_index < 0 or cli.max_points < 1 or cli.grid_size < 1:
        parser.error("Sequence counts, grid size, and max points must be positive")
    if cli.fps < 1 or cli.tail_length < 0:
        parser.error("fps must be positive and tail length non-negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Track-On2 inference")

    args = load_args_from_yaml(cli.config)
    clip_length = args.T if cli.clip_length is None else cli.clip_length
    if clip_length < 2:
        parser.error("--clip-length must be at least 2")
    dataset = FrameEventKubric(
        root=cli.data_root, clip_length=clip_length, num_points=args.N,
        crop_size=args.input_size, frame_interval=args.frame_interval,
        training=False, seed=args.seed, eval_start=cli.clip_start,
    )
    if cli.sequence:
        selected = [i for i, path in enumerate(dataset.sequences) if path.name == cli.sequence]
        if not selected:
            parser.error(f"Sequence {cli.sequence} not found under {cli.data_root}")
    else:
        selected = list(range(cli.start_index, min(cli.start_index + cli.num_sequences, len(dataset))))
        if not selected:
            parser.error("start-index is beyond the test split")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = Track_On2(args).to(device)
    step = load_event_checkpoint(model, cli.checkpoint, device)
    model.eval()
    print(f"Loaded {cli.checkpoint} (step={step}); input frame interval={args.frame_interval}", flush=True)
    output_dir = cli.output_dir or cli.checkpoint.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in selected:
        sequence = dataset.sequences[index]
        sample = dataset[index]
        n = min(cli.max_points, sample["tracks"].shape[1])
        tracks = sample["tracks"][:, :n].unsqueeze(0).to(device)
        visible = sample["visibility"][:, :n].unsqueeze(0).to(device)
        if cli.query_mode == "gt":
            queries = first_visible_queries(tracks, visible)
            sample["tracks"] = sample["tracks"][:, :n]
            sample["visibility"] = sample["visibility"][:, :n]
        else:
            grid = get_points_on_a_grid(cli.grid_size, args.input_size, device=device)
            queries = torch.cat((torch.zeros_like(grid[:, :, :1]), grid), dim=-1)
        video = sample["video"].unsqueeze(0).to(device)
        events = sample["events"].unsqueeze(0).to(device)
        predictions = {}
        for mode in ("rgb", "fused", "zero"):
            mode_events = None if mode == "rgb" else events if mode == "fused" else torch.zeros_like(events)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                output = model(video, queries.clone(), p_mask=0, events=mode_events)
            predictions[mode] = (
                output["P"][0].float().cpu().numpy(),
                (output["V_logit"][0].sigmoid() >= args.delta_v).cpu().numpy(),
            )
            del output
        paths = render_sequence(sample, sequence, queries, predictions, output_dir,
                                cli.fps, cli.tail_length, cli.display_clear, cli.query_mode == "gt")
        video_path, preview_path, gt_mode = paths
        summary = {
            "sequence": sequence.name,
            "checkpoint": str(cli.checkpoint),
            "config": cli.config,
            "frame_indices": sample["frame_indices"].tolist(),
            "query_mode": cli.query_mode,
            "query_count": int(queries.shape[1]),
            "display_clear": cli.display_clear,
            "mean_visible_error_px": {},
        }
        if gt_mode:
            gt = sample["tracks"].numpy()
            mask = sample["visibility"].numpy() & (
                np.arange(len(gt))[:, None] >= queries[0, :, 0].cpu().numpy()[None, :]
            )
            for mode, (pred, _) in predictions.items():
                summary["mean_visible_error_px"][mode] = float(np.linalg.norm(pred - gt, axis=-1)[mask].mean())
        with (output_dir / f"{sequence.name}_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"{sequence.name}: {video_path} | {preview_path}", flush=True)


if __name__ == "__main__":
    main()
