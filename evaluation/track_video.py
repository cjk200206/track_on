"""Render RGB/event point tracks to browser-friendly H.264 MP4 or VP8 WebM."""

import subprocess
from pathlib import Path

import cv2
import numpy as np


HEADER_HEIGHT = 30


def _colors(count):
    hues = np.linspace(0, 179, count, endpoint=False, dtype=np.uint8)
    hsv = np.stack((hues, np.full(count, 210, np.uint8),
                    np.full(count, 255, np.uint8)), axis=1).reshape(1, count, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(count, 3)


def _event_image(events):
    """Project five interleaved polarity bins: positive red, negative blue."""
    if events.ndim != 3 or events.shape[-1] != 10:
        raise ValueError("Event visualization requires HWC time surfaces with 10 channels")
    image = np.zeros((*events.shape[:2], 3), dtype=np.uint8)
    image[..., 0] = (events[..., 1::2].max(axis=-1).clip(0, 1) * 255).astype(np.uint8)
    image[..., 2] = (events[..., 0::2].max(axis=-1).clip(0, 1) * 255).astype(np.uint8)
    return image


def _panel(rgb, tracks, visible, t, colors, trail_length, label, scale, show_ids=False):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    panel = np.zeros((height + HEADER_HEIGHT, width, 3), dtype=np.uint8)
    panel[HEADER_HEIGHT:] = image
    cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(tracks.shape[1]):
        color = tuple(int(value) for value in colors[i])
        for previous in range(max(0, t - trail_length), t):
            a, b = tracks[previous, i], tracks[previous + 1, i]
            if not (visible[previous, i] and visible[previous + 1, i]):
                continue
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            a, b = np.rint(a * scale).astype(int), np.rint(b * scale).astype(int)
            if all(0 <= xy[0] < width and 0 <= xy[1] < height for xy in (a, b)):
                cv2.line(panel, (a[0], a[1] + HEADER_HEIGHT),
                         (b[0], b[1] + HEADER_HEIGHT), color, 2, cv2.LINE_AA)
        point = tracks[t, i]
        if visible[t, i] and np.isfinite(point).all():
            x, y = np.rint(point * scale).astype(int)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(panel, (x, y + HEADER_HEIGHT), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(panel, (x, y + HEADER_HEIGHT), 3, color, -1, cv2.LINE_AA)
                if show_ids:
                    cv2.putText(panel, str(i), (x + 6, y + HEADER_HEIGHT - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return panel


def _memory_panel(width, diagnostic, frame):
    """Post-write slots for every annotated point, oldest to newest within each bank."""
    step = diagnostic["score_steps"][frame]
    states = diagnostic["slot_states"][step]
    panel = np.full((62 + 18 * len(states), width, 3), 24, dtype=np.uint8)
    palette = [(65, 65, 65), (220, 155, 55), (85, 180, 90), (50, 140, 245), (60, 60, 230)]
    title = f'{diagnostic["mode"]} | step {step} | after write | older RGB / recent | GT held between RGB'
    cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (235, 235, 235), 1, cv2.LINE_AA)
    labels = ["empty", "RGB / GT visible", "event / loc. unscored", "GT occluded"]
    if "localization_error_px" in diagnostic:
        labels.append(f'RGB error > {diagnostic["error_threshold_px"]:g}px')
    for index, label in enumerate(labels):
        x = 10 + index * 205
        cv2.rectangle(panel, (x, 29), (x + 10, 39), palette[index], -1)
        cv2.putText(panel, label, (x + 15, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(panel, 'White border: newest slot; errors scored only at RGB', (1100, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225, 225, 225), 1, cv2.LINE_AA)
    left = 300
    cell = min(22, (width - left - 10) / states.shape[1])
    for point, row in enumerate(states):
        y = 56 + point * 18
        gt = int(not diagnostic["gt_occluded"][step, point])
        probability = diagnostic["probability"][step, point]
        label = f'P{point:02d} GT:{gt} p:{probability:.2f}'
        if "localization_error_px" in diagnostic:
            error = diagnostic["localization_error_px"][frame, point]
            label += f' e:{error:.1f}' if np.isfinite(error) else ' e:--'
        cv2.putText(panel, label, (10, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (225, 225, 225), 1, cv2.LINE_AA)
        for slot, state in enumerate(row):
            x, right = int(left + slot * cell), int(left + (slot + 1) * cell) - 2
            cv2.rectangle(panel, (x, y), (right, y + 12), palette[state], -1)
            if slot == len(row) - 1:
                cv2.rectangle(panel, (x, y), (right, y + 12), (255, 255, 255), 1)
        boundary = int(left + diagnostic["rgb_size"] * cell) - 1
        cv2.line(panel, (boundary, y - 2), (boundary, y + 14), (255, 255, 255), 1)
    return panel


def render_localization_timeline(sequence, errors, probability, output_path, error_threshold,
                                 visibility_threshold):
    """Compare localization and visibility at GT timestamps, without scoring occluded positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = errors.shape[1]
    frames = np.arange(len(errors))
    figure, axes = plt.subplots(count, 1, figsize=(13, 1.45 * count), sharex=True, squeeze=False)
    for point, axis in enumerate(axes[:, 0]):
        error = errors[:, point]
        bad = np.isfinite(error) & (error > error_threshold)
        axis.plot(frames, error, color="tab:blue", linewidth=1, label="position error")
        axis.scatter(frames[bad], error[bad], color="tab:red", s=7, label="error > threshold")
        axis.axhline(error_threshold, color="tab:red", linewidth=0.7, linestyle=":")
        axis.fill_between(frames, 0, 1, where=sequence.occluded[:, point],
                          transform=axis.get_xaxis_transform(), color="0.85", step="mid", label="GT occluded")
        axis.set_ylabel(f"P{point:02d}\nerror (px)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        right = axis.twinx()
        right.plot(frames, probability[:, point], color="tab:green", linewidth=1, alpha=0.8,
                   label="visibility")
        right.axhline(visibility_threshold, color="tab:green", linewidth=0.7, linestyle="--")
        right.set_ylim(-0.03, 1.03)
        right.set_yticks([0, visibility_threshold, 1])
        right.set_ylabel("visibility", color="tab:green")
        if point == 0:
            handles, labels = axis.get_legend_handles_labels()
            extra, names = right.get_legend_handles_labels()
            figure.legend(handles + extra, labels + names, loc="upper center", ncol=4,
                          bbox_to_anchor=(0.5, 0.975))
    axes[-1, 0].set_xlabel("RGB frame (0-based)")
    figure.suptitle(f"{sequence.dataset}/{sequence.name} | original pixels | "
                   f"error threshold={error_threshold:g}, visibility threshold={visibility_threshold:g}", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def ffmpeg_command(path, width, height, fps):
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
              f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an"]
    if path.suffix == ".mp4":
        return common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    if path.suffix == ".webm":
        return common + ["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "5",
                         "-b:v", "2M", "-pix_fmt", "yuv420p", str(path)]
    raise ValueError("Video format must be mp4 or webm")


def render_tracking_video(sequence, tracks, visible, output_path, fps=10, trail_length=8,
                          max_width=960, gt_tracks=None, gt_visible=None, memory_diagnostic=None,
                          event_stream=None):
    """Stream tracks and optional event panels at RGB timestamps, plus a PNG preview."""
    output_path = Path(output_path)
    if output_path.suffix not in {".mp4", ".webm"}:
        raise ValueError("Video format must be mp4 or webm")
    if fps <= 0 or trail_length < 0 or max_width <= 0:
        raise ValueError("fps/max_width must be positive and trail_length nonnegative")
    frame_count = len(sequence.image_paths)
    point_count = tracks.shape[1]
    if tracks.shape != (frame_count, point_count, 2) or visible.shape != (frame_count, point_count):
        raise ValueError("Prediction dimensions do not match the RGB sequence")
    if (gt_tracks is None) != (gt_visible is None):
        raise ValueError("GT tracks and visibility must be provided together")
    if gt_tracks is not None and (gt_tracks.shape != tracks.shape or gt_visible.shape != visible.shape):
        raise ValueError("GT dimensions do not match the predictions")

    if event_stream is None:
        frames = ((rgb, None) for rgb in sequence.frames())
    else:
        if [index for index in event_stream.score_indices if index is not None] != list(range(frame_count)):
            raise ValueError("Event stream must contain each RGB timestamp in order")
        frames = ((rgb, events) for rgb, events, index in event_stream.steps() if index is not None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _colors(point_count)
    process = None
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    try:
        for t, (rgb, events) in enumerate(frames):
            scale = min(1.0, max_width / rgb.shape[1])
            prediction = _panel(rgb, tracks, visible, t, colors, trail_length,
                                f"Track-On2 | frame {t}", scale, memory_diagnostic is not None)
            if gt_tracks is not None:
                ground_truth = _panel(rgb, gt_tracks, gt_visible, t, colors, trail_length,
                                      f"GT | frame {t}", scale, memory_diagnostic is not None)
                canvas = np.hstack((ground_truth, prediction))
            else:
                canvas = prediction
            if events is not None:
                event_panel = _panel(_event_image(events), tracks, visible, t, colors, trail_length,
                                     f"Events | frame {t} | + red / - blue", scale,
                                     memory_diagnostic is not None)
                canvas = np.hstack((canvas, event_panel))
            if memory_diagnostic is not None:
                canvas = np.vstack((canvas, _memory_panel(canvas.shape[1], memory_diagnostic, t)))
            # H.264/VP8 yuv420p needs even image dimensions.
            height, width = canvas.shape[:2]
            if height % 2 or width % 2:
                canvas = cv2.copyMakeBorder(canvas, 0, height % 2, 0, width % 2,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if process is None:
                height, width = canvas.shape[:2]
                process = subprocess.Popen(ffmpeg_command(output_path, width, height, fps),
                                           stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            if t == frame_count // 2 and not cv2.imwrite(str(preview_path), canvas):
                raise RuntimeError(f"Could not save preview: {preview_path}")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
        if process is None:
            raise ValueError("Sequence has no frames")
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.strip()}")
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if process is not None:
            process.stderr.close()
    return output_path, preview_path
