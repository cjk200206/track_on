"""Render RGB point tracks to browser-friendly H.264 MP4 or VP8 WebM."""

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


def _panel(rgb, tracks, visible, t, colors, trail_length, label, scale):
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
    return panel


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
                          max_width=960, gt_tracks=None, gt_visible=None):
    """Stream a labeled video and middle-frame PNG without holding frames in memory."""
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _colors(point_count)
    process = None
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    try:
        for t, rgb in enumerate(sequence.frames()):
            scale = min(1.0, max_width / rgb.shape[1])
            prediction = _panel(rgb, tracks, visible, t, colors, trail_length,
                                f"Track-On2 | frame {t}", scale)
            if gt_tracks is not None:
                ground_truth = _panel(rgb, gt_tracks, gt_visible, t, colors, trail_length,
                                      f"GT | frame {t}", scale)
                canvas = np.hstack((ground_truth, prediction))
            else:
                canvas = prediction
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
