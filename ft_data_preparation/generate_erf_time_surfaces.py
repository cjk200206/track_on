r"""Convert ERF events to TAPFormer time surfaces at every Nth native RGB frame.

Example (run from the project root):
    python ft_data_preparation/generate_erf_time_surfaces.py \
        --input-root data/erf_x170/train --output-root data/erf_x170/erf_time_surfaces/train \
        --stride 4

Each sequence needs processed_images/*.png and processed_events/*.npz.
By the agreed convention, packet i covers RGB i-1 -> i. Packet 0 is before
initialization and is excluded. A stride of 4 merges packets 1..4, then 5..8,
etc., with RGB endpoints 0, 4, 8, ... . RGB timestamps are ESTIMATED from the
midpoints between packet i's last event and packet i+1's first event. The final
RGB boundary uses its packet's last event because no following packet exists.
Event timestamps are interpreted as microseconds; these estimates are not
hardware exposure timestamps. Empty or overlapping packets cannot define
these boundaries and are rejected rather than filled using nominal 170 FPS.

Output: OUTPUT/stride_N/SEQUENCE.h5. The time_surface dataset is float32 THWC,
with five right-inclusive uniform time bins and interleaved polarities 0, 1.
Values are latest event times relative to the WHOLE window, divided by its
duration; empty pixels are zero. Coordinates are decoded by /128 then floored.
Spatial resolution is unchanged. Index 0 is an empty RGB initialization step;
subsequent windows are (previous RGB time, current RGB time]. The final partial
stride is omitted. Other datasets retain native frame indices, timestamps,
window bounds and event counts. This script neither copies nor subsamples RGB.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def estimate_rgb_timestamps(paths):
    """Return estimated RGB times and source packet bounds in the event clock."""
    bounds = []
    for path in paths:
        # Only decompress t during this pass; x/y/p are read when encoding.
        with np.load(path, allow_pickle=False) as packet:
            times = packet["t"]
        if (times.ndim != 1 or not len(times)
                or not np.issubdtype(times.dtype, np.signedinteger)
                or np.any(times[1:] < times[:-1])):
            raise ValueError(f"Timestamp estimation needs nonempty, sorted integer event times: {path}")
        bounds.append((int(times[0]), int(times[-1])))
    bounds = np.asarray(bounds, dtype=np.int64)
    gaps = bounds[1:, 0] - bounds[:-1, 1]
    if np.any(gaps <= 0):
        raise ValueError("Event packets overlap or share timestamps; cannot estimate RGB boundaries")
    timestamps = np.concatenate((bounds[:-1, 1] + gaps // 2, bounds[-1:, 1]))
    return timestamps, bounds


def load_events(path):
    with np.load(path, allow_pickle=False) as packet:
        arrays = tuple(packet[key] for key in ("t", "x", "y", "p"))
    times, x, y, polarity = arrays
    if any(array.ndim != 1 or array.shape != times.shape for array in arrays):
        raise ValueError(f"Expected matching one-dimensional t/x/y/p arrays: {path}")
    if any(not np.issubdtype(array.dtype, np.integer) for array in arrays):
        raise ValueError(f"Expected integer ERF t/x/y/p arrays: {path}")
    if np.any(times[1:] < times[:-1]):
        raise ValueError(f"Event timestamps are not sorted: {path}")
    return times, x, y, polarity


def encode_time_surface(parts, start, end, image_size):
    """Keep the latest timestamp per pixel/bin/polarity, normalized over (start, end]."""
    height, width = image_size
    surface = np.zeros((height, width, 10), dtype=np.int64)
    duration = int(end) - int(start)
    if duration <= 0:
        raise ValueError("Event windows must have positive duration")
    for times, raw_x, raw_y, polarity in parts:
        relative = times - start
        x, y = raw_x.astype(np.int64) // 128, raw_y.astype(np.int64) // 128
        if (np.any(relative <= 0) or np.any(relative > duration)
                or np.any(x < 0) or np.any(x >= width)
                or np.any(y < 0) or np.any(y >= height)
                or np.any((polarity != 0) & (polarity != 1))):
            raise ValueError("Events outside the time window, image bounds or polarity range")
        bins = (relative * 5 - 1) // duration
        indices = (y * width + x) * 10 + 2 * bins + polarity
        np.maximum.at(surface.reshape(-1), indices, relative)
    return surface.astype(np.float32) / np.float32(duration)


def convert_sequence(sequence, output, stride):
    images = sorted((sequence / "processed_images").glob("*.png"))
    events = sorted((sequence / "processed_events").glob("*.npz"))
    if len(images) < stride + 1 or len(events) != len(images):
        raise ValueError(f"Expected matching RGB/event files and at least {stride + 1} RGB frames: {sequence}")
    expected = list(range(len(images)))
    if ([int(path.stem) for path in images] != expected
            or [int(path.stem) for path in events] != expected):
        raise ValueError(f"RGB/event filenames must be consecutive indices starting at zero: {sequence}")
    if output.exists():
        raise FileExistsError(f"Output already exists; select unconverted sequences or a new output root: {output}")
    timestamps, packet_bounds = estimate_rgb_timestamps(events)
    indices = np.arange(0, len(images), stride, dtype=np.int64)
    sampled_times = timestamps[indices]
    with Image.open(images[0]) as image:
        width, height = image.size
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".h5.tmp")
    handle = h5py.File(temporary, "x")
    try:
        with handle:
            surfaces = handle.create_dataset(
                "time_surface", shape=(len(indices), height, width, 10), dtype="float32",
                chunks=(1, min(64, height), min(64, width), 10), compression="lzf", shuffle=True,
            )
            handle.create_dataset("frame_indices", data=indices)
            handle.create_dataset("rgb_timestamps_us", data=timestamps)
            handle.create_dataset("source_packet_bounds_us", data=packet_bounds)
            handle.create_dataset("timestamps_us", data=sampled_times)
            starts = np.concatenate((sampled_times[:1], sampled_times[:-1]))
            handle.create_dataset("windows_us", data=np.column_stack((starts, sampled_times)))
            counts = handle.create_dataset("event_counts", shape=(len(indices),), dtype="int64")
            handle.attrs.update(
                source=str(sequence.resolve()), rgb_timestamps_estimated=True,
                timestamp_source="midpoints of adjacent event packet bounds",
                final_rgb_timestamp_source="last event in final packet",
                packet_alignment="packet i covers RGB i-1 -> i; packet 0 excluded",
                native_rgb_stride=stride, image_size=[height, width], coordinate_scale=128,
                n_bins=5, channel_order="bin0_p0,bin0_p1,...,bin4_p0,bin4_p1",
                interval="(start_us,end_us]", initial_step="zero events",
                normalization="(latest_event_us-window_start_us)/window_duration_us",
                trailing_rgb_intervals=len(images) - 1 - int(indices[-1]),
            )
            # HDF5 initializes the first surface and event count to zero.
            for index in range(1, len(indices)):
                left, right = int(indices[index - 1]) + 1, int(indices[index]) + 1
                parts = [load_events(path) for path in events[left:right]]
                surfaces[index] = encode_time_surface(
                    parts, sampled_times[index - 1], sampled_times[index], (height, width),
                )
                counts[index] = sum(len(part[0]) for part in parts)
                if index % 25 == 0 or index == len(indices) - 1:
                    print(f"{sequence.name}: {index}/{len(indices) - 1} windows", flush=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output} ({len(indices)} steps including RGB initialization)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=Path("data/erf_x170/train"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4, help="Native RGB interval count per event update (default: 4)")
    parser.add_argument("--sequences", nargs="+", help="Convert only these sequence names, e.g. 0000 0001")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    sequences = ([args.input_root / name for name in args.sequences] if args.sequences else
                 sorted(path for path in args.input_root.iterdir() if path.is_dir()))
    if not sequences:
        parser.error("No sequences found")
    for sequence in sequences:
        if not sequence.is_dir():
            parser.error(f"Sequence not found: {sequence}")
    for sequence in sequences:
        convert_sequence(
            sequence, args.output_root / f"stride_{args.stride}" / f"{sequence.name}.h5",
            args.stride,
        )


if __name__ == "__main__":
    main()
