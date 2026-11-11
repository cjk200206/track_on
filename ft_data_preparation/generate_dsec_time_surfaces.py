r"""Generate native-RGB-aligned DSEC time surfaces (~20 Hz).

Example:
    /root/anaconda3/envs/track_on_r/bin/python -u -B \
        /remote-home/share/cjk/track_on/ft_data_preparation/generate_dsec_time_surfaces.py \
        --input-root /remote-home/share/cjk/track_on/data/DSEC \
        --output-root /remote-home/share/cjk/datasets/DSEC/dsec_time_surfaces/train \
        --sequences interlaken_00_c

Use integer left-exposure midpoints from images/exposure_timestamps_left.txt,
exactly as the DSEC pseudo-labels do. This table defines frame indices; RGB
files are not needed during conversion. Coordinates refer to the left event
camera's distorted 640x480 view (images/left/distorted for remapped RGB).
Read events/left/events.h5 using ms_to_idx; add t_offset to event times. Raw
sensor coordinates need neither /128 scaling nor rectification.

Output: OUTPUT/stride_1/SEQUENCE.h5. time_surface is float32 THWC with 5 uniform,
right-inclusive time bins and interleaved polarities [bin0_p0, bin0_p1, ...].
Each pixel/bin/polarity stores the latest event time relative to the WHOLE
window, divided by its duration; empty locations are zero. Windows cover
(previous exposure midpoint, current exposure midpoint], typically ~50 ms.
The first step is zero-event RGB initialization. RGB thinning (e.g. 10 Hz for
F:E=1:2) belongs to the training loader and does not change these caches.

If the event stream starts inside the first window, the unavailable prefix is
recorded as initial_uncovered_us; exposure timestamps remain unchanged.
Requires numpy, h5py and hdf5plugin (for DSEC's Blosc/ZSTD input).
Existing outputs or temporary files are never overwritten.
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # Register the Blosc/ZSTD filters used by DSEC input files.
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_sequence(sequence):
    """Exposure rows define native frame indices, independently of RGB recovery."""
    path = sequence / "images/exposure_timestamps_left.txt"
    exposures = np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64, ndmin=2)
    if exposures.shape[1] != 2 or len(exposures) < 2 or np.any(exposures[:, 1] <= exposures[:, 0]):
        raise ValueError(f"Expected one positive exposure interval per RGB frame: {path}")
    times = exposures[:, 0] + (exposures[:, 1] - exposures[:, 0]) // 2
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Left-exposure midpoints must be strictly increasing: {path}")
    return times, exposures


def read_events(events, ms_to_idx, offset, start, end):
    """Read only the needed HDF5 slices; exact integer bounds are (start, end]."""
    left_time, right_time = int(start) - offset, int(end) - offset
    start_ms, end_ms = max(0, left_time // 1000), right_time // 1000 + 1
    left = int(ms_to_idx[start_ms])
    right = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else len(events["t"])
    times = events["t"][left:right].astype(np.int64)
    first, last = np.searchsorted(times, (left_time, right_time), side="right")
    begin, finish = left + int(first), left + int(last)
    return (times[first:last] + offset,
            *(events[key][begin:finish].astype(np.int64) for key in ("x", "y", "p")))


def encode_time_surface(events, start, end):
    """Encode latest relative timestamps in the existing 5-bin, 2-polarity format."""
    times, x, y, polarity = events
    duration = int(end) - int(start)
    relative = times - start
    if (duration <= 0 or np.any(relative <= 0) or np.any(relative > duration)
            or np.any(x < 0) or np.any(x >= 640) or np.any(y < 0) or np.any(y >= 480)
            or np.any((polarity != 0) & (polarity != 1))):
        raise ValueError("Events outside the window, 640x480 image or polarity range")
    surface = np.zeros((480, 640, 10), dtype=np.int64)
    bins = (relative * 5 - 1) // duration
    indices = (y * 640 + x) * 10 + 2 * bins + polarity
    np.maximum.at(surface.reshape(-1), indices, relative)
    return surface.astype(np.float32) / np.float32(duration)


def convert_sequence(sequence, output):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    times, exposures = read_sequence(sequence)
    source = sequence / "events/left/events.h5"
    with h5py.File(source, "r") as raw:
        events = raw["events"]
        count = len(events["t"])
        if not count or any(events[key].shape != (count,) for key in ("t", "x", "y", "p")):
            raise ValueError(f"Expected matching nonempty one-dimensional t/x/y/p datasets: {source}")
        offset = int(raw["t_offset"][()])
        ms_to_idx = raw["ms_to_idx"][:].astype(np.int64)
        if (times[1] <= offset or times[-1] > int(events["t"][-1]) + offset
                or (int(times[-2]) - offset) // 1000 >= len(ms_to_idx)):
            raise ValueError(f"Event stream/index does not cover the RGB time range: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".h5.tmp")
        handle = h5py.File(temporary, "x")
        try:
            with handle:
                surfaces = handle.create_dataset(
                    "time_surface", shape=(len(times), 480, 640, 10), dtype="float32",
                    chunks=(1, 64, 64, 10), compression="lzf", shuffle=True,
                )
                handle.create_dataset("frame_indices", data=np.arange(len(times), dtype=np.int64))
                handle.create_dataset("rgb_timestamps_us", data=times)
                handle.create_dataset("timestamps_us", data=times)
                handle.create_dataset("exposure_timestamps_us", data=exposures)
                starts = np.concatenate((times[:1], times[:-1]))
                handle.create_dataset("windows_us", data=np.column_stack((starts, times)))
                counts = handle.create_dataset("event_counts", shape=(len(times),), dtype="int64")
                handle.attrs.update(
                    source=str(sequence.resolve()), events_source=str(source.resolve()),
                    rgb_directory="images/left/distorted", rgb_timestamps_estimated=False,
                    timestamp_source="images/exposure_timestamps_left.txt; floor of left-exposure midpoint (us)",
                    t_offset_us=offset, native_rgb_stride=1, image_size=[480, 640], coordinate_scale=1,
                    initial_uncovered_us=max(0, offset - int(times[0])),
                    coordinates="distorted left event-camera pixels (x,y)",
                    n_bins=5, channel_order="bin0_p0,bin0_p1,...,bin4_p0,bin4_p1",
                    interval="(start_us,end_us]", initial_step="zero events",
                    normalization="(latest_event_us-window_start_us)/window_duration_us",
                )
                # HDF5 initializes the first surface and event count to zero.
                for index in range(1, len(times)):
                    window = read_events(events, ms_to_idx, offset, times[index - 1], times[index])
                    surfaces[index] = encode_time_surface(window, times[index - 1], times[index])
                    counts[index] = len(window[0])
                    if index % 25 == 0 or index == len(times) - 1:
                        print(f"{sequence.name}: {index}/{len(times) - 1} windows", flush=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Saved {output} ({len(times)} steps including RGB initialization)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/DSEC")
    parser.add_argument("--output-root", type=Path, help="Parent of stride_1; default: beside resolved input split")
    parser.add_argument("--sequences", nargs="+", help="Convert only these sequence names")
    args = parser.parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    output_root = (args.output_root.expanduser().resolve() if args.output_root else
                   args.input_root.parent / "dsec_time_surfaces" / args.input_root.name)
    sequences = ([args.input_root / name for name in args.sequences] if args.sequences else
                 sorted(path for path in args.input_root.iterdir() if path.is_dir()))
    if not sequences:
        parser.error("No sequences found")
    for sequence in sorted(set(sequences)):
        convert_sequence(sequence, output_root / "stride_1" / f"{sequence.name}.h5")


if __name__ == "__main__":
    main()
