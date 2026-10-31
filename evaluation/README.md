# ⚖️ Evaluation

In addition to the **Track-On** models, this module allows evaluating the **verifier** and other point tracking models using a unified evaluation pipeline.

## Parameters

### Required

- `--dataset_name`: Evaluation dataset. One of:  
  `davis`, `kinetics`, `robotap`, `point_odyssey`, `dynamic_replica`, `ego_points`

- `--dataset_path`: Path to the selected evaluation dataset.

- `--model_names`: One or more model names to evaluate. Supported values:  
  `trackon2`, `tapir`, `bootstapir`, `tapnext`, `bootstapnext`, `cotracker3_video`, `cotracker3_window`, `locotrack`, `anthro_locotrack`, `alltracker`, `verifier`, `oracle`, `random`

  Multiple models can be listed simultaneously. When `verifier` is included, it re-ranks the predictions of the other listed models.  
  `oracle` computes the ensemble upper bound by selecting the best candidate per point (no checkpoint needed).  
  `random` selects a candidate model uniformly at random as a lower-bound baseline (no checkpoint needed).  
  `cotracker3_video` and `cotracker3_window` download their checkpoints automatically on first use.  
  All other models require their official checkpoint to be downloaded. See [`ensemble/README.md`](../ensemble/README.md) for instructions.

### Track-On

- `--trackon_config_path`: Path to the Track-On config file (e.g., `config/test.yaml`).  
  Required if `trackon2` is included in `--model_names` and you want custom model parameters. If omitted, built-in defaults are used.

- `--trackon_checkpoint_path`: Path to the Track-On checkpoint.  
  Required if `trackon2` is included in `--model_names`.

### Verifier

- `--verifier_config_path`: Path to the verifier config file.  
  If omitted, built-in defaults are used.

- `--verifier_checkpoint_path`: Path to the verifier checkpoint.  
  Required if `verifier` is included in `--model_names`.

### Other model checkpoints

Required only for the corresponding model when listed in `--model_names`.

- `--tapir_checkpoint_path`: Path to the TAPIR checkpoint.
- `--bootstapir_checkpoint_path`: Path to the BootsTAPIR checkpoint.
- `--tapnext_checkpoint_path`: Path to the TAPNext checkpoint.
- `--bootstapnext_checkpoint_path`: Path to the BootsTAPNext checkpoint.
- `--locotrack_checkpoint_path`: Path to the LocoTrack checkpoint.
- `--anthro_locotrack_checkpoint_path`: Path to the Anthro-LocoTrack checkpoint.
- `--alltracker_checkpoint_path`: Path to the AllTracker checkpoint.

### Prediction caching

- `--cache_predictions`: Flag. When set, model predictions are cached to disk.  
  Highly recommended when evaluating the **verifier**, so teacher predictions are not recomputed on every run.

- `--cache_dir`: Directory where cached predictions are stored (default: `./cached_predictions`).  
  Only used when `--cache_predictions` is set.

---

## Example Usage

### Example 1

Evaluate **Track-On2** and **BootsTAPNext** on **DAVIS**.

```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
--dataset_name "davis" \
--dataset_path /path/to/davis \
--model_names "trackon2" "bootstapnext" \
--trackon_config_path config/test.yaml \
--trackon_checkpoint_path /path/to/trackon/ckpt \
--bootstapnext_checkpoint_path /path/to/bootstapnext/ckpt
```

### Example 2

Evaluate **Track-On2**, **Anthro-LocoTrack**, **AllTracker**, **CoTracker3 (window)**, and **Verifier** on Kinetics.
The verifier re-ranks the predictions of the preceding models as candidate trajectories.

```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
--dataset_name "kinetics" \
--dataset_path /path/to/kinetics \
--model_names "trackon2" "anthro_locotrack" "alltracker" "cotracker3_window" "verifier" \
--trackon_config_path config/test.yaml \
--trackon_checkpoint_path /path/to/trackon/ckpt \
--anthro_locotrack_checkpoint_path /path/to/anthrolocotrack/ckpt \
--alltracker_checkpoint_path /path/to/alltracker/ckpt \
--verifier_checkpoint_path /path/to/verifier/ckpt
```


## TAPFormer real datasets with the original RGB Track-On2

The separate `test_tapformer_real` entry point evaluates Track-On2 on **InivTAP** and
**DrivTAP** from TAPFormer's `TAPFormer_Dataset`. Configure dataset location,
checkpoint, model config, sequence list, support grid, and output path in
[`config/test_tapformer_real.yaml`](../config/test_tapformer_real.yaml). Relative
paths in the YAML are resolved from the Track-On2 repository root.

```bash
conda activate track_on_r
python -m evaluation.test_tapformer_real --config config/test_tapformer_real.yaml
# Quick check of one configured sequence:
python -m evaluation.test_tapformer_real --config config/test_tapformer_real.yaml --sequence ID_Fast_Fruit
```

The evaluator streams the original RGB images through Track-On2 and computes the
repository's TAP-Vid AJ, delta average, and OA metrics. InivTAP annotations
match the image frames. For DrivTAP sequences with twice as many GT samples as
images, `drivtap_gt_mode: downsample_even` selects GT indices `0, 2, ...`, as
in TAPFormer's configuration. Each dataset's result is the unweighted mean of
its sequence metrics. Results are written to `output_dir/summary.json`; set
`save_predictions: true` to also save per-sequence NPZ predictions.

`support_grid_size: S` adds an `S × S` global grid at the first frame (zero
disables it). Optional `local_support_radius: R` adds the eight neighbors of a
3 × 3 grid around each original query, with offsets `-R, 0, R` in model-input
pixels, scaled to the original image. Out-of-bounds neighbors are dropped;
the default radius is zero. All support points are tracked throughout the
sequence, while only original queries are scored. JSON sequence rows also
record the sampling settings, actual support count, and peak allocated CUDA
memory in MiB.

This RGB evaluation follows the image frame rate. TAPFormer's event-based run
tracks intermediate event timestamps too, so scores from the two protocols are
not directly comparable.

### Optional recent and RGB memory for frame/event evaluation

The recurrent, motion/ETAP, and RGB-aligned test configs expose an optional
inference-only policy. Missing `memory_policy` or `type: fifo` uses the original
Predictor. To enable the previously tested 24+48 split:

```yaml
inference_memory_size: 72
memory_policy:
  type: recent_rgb
  recent_size: 24
  rgb_size: 48
```

Every update enters the recent queue. Once full, its evicted entry enters the
older RGB queue only if it was generated at a new RGB timestamp. These are
fused queries from RGB times, not separately extracted RGB features. Both
queues are read together, without duplicates; unfilled slots remain masked.
All queues reset between sequences, and point features, validity masks, and
optional trajectory history stay synchronized. Training and the base Predictor
are unchanged.

`recent_size` must be a positive integer; `rgb_size` may be zero. Their sum must
equal `inference_memory_size`. For example, use 36+36 or 48+24 to compare splits
at the same total attention width. Changing the total also changes the existing
time-position embedding interpolation. The policy does not change `event_dt`,
support points, or RGB-time scoring. In RGB-only evaluation, every step is RGB.

Run the same `evaluation.test_tapformer_real` command with your edited config.
Use distinct output directories for different policies to preserve results.
JSON sequence rows record `memory_policy` and `inference_memory_size`; CSV rows
also include `memory_policy`, `recent_memory_size`, `rgb_memory_size`, and
`inference_memory_size` for both sequences and dataset means.

### EDS and EC feature tracking

The EDS/EC port uses the original RGB images and TAPFormer's sparse GT files.
Configure it in [`config/test_tapformer_features.yaml`](../config/test_tapformer_features.yaml),
then run:

```bash
python -m evaluation.test_tapformer_features --config config/test_tapformer_features.yaml
```

It reports TAPFormer's mean error, feature age (FA), and expected feature age
(EFA), each averaged over the configured 1–31 pixel thresholds. Predictions are
interpolated from RGB frame timestamps to GT timestamps for the metric.
TAPFormer's event model can update between RGB frames, so its EDS/EC scores and
these RGB scores use different inference timelines.

Both TAPFormer entry points also write `summary.csv` next to `summary.json`.
The CSV has one row per sequence and one `dataset_mean` row per dataset; it
includes model/checkpoint identifiers and frame timing. InivTAP/DrivTAP metric
columns end in `_pct` and contain percentage values (for example, `46.26`),
while EDS/EC columns retain FA/EFA fractions and pixel error. Set
`csv_filename` in the corresponding YAML to rename the CSV.

### Video previews and VS Code playback

Set `visualization.enabled: true` in the relevant YAML to save one video and
middle-frame PNG per sequence under `output_dir/videos/`. InivTAP/DrivTAP videos
show GT and Track-On2 predictions side by side; EDS/EC videos show predictions
on the RGB frames. `visualization.format: mp4` writes H.264/yuv420p with
`faststart`; set it to `webm` for VP8. Adjust `fps`, `trail_length`, and
`max_width` in the same config. The JSON and CSV store the paths to each video
and PNG preview. Video export requires the system `ffmpeg` executable with
`libx264` (MP4) or `libvpx` (WebM).

Older visualization files in this repository were written with OpenCV's `mp4v`
codec. VS Code's built-in media preview uses a browser video element, which may
show "unsupported format" for those MP4 files even though `ffprobe` can read
them. Convert one without rerunning inference:

```bash
ffmpeg -i old.mp4 -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart old_h264.mp4
```

If the client still cannot play H.264, use the VP8 WebM option or inspect the
PNG preview.
