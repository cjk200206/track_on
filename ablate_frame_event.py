"""Fixed-checkpoint frame/event ablations using the existing matching diagnostics.

CUDA_VISIBLE_DEVICES=4 python ablate_frame_event.py
Use --modes to split independent groups across GPUs. No model parameters change.
"""

import argparse
import hashlib
import json
from argparse import Namespace
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from dataset.tapformer_real import load_real_tap_event_stream, load_real_tap_sequence
from diagnose_tapformer_matching import (
    array, inspect_sequence, prepare_step, resolve, stream_steps, verify_observer,
    write_csv, write_json,
)
from model.event_motion import resize_warp, warp_features
from model.trackon_predictor import Predictor
from visualize_frame_event_features import capture_features


MODES = ("baseline", "rgb_anchor", "rgb_memory", "warp_only")
DESCRIPTIONS = {
    "baseline": "Original frame/event inference.",
    "rgb_anchor": "After initialization, new RGB uses unwarped anchor content; motion state still updates.",
    "rgb_memory": "Every step predicts; only new RGB commits point memory and its validity mask.",
    "warp_only": "Warp RGB content at every step; omit the additive event residual.",
}


@contextmanager
def ablation(predictor, mode, audit=False):
    """Apply one streaming-only intervention and restore methods on exit."""
    fusion = predictor.model.backbone.event_fusion
    counts = dict(rgb_bypasses=0, skipped_memory_writes=0, residual_bypasses=0)
    with ExitStack() as stack:
        if mode == "rgb_anchor":
            original = fusion.step

            def rgb_step(rgb_features, event_features, img_ifnew, state):
                features = original(rgb_features, event_features, img_ifnew, state)
                if predictor.t > 0 and bool(img_ifnew):
                    counts["rgb_bypasses"] += 1
                    return state.anchor
                return features

            stack.enter_context(patch.object(fusion, "step", new=rgb_step))
        elif mode == "rgb_memory":
            original = predictor.forward_frame

            def memory_step(frame, new_queries=None, events=None, img_ifnew=None):
                saved = {}
                if not bool(img_ifnew):
                    if new_queries is not None:
                        raise ValueError("This ablation initializes queries only at the first RGB")
                    for name in ("point_memory", "temporal_mask", "trajectory_history"):
                        value = getattr(predictor, name)
                        if value is not None:
                            saved[name] = value[:predictor.N].clone()
                result = original(frame, new_queries=new_queries, events=events, img_ifnew=img_ifnew)
                for name, value in saved.items():
                    getattr(predictor, name)[:predictor.N].copy_(value)
                    if audit and not torch.equal(getattr(predictor, name)[:predictor.N], value):
                        raise AssertionError(f"Intermediate step changed {name}")
                counts["skipped_memory_writes"] += bool(saved)
                return result

            stack.enter_context(patch.object(predictor, "forward_frame", new=memory_step))
        elif mode == "warp_only":
            def warp_fuse(anchor, event4, motion, warp, activity):
                counts["residual_bypasses"] += 1
                return tuple(warp_features(frame, resize_warp(warp, frame.shape[-2:])) for frame in anchor)

            stack.enter_context(patch.object(fusion, "_fuse", new=warp_fuse))
        elif mode != "baseline":
            raise ValueError(f"Unknown ablation: {mode}")
        yield counts


@torch.no_grad()
def verify_interventions(predictor, sequence, config, modes, device):
    """Replay seven actual steps, including RGB boundaries, before long runs."""
    reference, results = [], {}
    for mode in dict.fromkeys(("baseline", *modes)):
        predictor.reset()
        frame = previous_image = None
        new_rgb_steps = 0
        with ablation(predictor, mode, audit=True) as counts, capture_features(predictor.model) as captured:
            for step, (image, events, _) in enumerate(stream_steps(sequence, config, True)):
                if step == 7:
                    break
                new_rgb = image is not previous_image
                new_rgb_steps += new_rgb
                frame, kwargs = prepare_step(predictor, image, events, new_rgb, frame, sequence,
                                             config.get("support_grid_size", 20), step == 0, device)
                previous_image = image
                if step == 0 and np.any(events):
                    raise AssertionError("Expected the aligned cache to initialize with zero events")
                before = (None if predictor.point_memory is None else
                          predictor.point_memory[:predictor.N].clone())
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=config.get("amp", True) and device.type == "cuda"):
                    predictor.forward_frame(frame, **kwargs)
                state = predictor.frontend_state
                motion = (array(state.warp), array(state.history[-1].feature))
                if mode == "baseline":
                    reference.append(motion)
                    if step == 0:
                        initial_query = array(predictor.q_init[:predictor.N])
                else:
                    if not all(np.array_equal(a, b) for a, b in zip(motion, reference[step])):
                        raise AssertionError(f"{mode} changed the frontend motion state at step {step}")
                    if step == 0 and not np.array_equal(initial_query, array(predictor.q_init[:predictor.N])):
                        raise AssertionError(f"{mode} changed query initialization")
                if mode == "rgb_anchor" and new_rgb and step > 0:
                    if not all(torch.equal(a, b) for a, b in zip(captured["content"], state.anchor)):
                        raise AssertionError("RGB bypass differs from the original anchor content")
                if mode == "warp_only":
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=config.get("amp", True) and device.type == "cuda"):
                        expected = tuple(warp_features(a, resize_warp(state.warp, a.shape[-2:])) for a in state.anchor)
                    if not all(torch.equal(a, b) for a, b in zip(captured["content"], expected)):
                        errors = [(a.float() - b.float()).abs().max().item() for a, b in zip(captured["content"], expected)]
                        raise AssertionError(f"Warp-only content mismatch at step {step}: {errors}")
                if mode == "rgb_memory" and before is not None:
                    expected = before if not new_rgb else torch.cat((before[:, 1:], predictor.point_memory[:predictor.N, -1:]), 1)
                    if not torch.equal(predictor.point_memory[:predictor.N], expected):
                        raise AssertionError("Point memory has an unexpected FIFO update")
            steps = step
        expected_counts = dict(rgb_bypasses=(new_rgb_steps - 1 if mode == "rgb_anchor" else 0),
                               skipped_memory_writes=(steps - new_rgb_steps if mode == "rgb_memory" else 0),
                               residual_bypasses=(steps if mode == "warp_only" else 0))
        if counts != expected_counts or new_rgb_steps < 2:
            raise AssertionError(f"Unexpected intervention counts: {counts}, expected {expected_counts}")
        results[mode] = dict(steps=steps, **counts, motion_state_unchanged=True, initial_query_unchanged=True)
    predictor.reset()
    return results


def write_comparison(output):
    """Sequence means for TAP-Vid; visible-point-weighted matching diagnostics."""
    comparison = []
    for mode in MODES:
        path = output / mode / "summary.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text())
        for dataset in dict.fromkeys(row["dataset"] for row in rows):
            selected = [row for row in rows if row["dataset"] == dataset]
            count = sum(row["gt_point_steps"] for row in selected)
            row = dict(mode=mode, dataset=dataset, sequences=len(selected), gt_point_steps=count)
            for key in ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy"):
                row[key] = float(np.mean([s[key] for s in selected]))
            for key in ("recall16_r8", "c1_correct", "c2_correct", "error_input_px"):
                row[key] = sum(s[key] * s["gt_point_steps"] for s in selected) / count
            for key in ("jump_onsets", "candidate_miss_count", "candidate_present_c2_wrong_count", "c1_correct_c2_wrong_count"):
                row[key] = sum(s[key] for s in selected)
            comparison.append(row)
    write_csv(output / "comparison.csv", comparison)
    write_json(output / "comparison.json", comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/test_tapformer_real_frame_event_rgb_aligned.yaml")
    parser.add_argument("--output-dir", default="outputs/frame_event_ablations")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--sequence", nargs="+")
    parser.add_argument("--verify-only", action="store_true")
    cli = parser.parse_args()
    config = yaml.safe_load(resolve(cli.config).read_text())
    checkpoint = resolve(config["checkpoint"])
    checkpoint_stat = checkpoint.stat()
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = json.loads(resolve(config["model_config"]).read_text())
    args = Namespace(**saved["args"])
    args.M_i = int(config.get("inference_memory_size", 72))
    for name in ("vit_pretrained_path", "event_pretrained_path"):
        if getattr(args, name, None):
            setattr(args, name, str(resolve(getattr(args, name))))
    if args.event_fusion_type != "motion_state" or getattr(args, "trajectory_history_size", 0):
        parser.error("Use the original motion_state checkpoint without a trajectory scorer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(args, str(checkpoint), support_grid_size=0).to(device).eval()
    output = resolve(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = [(dataset, name) for dataset, names in config["sequences"].items() for name in names
                if not cli.sequence or name in cli.sequence]
    if not selected or set(cli.sequence or []) - {name for _, name in selected}:
        parser.error("Unknown or empty sequence selection")
    # Inspect the two agreed pilot sequences first, then the remaining sequences.
    selected.sort(key=lambda item: (item[1] not in ("ID_Fast_Fruit", "Day_Fast"),))
    verification = {}
    for dataset, name in selected:
        if dataset in verification:
            continue
        sequence = load_real_tap_sequence(resolve(config["dataset_root"]), dataset, name,
                                          config.get("drivtap_gt_mode", "downsample_even"))
        verification[dataset] = dict(sequence=name,
                                    observer=verify_observer(predictor, sequence, config, True, device),
                                    interventions=verify_interventions(predictor, sequence, config, cli.modes, device))
    print(f"Intervention verification passed: {verification}", flush=True)
    if cli.verify_only:
        write_json(output / ("verification_" + "_".join(cli.modes) + ".json"), verification)
        return
    diagnostic_options = Namespace(model="frame_event", max_steps=None, no_details=True)
    summaries = {mode: [] for mode in cli.modes}
    groups = {mode: [] for mode in cli.modes}
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=checksum, model_args=vars(args),
                    evaluation_config=config, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    verification=verification, complete=False,
                    gt_policy="RGB times only; Driv even GT; matching excludes initialization, occlusion and out-of-bounds GT",
                    aggregation="TAP-Vid: sequence mean; matching: valid-point weighted; rates in [0,1]",
                    memory_caveat="rgb_memory also increases the physical time covered by the same 72 slots")
    for mode in cli.modes:
        directory = output / mode
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "run_config.json", dict(metadata, mode=mode, intervention=DESCRIPTIONS[mode]))
    for dataset, name in selected:
        sequence = load_real_tap_sequence(resolve(config["dataset_root"]), dataset, name,
                                          config.get("drivtap_gt_mode", "downsample_even"))
        stream = load_real_tap_event_stream(sequence, config["event_dt"][dataset])
        sequence.event_timestamps = np.array([int(path.stem) for path in stream.event_paths])
        for mode in cli.modes:
            destination = output / mode / dataset / name
            destination.mkdir(parents=True, exist_ok=True)
            with ablation(predictor, mode) as counts:
                summary, sequence_groups = inspect_sequence(predictor, sequence, config, diagnostic_options, destination, device)
            summary.update(mode=mode, **counts)
            summaries[mode].append(summary)
            groups[mode].extend(sequence_groups)
            directory = output / mode
            write_csv(directory / "summary.csv", summaries[mode])
            write_json(directory / "summary.json", summaries[mode])
            write_csv(directory / "groups.csv", groups[mode])
            print(f"{mode} {dataset}/{name}: AJ={summary['average_jaccard']:.4f}, "
                  f"delta={summary['average_pts_within_thresh']:.4f}, R16@8={summary['recall16_r8']:.4f}, "
                  f"jumps={summary['jump_onsets']}, interventions={counts}", flush=True)
    current = checkpoint.stat()
    if (current.st_size, current.st_mtime_ns) != (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns):
        raise RuntimeError("Checkpoint changed during the run")
    for mode in cli.modes:
        write_json(output / mode / "run_config.json", dict(metadata, mode=mode, intervention=DESCRIPTIONS[mode], complete=True))
    write_comparison(output)


if __name__ == "__main__":
    main()
