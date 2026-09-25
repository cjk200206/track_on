"""Feasibility experiment for paired Kubric frames and time surfaces.

Run --check-data first, then --overfit-one-batch, then the default training run.
Use --gpus 0 1 2 3 to select GPU IDs, or launch with torchrun.
Use --eval-checkpoint PATH to repeat the three validation modes.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.frame_event_kubric import FrameEventKubric
from model.loss import Loss_Function
from model.trackon import Track_On2
from utils.coord_utils import get_queries
from utils.eval_utils import compute_tapvid_metrics
from utils.train_utils import load_args_from_yaml, load_pretrained_weights


def create_run_directory(base_dir):
    """Create one timestamped output directory shared by all training ranks."""
    run_dir = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        for suffix in range(100):
            candidate = base / (timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}")
            try:
                candidate.mkdir()
                run_dir = str(candidate)
                break
            except FileExistsError:
                continue
        if run_dir is None:
            raise RuntimeError(f"Could not create a unique run directory under {base}")
    if dist.is_initialized():
        paths = [run_dir]
        dist.broadcast_object_list(paths, src=0)
        run_dir = paths[0]
    return Path(run_dir)


def first_visible_queries(tracks, visibility):
    """Return (t, x, y) queries for tracks visible at least once."""
    batch, _, points, _ = tracks.shape
    first = visibility.long().argmax(dim=1)
    batch_index = torch.arange(batch, device=tracks.device)[:, None]
    point_index = torch.arange(points, device=tracks.device)[None, :]
    xy = tracks[batch_index, first, point_index]
    return torch.cat((first.unsqueeze(-1), xy), dim=-1)


def make_loader(args, training, rank=0, world_size=1):
    dataset = FrameEventKubric(
        root=args.train_root if training else args.val_root,
        clip_length=args.T,
        num_points=args.N,
        crop_size=args.input_size,
        frame_interval=args.frame_interval,
        max_sequences=args.train_sequences if training else args.val_sequences,
        training=training,
        seed=args.seed,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if training and world_size > 1 else None
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def inspect_data(args):
    sample = make_loader(args, training=False).dataset[0]
    for name, value in sample.items():
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"events range: {sample['events'].min().item():.4f} to {sample['events'].max().item():.4f}")
    print(f"new RGB steps: {sample['img_ifnew'].nonzero().flatten().tolist()}")
    print(f"annotation indices: {sample['frame_indices'].tolist()}")

    rgb = Image.fromarray(sample["video"][0].byte().permute(1, 2, 0).numpy())
    event_image = (sample["events"][0].amax(dim=0).clamp(0, 1) * 255).byte().numpy()
    event = Image.fromarray(event_image).convert("RGB")
    overlay = Image.new("RGB", (rgb.width * 2, rgb.height))
    overlay.paste(rgb, (0, 0))
    overlay.paste(event, (rgb.width, 0))
    draw = ImageDraw.Draw(overlay)
    for xy in sample["tracks"][0, sample["visibility"][0]][:32]:
        x, y = xy.tolist()
        for offset in (0, rgb.width):
            draw.ellipse((x + offset - 3, y - 3, x + offset + 3, y + 3), outline="red", width=2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sample_alignment.png"
    overlay.save(path)
    print(f"alignment preview: {path}")


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_model(args, device):
    model = Track_On2(args).to(device)
    load_pretrained_weights(
        model,
        args.checkpoint_path,
        extra_allowed_missing_prefixes=("backbone.event_fusion.",),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("backbone.event_fusion."))
    model.eval()  # Keep the pretrained BatchNorm statistics fixed during event training.
    if int(os.environ.get("RANK", "0")) == 0:
        dino_path = getattr(args, "vit_pretrained_path", None)
        if dino_path is not None:
            print(f"DINOv3 initialized from {dino_path} and frozen", flush=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable event-fusion parameters: {trainable}", flush=True)
    return model


def load_event_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    # Accept checkpoints saved before event fusion moved into its own module.
    state = {
        key.replace("backbone.event_encoder.layers.", "backbone.event_fusion.encoder.")
           .replace("backbone.event_gates", "backbone.event_fusion.gates"): value
        for key, value in state.items()
    }
    model.load_state_dict(state)
    return checkpoint.get("step")


@torch.no_grad()
def evaluate(model, loader, args, device, event_mode):
    metrics = []
    position_errors = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        tracks = batch["tracks"]
        visibility = batch["visibility"]
        queries = first_visible_queries(tracks, visibility)
        if event_mode == "rgb":
            events = None
        elif event_mode == "zero":
            events = torch.zeros_like(batch["events"])
        else:
            events = batch["events"]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            output = model(batch["video"], queries.clone(), p_mask=0, events=events)
        predicted = output["P"]
        predicted_visible = output["V_logit"].sigmoid() >= args.delta_v
        metric = compute_tapvid_metrics(
            queries[..., [0, 2, 1]].cpu().numpy(),
            (~visibility).permute(0, 2, 1).cpu().numpy(),
            tracks.permute(0, 2, 1, 3).cpu().numpy(),
            (~predicted_visible).permute(0, 2, 1).cpu().numpy(),
            predicted.permute(0, 2, 1, 3).cpu().numpy(),
            "first",
        )
        metrics.append(metric)
        after_query = torch.arange(tracks.shape[1], device=device)[None, :, None] > queries[:, None, :, 0]
        scored = visibility & after_query
        position_errors.extend(torch.linalg.vector_norm(predicted - tracks, dim=-1)[scored].cpu().tolist())

    result = {
        "AJ": float(np.nanmean([m["average_jaccard"][0] for m in metrics])),
        "OA": float(np.nanmean([m["occlusion_accuracy"][0] for m in metrics])),
        "delta_avg": float(np.nanmean([m["average_pts_within_thresh"][0] for m in metrics])),
        "mean_error_px": float(np.mean(position_errors)),
    }
    print(f"{event_mode}: " + ", ".join(f"{key}={value:.4f}" for key, value in result.items()))
    return result


def train(model, loader, args, device, steps, overfit_one_batch, rank=0):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.wd,
    )
    loss_fn = Loss_Function(args)
    iterator = iter(loader)
    epoch = 0
    fixed_batch = move_batch(next(iterator), device) if overfit_one_batch else None
    fixed_queries = (
        first_visible_queries(fixed_batch["tracks"], fixed_batch["visibility"])
        if fixed_batch is not None else None
    )
    loss_names = ("patch1", "patch2", "visibility", "offset", "uncertainty", "topk_uncertainty", "ranking")
    window_sums = torch.zeros(len(loss_names) + 1, device=device)
    running_sums = torch.zeros_like(window_sums)
    window_steps = 0
    running_steps = 0

    for step in range(1, steps + 1):
        if fixed_batch is None:
            try:
                batch = move_batch(next(iterator), device)
            except StopIteration:
                epoch += 1
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = move_batch(next(iterator), device)
            queries = get_queries(batch["tracks"], batch["visibility"])
        else:
            batch = fixed_batch
            queries = fixed_queries
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            output = model(batch["video"], queries.clone(), events=batch["events"], p_mask=0 if overfit_one_batch else 0.1)
            losses = loss_fn(output, batch["tracks"], batch["visibility"], queries[..., 0].long())
            loss = sum(losses)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite event gradient at step {step}")
        optimizer.step()
        window_sums += torch.stack((loss.detach().float(), *(part.detach().float() for part in losses)))
        window_steps += 1
        if step % args.log_every == 0 or step == steps:
            if dist.is_initialized():
                dist.all_reduce(window_sums, op=dist.ReduceOp.SUM)
            running_sums += window_sums
            running_steps += window_steps
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            recent_means = (window_sums / (window_steps * world_size)).tolist()
            running_means = (running_sums / (running_steps * world_size)).tolist()
            if rank == 0:
                recent_components = dict(zip(loss_names, recent_means[1:]))
                running_components = dict(zip(loss_names, running_means[1:]))
                parts = ", ".join(f"{name}={value:.4f}" for name, value in recent_components.items())
                record = {
                    "step": step,
                    "window_steps": window_steps,
                    "loss_avg": running_means[0],
                    "loss_recent": recent_means[0],
                    "components_avg": running_components,
                    "components_recent": recent_components,
                }
                with (Path(args.output_dir) / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"step={step}/{steps} loss_avg={running_means[0]:.5f} "
                    f"loss_recent={recent_means[0]:.5f} components_recent=[{parts}]",
                    flush=True,
                )
            window_sums.zero_()
            window_steps = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_frame_event.yaml")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gpus", nargs="+", type=int, metavar="ID", help="GPU IDs to use, e.g. --gpus 0 1 2 3")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--eval-checkpoint", type=str)
    cli = parser.parse_args()
    args = load_args_from_yaml(cli.config)
    if cli.output_dir:
        args.output_dir = cli.output_dir
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cli.gpus is not None:
        if any(gpu < 0 for gpu in cli.gpus) or len(set(cli.gpus)) != len(cli.gpus):
            parser.error("--gpus requires distinct, non-negative GPU IDs")
        if len(cli.gpus) > 1 and (cli.check_data or cli.eval_checkpoint):
            parser.error("multiple --gpus IDs are only supported for training")
        selected = ",".join(map(str, cli.gpus))
        if os.environ.get("_TRACK_ON_FE_GPU_LAUNCHED") == "1":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != selected or world_size != len(cli.gpus):
                parser.error("GPU launcher environment does not match --gpus")
        else:
            if world_size > 1:
                parser.error("use --gpus with python, or use torchrun without --gpus")
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    check=True, capture_output=True, text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                parser.error(f"cannot list GPU IDs with nvidia-smi: {exc}")
            available = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
            missing = set(cli.gpus) - available
            if missing:
                parser.error(f"GPU IDs {sorted(missing)} are unavailable; available IDs: {sorted(available)}")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = selected
            environment["_TRACK_ON_FE_GPU_LAUNCHED"] = "1"
            print(f"Launching on GPU IDs: {selected}", flush=True)
            command = [sys.executable]
            if len(cli.gpus) > 1:
                command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(cli.gpus)}"]
            command += [str(Path(__file__).resolve()), *sys.argv[1:]]
            raise SystemExit(subprocess.call(command, env=environment))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if cli.check_data:
        if rank == 0:
            args.output_dir = str(create_run_directory(args.output_dir))
            print(f"run directory: {args.output_dir}", flush=True)
            inspect_data(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")

    if world_size > 1 and not cli.eval_checkpoint:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank if world_size > 1 else 0)
    if cli.eval_checkpoint and rank != 0:
        return
    args.output_dir = str(create_run_directory(args.output_dir))
    if rank == 0:
        print(f"run directory: {args.output_dir}", flush=True)
        with (Path(args.output_dir) / "run_config.json").open("w") as handle:
            json.dump({"config_path": cli.config, "args": vars(args)}, handle, indent=2)
    model = build_model(args, device)
    if cli.eval_checkpoint:
        load_event_checkpoint(model, cli.eval_checkpoint, device)
    else:
        train_loader = make_loader(args, training=True, rank=rank, world_size=world_size)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        steps = cli.max_steps if cli.max_steps is not None else (
            args.overfit_steps if cli.overfit_one_batch else args.max_steps
        )
        if steps < 1:
            raise ValueError("Training steps must be positive")
        train(model, train_loader, args, device, steps, cli.overfit_one_batch, rank=rank)
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
            model = model.module
        if rank != 0:
            return
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("overfit.pt" if cli.overfit_one_batch else "frame_event_last.pt")
        torch.save({"model": model.state_dict(), "step": steps}, path)
        print(f"checkpoint: {path}")

    if not cli.overfit_one_batch and not cli.skip_eval:
        val_loader = make_loader(args, training=False)
        results = {
            mode: evaluate(model, val_loader, args, device, mode)
            for mode in ("rgb", "fused", "zero")
        }
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.json").open("w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
