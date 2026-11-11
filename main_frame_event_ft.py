"""Frame-Event fine-tuning with real pseudo labels and optional Kubric mixing.

Use --check-data for alignment previews and --max-steps for a short training run.
Use --gpus 0 1 or torchrun for DDP. --resume restores this entry's training state.
With --resume, --config overrides saved settings; equivalent batch size is required.
TensorBoard scalars are saved under each run's tensorboard/ directory.

Examples:
    python main_frame_event_ft.py --check-data --sequences 0000 0020
    python main_frame_event_ft.py --gpus 0
    python main_frame_event_ft.py --gpus 0 --resume /path/to/run/last.pt
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from dataset.event_kubric_rgb import build_sources
from dataset.frame_event_erf import erf_source
from dataset.frame_event_real import FrameEventReal, build_real_index, real_source_args
from dataset.frame_event_kubric import FrameEventKubric
from dataset.frame_event_mixed import MixedFrameEventDataset, MixedFrameEventSampler, WeightedSourceSampler
from model.loss import Loss_Function
from utils.tensorboard_utils import FrameEventTensorBoard
from utils.train_utils import get_scheduler, load_args_from_yaml, load_pretrained_weights
from utils.memory_training import build_ft_optimizer, MemoryTrainingSchedule, training_clip


class RealFineTuneLoss(Loss_Function):
    """Track-On-R's five FT terms, with unknown labels excluded explicitly."""

    def forward(self, out, batch):
        tracks = batch["tracks"]
        visibility = batch["visibility"].bool()
        height, width = self.size
        valid = (batch["label_valid"].bool() & torch.isfinite(tracks).all(-1)
                 & (tracks[..., 0] >= 0) & (tracks[..., 0] < width)
                 & (tracks[..., 1] >= 0) & (tracks[..., 1] < height))
        tracks = torch.where(valid[..., None], tracks, 0)
        query_times = batch["queries"][..., 0].long()
        visible_points = visibility & valid
        offsets = out["O"]
        offset = sum(self.offset_loss(offsets[:, :, layer], out["P_patch"], tracks,
                                      visible_points, query_times)
                     for layer in range(offsets.shape[2])) / offsets.shape[2]

        # Original FT supervises top-k uncertainty even at known invisible times.
        # A missing label must not become an "invisible/uncertain" training target.
        after_query = torch.arange(tracks.shape[1], device=tracks.device)[None, :, None] > query_times[:, None]
        mask = (valid & after_query)[..., None]
        distances = torch.linalg.vector_norm(out["P_patch_top_k"].detach().float() - tracks[..., None, :], dim=-1)
        uncertain = (distances > 12) | ~visibility[..., None]
        uncertainty = F.binary_cross_entropy_with_logits(
            out["U_logit_top_k"].float(), uncertain.float(), reduction="none",
        )
        uncertainty = (uncertainty * mask).sum() / (mask.sum() * uncertainty.shape[-1]).clamp_min(1)
        return {
            "patch1": self.patch_classification_loss(out["C1"], tracks, visible_points, query_times),
            "patch2": self.patch_classification_loss(out["C2"], tracks, visible_points, query_times),
            "offset": offset,
            "ranking": self.region_rank_loss(out["S_logit_top_k"], out["P_patch_top_k"],
                                             tracks, visible_points, query_times),
            "topk_uncertainty": uncertainty,
        }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_resume_config(saved_config, args):
    ignored = {"output_dir", "num_workers", "val_num_workers", "grad_accum_steps"}
    if ({key: value for key, value in saved_config.items() if key not in ignored}
            != {key: value for key, value in vars(args).items() if key not in ignored}):
        raise ValueError("Resume configuration differs from the saved run; only gradient accumulation, "
                         "worker counts and output_dir may differ")


def resume_batch_index(state, args, batches_per_epoch, world_size):
    old_accum = state["config"]["grad_accum_steps"]
    old_batches = state["batches_per_epoch"]
    old_world_size = state.get("world_size")
    if old_world_size is None:
        if old_accum != args.grad_accum_steps:
            raise ValueError("Checkpoint lacks world_size; cannot change gradient accumulation safely")
    elif (state["config"]["bs"] * old_world_size * old_accum
          != args.bs * world_size * args.grad_accum_steps):
        raise ValueError("Resume must keep bs * world_size * grad_accum_steps unchanged")
    if old_batches * args.grad_accum_steps != batches_per_epoch * old_accum:
        raise ValueError("Resume must keep the same optimizer-update budget per epoch")
    batch_index = state["batch_in_epoch"]
    if (not 0 <= batch_index <= old_batches
            or (batch_index % old_accum and batch_index != old_batches)):
        raise ValueError("Resume checkpoint must be at a gradient-accumulation boundary")
    return min(math.ceil(batch_index / old_accum) * args.grad_accum_steps, batches_per_epoch)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)


def mixture_weights(args, epoch, batch_index, batches_per_epoch):
    if not (getattr(args, "syn_real_training", False) and getattr(args, "schedule_syn_weight", False)):
        return 1.0, 1.0
    progress = (epoch * batches_per_epoch + batch_index) / max(args.epoch_num * batches_per_epoch - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - progress, 1.0 + progress


def make_loaders(args, rank, world_size, run_dir, sequences=None, resume=False):
    multi_source = hasattr(args, "real_sources")
    mixed = getattr(args, "syn_real_training", False)
    options = ([real_source_args(args, source) for source in args.real_sources]
               if multi_source else [erf_source(args)])
    names = [source.name for source in options]
    if (not names or len(set(names)) != len(names)
            or any(not name or name == "synthetic" or "/" in name for name in names)):
        raise ValueError("Real source names must be nonempty, unique and different from synthetic")
    if multi_source and sequences:
        if any("/" not in name or name.split("/", 1)[0] not in names for name in sequences):
            raise ValueError("With real_sources, --sequences expects source/sequence names")
    payload = [None]
    if rank == 0:
        indices = []
        for source in options:
            selected = ([name.split("/", 1)[1] for name in sequences if name.startswith(source.name + "/")]
                        if multi_source and sequences else sequences)
            sequence_names = (sorted(selected) if selected is not None else
                              sorted(path.name for path in Path(source.root).iterdir() if path.is_dir()))
            if len(sequence_names) != len(set(sequence_names)) or not sequence_names:
                raise ValueError(f"{source.name}: sequence list must be non-empty and unique")
            val_count = max(1, round(len(sequence_names) * args.val_fraction)) if args.val_fraction else 0
            if val_count >= len(sequence_names):
                raise ValueError(f"{source.name}: sequence split leaves no training sequences")
            permutation = np.random.default_rng(args.split_seed).permutation(sequence_names)
            val_names = set(permutation[:val_count].tolist())
            records, summary = build_real_index(source, sequence_names)
            for item in summary:
                item["split"] = "val" if item["sequence"] in val_names else "train"
            split = {"train": [name for name in sequence_names if name not in val_names], "val": sorted(val_names)}
            indices.append({"split": split, "summary": summary, "clips": records})
            print(f"{source.name}: {len(records)} / {sum(item['candidates'] for item in summary)} clips retained; "
                  f"sequences train={len(split['train'])}, val={len(split['val'])}", flush=True)
        index = {"sources": dict(zip(names, indices))} if multi_source else indices[0]
        index_path = run_dir / "data_index.json"
        if resume:
            if json.loads(index_path.read_text()) != index:
                raise ValueError("Real data index changed since the saved run")
        else:
            write_json(index_path, index)
        payload[0] = indices
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    indices = payload[0]
    split = ({name: index["split"] for name, index in zip(names, indices)}
             if multi_source else indices[0]["split"])
    train_sets, val_sets = [], {}
    for source_id, (source, index) in enumerate(zip(options, indices), start=int(mixed)):
        records, source_split = index["clips"], index["split"]
        identity = source_id if multi_source else None
        train_sets.append(FrameEventReal(
            source, [r for r in records if r["sequence"] in source_split["train"]], True, identity,
        ))
        if source_split["val"]:
            val_sets[source.name] = FrameEventReal(
                source, [r for r in records if r["sequence"] in source_split["val"]], False, identity,
            )
    train_set = ConcatDataset(train_sets) if multi_source else train_sets[0]
    train_sampler, synthetic_set, synthetic_sources = None, None, None
    if mixed:
        sources = [None]
        if rank == 0:
            paths, _, groups, weights = build_sources(args)
            sources[0] = {"paths": [str(path.resolve()) for path in paths],
                          "group_ids": groups, "group_weights": weights}
            path = run_dir / "synthetic_index.json"
            if resume:
                if json.loads(path.read_text()) != sources[0]:
                    raise ValueError("Synthetic data index changed since the saved run")
            else:
                write_json(path, sources[0])
        if dist.is_initialized():
            dist.broadcast_object_list(sources, src=0)
        synthetic_sources = sources[0]
        frame_interval = getattr(args, "synthetic_frame_interval", None)
        if frame_interval is None:
            frame_interval = args.rgb_native_interval // args.event_native_stride
        synthetic_set = FrameEventKubric(
            root=args.event_kubric_root, clip_length=args.T, num_points=args.N,
            crop_size=args.input_size,
            frame_interval=frame_interval,
            training=True, seed=args.seed, sequence_paths=[Path(path) for path in synthetic_sources["paths"]],
            anchor_aligned=True, spatial_augmentation=getattr(args, "spatial_augmentation", False),
            spatial_scale_range=getattr(args, "spatial_scale_range", (1.0, 1.2)),
        )
    if multi_source:
        ranges = {"synthetic": (len(train_set), len(synthetic_set))} if mixed else {}
        offset = 0
        for name, dataset in zip(names, train_sets):
            ranges[name] = (offset, len(dataset))
            offset += len(dataset)
        if mixed:
            train_set = MixedFrameEventDataset(train_set, synthetic_set, source_names=ranges)
        else:
            train_set.source_names = tuple(ranges)
        train_sampler = WeightedSourceSampler(
            ranges, args.sampling_weights, args.updates_per_epoch * args.grad_accum_steps,
            args.bs, rank, world_size, args.seed,
            synthetic_sources["group_ids"] if mixed else None,
            synthetic_sources["group_weights"] if mixed else None,
        )
        if rank == 0:
            counts = {name: count * args.bs * world_size
                      for name, count in zip(train_sampler.names, train_sampler.counts.tolist())}
            print(f"Samples per epoch by source: {counts}", flush=True)
    elif mixed:
        train_set = MixedFrameEventDataset(train_set, synthetic_set)
        train_sampler = MixedFrameEventSampler(
            train_set, synthetic_sources["group_ids"], synthetic_sources["group_weights"],
            args.bs, rank, world_size, args.seed,
        )
    train_set.frequency_keys = tuple((name, stride) for name, dataset in zip(names, train_sets)
                                     if len(dataset.strides) > 1 for stride in dataset.strides)

    def loader(dataset, training):
        workers = args.num_workers if training else args.val_num_workers
        if training:
            sampler = train_sampler if train_sampler is not None else DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, seed=args.seed,
            )
        else:
            sampler = range(rank, len(dataset), world_size)
        return DataLoader(
            dataset, batch_size=args.bs, sampler=sampler, num_workers=workers,
            pin_memory=True, worker_init_fn=seed_worker, persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(args.seed + rank),
            **({"prefetch_factor": 1} if workers else {}),
        )
    val_loaders = {
        name: ({stride: loader(dataset.for_stride(stride), False) for stride in dataset.strides
                if any(str(stride) in record["frequencies"] for record in dataset.records)}
               if len(dataset.strides) > 1 else loader(dataset, False))
        for name, dataset in val_sets.items()
    }
    validation = (val_loaders if multi_source else next(iter(val_loaders.values()), None))
    return loader(train_set, True), validation or None, split


def inspect_data(loader, run_dir):
    from PIL import Image, ImageDraw

    real_dataset = getattr(loader.dataset, "real_dataset", loader.dataset)
    if isinstance(real_dataset, ConcatDataset):
        for dataset in real_dataset.datasets:
            directory = run_dir / dataset.args.name
            directory.mkdir(exist_ok=True)
            inspect_data(argparse.Namespace(dataset=dataset), directory)
        return
    sample = loader.dataset[0]
    for key, value in sample.items():
        print(f"{key}: {tuple(value.shape)}, {value.dtype}")
    assert sample["img_ifnew"][0] and not sample["events"][0].any()
    assert torch.equal(sample["queries"][:, 1:], sample["tracks"][0])
    assert all(torch.isfinite(value).all() for value in sample.values())
    new_steps = sample["img_ifnew"].nonzero().flatten().tolist()
    for step in range(1, len(sample["video"])):
        if not sample["img_ifnew"][step]:
            assert torch.equal(sample["video"][step], sample["video"][step - 1])
    panels = []
    for step in (0, 1, len(sample["video"]) - 1):
        rgb = Image.fromarray(sample["video"][step].byte().permute(1, 2, 0).numpy())
        event = Image.fromarray((sample["events"][step].amax(0).clamp(0, 1) * 255).byte().numpy()).convert("RGB")
        for panel in (rgb, event):
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8), f"native={sample['frame_indices'][step].item()} new_rgb={bool(sample['img_ifnew'][step])}", fill="orange")
            for x, y in sample["tracks"][step, sample["visibility"][step] & sample["label_valid"][step]][:32].tolist():
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red")
            panels.append(panel)
    width, height = panels[0].size
    preview = Image.new("RGB", (2 * width, 3 * height))
    for index, panel in enumerate(panels):
        preview.paste(panel, ((index % 2) * width, (index // 2) * height))
    preview.save(run_dir / "sample_alignment.png")
    print(f"new RGB steps: {new_steps}; native indices: {sample['frame_indices'].tolist()}")
    print(f"alignment preview: {run_dir / 'sample_alignment.png'}", flush=True)


def validation_size(loader):
    if isinstance(loader, dict):
        return sum(validation_size(child) for child in loader.values())
    return len(loader.dataset) if loader is not None else 0


@torch.no_grad()
def evaluate(model, loader, args, device, weights=None):
    if isinstance(loader, dict):
        scores = {}
        for name, source_loader in loader.items():
            stride_weights = (next(iter(source_loader.values())).dataset.args.stride_weights
                              if isinstance(source_loader, dict) else None)
            scores[name] = evaluate(model, source_loader, args, device, stride_weights)
        weights = args.sampling_weights if weights is None else weights
        total_weight = sum(weights[name] for name in scores)
        result = {metric: sum(weights[name] * score[metric] for name, score in scores.items())
                  / total_weight for metric in ("pseudo_error_px", "supervised_fraction")}
        for name, score in scores.items():
            label = f"stride_{name}" if isinstance(name, int) else name
            result.update({f"{label}/{metric}": value for metric, value in score.items()})
        return result
    model.eval()
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            out = model(batch["video"], batch["queries"].clone(), p_mask=0,
                        events=batch["events"], img_ifnew=batch["img_ifnew"])
        after = torch.arange(batch["tracks"].shape[1], device=device)[None, :, None] > batch["queries"][:, None, :, 0]
        mask = batch["label_valid"] & batch["visibility"] & after
        error = torch.linalg.vector_norm(out["P"].float() - batch["tracks"], dim=-1)
        if not torch.isfinite(error[mask]).all():
            raise RuntimeError("Non-finite validation predictions")
        totals[0] += error[mask].double().sum()
        totals[1] += mask.sum()
        totals[2] += after.sum()
        del out, batch
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[1] == 0:
        raise RuntimeError("Validation has no supervised points")
    return {"pseudo_error_px": (totals[0] / totals[1]).item(),
            "supervised_fraction": (totals[1] / totals[2]).item()}


def train(model, loader, val_loader, args, cli, device, rank, run_dir, split, logger):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    # Register all originally trainable parameters before temporarily freezing FE.
    # DDP is also constructed before freezing, with dynamic unused-parameter detection.
    optimizer = build_ft_optimizer(raw_model, args)
    scheduler = get_scheduler(args, optimizer, loader, warmup=False)
    stages = MemoryTrainingSchedule(raw_model, args)
    scaler = torch.GradScaler(enabled=args.amp)
    loss_fn = RealFineTuneLoss(args)
    mixed = getattr(args, "syn_real_training", False)
    synthetic_loss_fn = Loss_Function(args) if mixed else None
    synthetic_names = ("patch1", "patch2", "visibility", "offset", "uncertainty", "topk_uncertainty", "ranking")
    loss_names = synthetic_names if mixed else ("patch1", "patch2", "offset", "ranking", "topk_uncertainty")
    source_names = getattr(loader.dataset, "source_names", ())
    frequency_keys = getattr(loader.dataset, "frequency_keys", ())
    start_epoch, start_batch, steps, best_error = 0, 0, 0, float("inf")
    successful_updates = 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if cli.resume:
        state = torch.load(cli.resume, map_location="cpu", weights_only=False)
        check_resume_config(state["config"], args)
        if state["split"] != split:
            raise ValueError("Resume sequence split differs from the saved run")
        start_batch = resume_batch_index(state, args, len(loader), world_size)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, steps = state["epoch"], state["step"]
        best_error = state["best_pseudo_error_px"]
        successful_updates = state.get("successful_updates", scheduler.last_epoch)
        if rank == 0:
            write_json(run_dir / "run_config.json", vars(args))
            print(f"Resumed {cli.resume}: epoch={start_epoch + 1}, batch={start_batch}, step={steps}; "
                  f"world_size={state.get('world_size', 'unknown')} -> {world_size}, "
                  f"grad_accum_steps={state['config']['grad_accum_steps']} -> {args.grad_accum_steps}", flush=True)
        del state
    logger.start(steps)
    if rank == 0:
        val_clips = validation_size(val_loader)
        print(f"Train samples/epoch={len(loader.sampler) * world_size}, "
              f"val clips={val_clips}; "
              f"effective batch={args.bs * world_size * args.grad_accum_steps}; "
              f"updates/epoch={math.ceil(len(loader) / args.grad_accum_steps)}", flush=True)
        if mixed:
            mix = args.sampling_weights if source_names else "synthetic:real=1:1"
            print(f"Mixing {mix}; synthetic losses=7, real losses=5; "
                  f"linear loss-weight schedule={getattr(args, 'schedule_syn_weight', False)}", flush=True)
        print(f"Adaptation: {stages.steps} successful updates, T={stages.length}; "
              f"joint T={args.T}; recent/RGB memory={raw_model.M}/{raw_model.rgb_memory_size}", flush=True)
    previous_phase = None
    for epoch in range(start_epoch, args.epoch_num):
        if cli.max_steps is not None and steps >= cli.max_steps:
            break
        loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window = None
        window_batches = 0
        next_batch = 0
        for batch_index, raw_batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            synthetic = bool(raw_batch["is_synthetic"][0]) if mixed else False
            if mixed and not (raw_batch["is_synthetic"] == synthetic).all():
                raise ValueError("FT expects homogeneous real or synthetic microbatches")
            if source_names:
                source_id = int(raw_batch["source_id"][0])
                if not (raw_batch["source_id"] == source_id).all():
                    raise ValueError("FT expects each microbatch to contain one data source")
            syn_weight, real_weight = mixture_weights(args, epoch, batch_index, len(loader))
            length = stages.apply(successful_updates)
            phase = stages.phase(successful_updates)
            if phase != previous_phase:
                if rank == 0:
                    print(f"Training phase: {phase}, successful_updates={successful_updates}, T={length}", flush=True)
                previous_phase = phase
            # Slice before device transfer so the adaptation phase uses less VRAM.
            batch = move_batch(training_clip(raw_batch, length), device)
            clip_length = batch["video"].shape[1]
            window_start = batch_index // args.grad_accum_steps * args.grad_accum_steps
            window_size = min(args.grad_accum_steps, len(loader) - window_start)
            update_now = batch_index - window_start + 1 == window_size
            sync = model.no_sync() if isinstance(model, DistributedDataParallel) and not update_now else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    out = model(batch["video"], batch["queries"].clone(), events=batch["events"],
                                img_ifnew=batch["img_ifnew"], p_mask=args.memory_mask_probability)
                    if synthetic:
                        parts = dict(zip(synthetic_names, synthetic_loss_fn(
                            out, batch["tracks"], batch["visibility"], batch["queries"][..., 0].long(),
                        )))
                    else:
                        parts = loss_fn(out, batch)
                    raw_loss = sum(parts.values())
                    loss = (syn_weight if synthetic else real_weight) * raw_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite FT loss at epoch={epoch}, batch={batch_index}")
                scaler.scale(loss / window_size).backward()
            values = [loss.detach().float(),
                      *(parts[name].detach().float() if name in parts else loss.new_zeros((), dtype=torch.float32)
                        for name in loss_names)]
            if mixed:
                values.extend((raw_loss.detach().float() * synthetic, raw_loss.detach().float() * (not synthetic),
                               loss.new_tensor(float(synthetic)), loss.new_tensor(float(not synthetic)),
                               loss.new_tensor(syn_weight), loss.new_tensor(real_weight)))
            if source_names:
                source_values = loss.new_zeros(2 * len(source_names), dtype=torch.float32)
                source_values[source_id] = raw_loss.detach().float()
                source_values[len(source_names) + source_id] = 1
                values.extend(source_values.unbind())
            if frequency_keys:
                count = len(frequency_keys)
                frequency_values = loss.new_zeros(3 * count, dtype=torch.float32)
                strides = raw_batch.get("event_stride")
                if strides is not None:
                    for index, (name, stride) in enumerate(frequency_keys):
                        if name != source_names[source_id]:
                            continue
                        selected = strides == stride
                        frequency_values[2 * count + index] = selected.float().mean()
                        # A mixed-frequency batch has no single per-frequency loss.
                        if selected.all():
                            frequency_values[index] = raw_loss.detach().float()
                            frequency_values[count + index] = 1
                values.extend(frequency_values.unbind())
            values = torch.stack(values)
            window = values if window is None else window + values
            window_batches += 1
            next_batch = batch_index + 1
            del out, parts, loss, raw_loss, batch
            if not update_now:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            if not args.amp and not torch.isfinite(norm):
                raise RuntimeError("Non-finite FT gradients")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= previous_scale
            if updated:
                scheduler.step()
                successful_updates += 1
            elif rank == 0:
                print(f"AMP skipped update at epoch={epoch + 1}, batch={next_batch}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if dist.is_initialized():
                dist.all_reduce(window)
            means = (window / (window_batches * world_size)).tolist()
            components = dict(zip(loss_names, means[1:1 + len(loss_names)]))
            mixture = {}
            metric_offset = 1 + len(loss_names)
            if mixed:
                syn_loss, real_loss, syn_fraction, real_fraction, syn_weight, real_weight = means[metric_offset:metric_offset + 6]
                metric_offset += 6
                mixture = {"synthetic_loss": syn_loss / syn_fraction if syn_fraction else None,
                           "real_loss": real_loss / real_fraction if real_fraction else None,
                           "synthetic_fraction": syn_fraction, "real_fraction": real_fraction,
                           "syn_weight": syn_weight, "real_weight": real_weight}
            for source_id, name in enumerate(source_names):
                fraction = means[metric_offset + len(source_names) + source_id]
                mixture[f"sources/{name}/loss"] = means[metric_offset + source_id] / fraction if fraction else None
                mixture[f"sources/{name}/fraction"] = fraction
            metric_offset += 2 * len(source_names)
            for index, (name, stride) in enumerate(frequency_keys):
                count = len(frequency_keys)
                batches = means[metric_offset + count + index]
                fraction = means[metric_offset + 2 * count + index]
                source_fraction = mixture[f"sources/{name}/fraction"]
                prefix = f"frequencies/{name}/stride_{stride}"
                mixture[f"{prefix}/loss"] = means[metric_offset + index] / batches if batches else None
                mixture[f"{prefix}/fraction"] = fraction / source_fraction if source_fraction else None
            training_metrics = {**mixture, "successful_updates": successful_updates,
                                "adaptation": int(phase == "adaptation"), "clip_length": clip_length,
                                "memory_lr": optimizer.param_groups[-1]["lr"]}
            logger.training(steps, loss=means[0], components=components,
                            lr=optimizer.param_groups[0]["lr"], epoch=epoch + 1, grad_norm=norm,
                            optimizer_updated=updated, amp_scale=scaler.get_scale(), microbatches=window_size,
                            metrics=training_metrics)
            if rank == 0 and (steps % args.log_every == 0 or steps == 1 or next_batch == len(loader)):
                record = {"epoch": epoch + 1, "step": steps, "loss": means[0],
                          "components": components, "lr": optimizer.param_groups[0]["lr"],
                          "optimizer_updated": updated, "grad_norm_before_clip": float(norm),
                          "amp_scale": scaler.get_scale(), "phase": phase,
                          "successful_updates": successful_updates, "clip_length": clip_length}
                if mixture:
                    record["mixture"] = mixture
                with (run_dir / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                mix_text = (f" syn_w={syn_weight:.4f} real_w={real_weight:.4f}"
                            f" real_fraction={mixture['real_fraction']:.2f}" if mixed else "")
                print(f"epoch={epoch + 1} step={steps} loss={means[0]:.5f} lr={record['lr']:.8g}{mix_text}", flush=True)
            window, window_batches = None, 0
            if cli.max_steps is not None and steps >= cli.max_steps:
                break

        score = evaluate(raw_model, val_loader, args, device) if val_loader is not None and not cli.skip_eval else None
        improved = score is not None and score["pseudo_error_px"] < best_error
        if improved:
            best_error = score["pseudo_error_px"]
        complete = next_batch == len(loader)
        if rank == 0:
            if score is not None:
                logger.evaluation(score, steps, epoch=epoch + 1)
                with (run_dir / "eval_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, "step": steps, **score}) + "\n")
                print(f"Pseudo-label validation (not GT): {score}", flush=True)
            state = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch + int(complete), "batch_in_epoch": 0 if complete else next_batch,
                     "step": steps, "best_pseudo_error_px": best_error, "config": vars(args), "split": split,
                     "batches_per_epoch": len(loader), "world_size": world_size}
            state.update(successful_updates=successful_updates, phase=stages.phase(successful_updates))
            paths = [run_dir / "last.pt"]
            if improved:
                paths.append(run_dir / "best_pseudo.pt")
            if complete and args.save_every and (epoch + 1) % args.save_every == 0:
                paths.append(run_dir / f"epoch_{epoch + 1:04d}.pt")
            for path in paths:
                temporary = path.with_suffix(".pt.tmp")
                torch.save(state, temporary)
                os.replace(temporary, path)
            print(f"checkpoint: {run_dir / 'last.pt'}", flush=True)
        if dist.is_initialized():
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to config/train_frame_event_real_ft.yaml; "
                        "with --resume, defaults to the saved run_config.json")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--sequences", nargs="+", help="Restrict data; use source/sequence with real_sources")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Stop after this many total optimizer attempts")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Restore this entry's last.pt, including optimizer and progress")
    cli = parser.parse_args()
    if cli.max_steps is not None and cli.max_steps < 1:
        parser.error("--max-steps must be positive")
    if cli.resume and (cli.check_data or cli.output_dir):
        parser.error("--resume cannot be combined with --check-data or --output-dir")
    if cli.gpus is not None:
        if not cli.gpus or min(cli.gpus) < 0 or len(set(cli.gpus)) != len(cli.gpus):
            parser.error("--gpus requires distinct non-negative GPU IDs")
        if cli.check_data and len(cli.gpus) > 1:
            parser.error("--check-data uses one process")
        if "LOCAL_RANK" not in os.environ and not cli.check_data:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cli.gpus))
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(cli.gpus)}",
                       str(Path(__file__).resolve()), *sys.argv[1:]]
            raise SystemExit(subprocess.call(command, env=environment))

    saved_config = (json.loads((Path(cli.resume).resolve().parent / "run_config.json").read_text())
                    if cli.resume else None)
    config_path = cli.config or Path(__file__).resolve().parent / "config/train_frame_event_real_ft.yaml"
    args = argparse.Namespace(**saved_config) if saved_config is not None and cli.config is None else load_args_from_yaml(config_path)
    # JSON stores mapping keys as strings; keep checkpoint/config comparisons stable.
    for source in getattr(args, "real_sources", ()):
        if "event_stride_weights" in source:
            source["event_stride_weights"] = {str(key): value for key, value in source["event_stride_weights"].items()}
    args.sequences = (sorted(cli.sequences) if cli.sequences else
                      saved_config.get("sequences") if saved_config is not None else None)
    if saved_config is not None:
        check_resume_config(saved_config, args)
    if (args.T < 2 or args.N < 1 or args.bs < 1 or args.grad_accum_steps < 1 or args.epoch_num < 1
            or not 0 < args.min_visible_fraction <= 1 or not 0 <= args.val_fraction < 1
            or min(args.num_workers, args.val_num_workers) < 0 or args.log_every < 1
            or not 0 <= args.memory_mask_probability < 1):
        raise ValueError("Invalid ERF training, frequency, split or trajectory screening configuration")
    if not hasattr(args, "real_sources") and (args.event_native_stride < 1 or args.rgb_native_interval < 1
                                               or args.rgb_native_interval % args.event_native_stride):
        raise ValueError("Invalid ERF event/RGB frequencies")
    if hasattr(args, "real_sources"):
        if not isinstance(args.updates_per_epoch, int) or args.updates_per_epoch < 1:
            raise ValueError("updates_per_epoch must be a positive integer")
        for source in args.real_sources:
            real_source_args(args, source)
    if args.event_fusion_type != "recurrent" or args.event_channels != 10:
        raise ValueError("This entry expects the recurrent 10-channel FE model")
    if (int(getattr(args, "adaptation_steps", 0)) < 0
            or not 2 <= int(getattr(args, "adaptation_T", args.T)) <= args.T
            or not math.isfinite(float(getattr(args, "memory_lr", args.lr)))
            or float(getattr(args, "memory_lr", args.lr)) <= 0):
        raise ValueError("Invalid memory learning rate or adaptation schedule")
    if cli.output_dir:
        args.output_dir = cli.output_dir
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    if not cli.check_data:
        if not torch.cuda.is_available():
            raise RuntimeError("FE fine-tuning requires CUDA")
        torch.cuda.set_device(device)
        if "RANK" in os.environ:
            dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cv2.setNumThreads(1)
    directory = [str(Path(cli.resume).resolve().parent) if cli.resume else None]
    if rank == 0 and not cli.resume:
        directory[0] = str(Path(args.output_dir) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
        Path(directory[0]).mkdir(parents=True)
    if dist.is_initialized():
        dist.broadcast_object_list(directory, src=0)
    run_dir = Path(directory[0])
    if rank == 0:
        if not cli.resume:
            write_json(run_dir / "run_config.json", vars(args))
        print(f"run directory: {run_dir}", flush=True)
    logger = FrameEventTensorBoard(args, run_dir, rank)
    try:
        train_loader, val_loader, split = make_loaders(args, rank, world_size, run_dir, args.sequences, bool(cli.resume))
        if cli.check_data:
            inspect_data(train_loader, run_dir)
            return
        from model.trackon import Track_On2

        model = Track_On2(args).to(device)
        if not cli.resume:
            load_pretrained_weights(model, args.checkpoint_path)
        if dist.is_initialized():
            model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        train(model, train_loader, val_loader, args, cli, device, rank, run_dir, split, logger)
    finally:
        logger.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
