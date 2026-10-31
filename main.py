import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import json
import math
import warnings

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pathlib import Path

import argparse

from utils.train_utils import init_distributed_mode, fix_random_seeds
from utils.train_utils import get_dataloaders, get_scheduler
from utils.train_utils import restart_from_checkpoint
from utils.train_utils import load_args_from_yaml
from utils.train_utils import load_pretrained_weights

from utils.tensorboard_utils import init_tensorboard, TensorBoardLossTracker, LOSS_NAMES

from utils.coord_utils import get_queries, get_points_on_a_grid

from utils.eval_utils import Evaluator, compute_tapvid_metrics

from model.trackon import Track_On2
from model.loss import Loss_Function


def prepare_run_directory(args, config_path):
    """Create one timestamped run directory and snapshot its configuration."""
    result = {"path": None, "error": None}
    if args.rank == 0:
        try:
            base = Path(args.model_save_path)
            base.mkdir(parents=True, exist_ok=True)
            started_at = datetime.datetime.now().astimezone()
            stamp = started_at.strftime("%Y%m%d_%H%M%S")
            for suffix in range(100):
                name = stamp if suffix == 0 else f"{stamp}_{suffix:02d}"
                candidate = base / name
                try:
                    candidate.mkdir()
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError(f"Could not create a unique run directory under {base}")

            source = Path(config_path)
            (candidate / "run_config.yaml").write_bytes(source.read_bytes())
            resolved_args = vars(args).copy()
            resolved_args["model_save_path"] = str(candidate)
            metadata = {
                "config_path": str(source.resolve()),
                "started_at": started_at.isoformat(timespec="seconds"),
                "run_directory": str(candidate.resolve()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "command": [sys.executable, *sys.argv],
                "args": resolved_args,
            }
            with (candidate / "run_config.json").open("w") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            result["path"] = str(candidate)
        except Exception as exc:
            result["error"] = str(exc)

    if dist.is_initialized():
        shared = [result]
        dist.broadcast_object_list(shared, src=0)
        result = shared[0]
    if result["error"]:
        raise RuntimeError(f"Failed to prepare training run directory: {result['error']}")
    args.model_save_path = result["path"]
    if args.rank == 0:
        print(f"Run directory: {Path(args.model_save_path).resolve()}", flush=True)
    return Path(args.model_save_path)


def save_training_checkpoints(state, run_dir, epoch, is_best, save_every):
    """Keep latest, selected best, and optional numbered epoch snapshots."""
    output = Path(run_dir)

    def save_atomic(path):
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    save_atomic(output / "last.pt")
    # Preserve the older checkpoint name without writing a second full copy.
    legacy = output / "checkpoint.pt"
    if not legacy.exists() and not legacy.is_symlink():
        legacy.symlink_to("last.pt")
    if is_best:
        save_atomic(output / "best.pt")
    if save_every and epoch % save_every == 0:
        save_atomic(output / f"epoch_{epoch:04d}.pt")


def nonfinite_loss_details(args, epoch, batch_index, loss_values, loss_fn, visibility,
                           query_times, video, tracks_gt, queries, out, model, sample_info,
                           bad_running_buffers, bad_running_buffers_before, probes):
    """Build a warning from the failed forward without repeating or changing it."""
    point_mask, visible_mask = loss_fn.get_masks(visibility, query_times)
    samples = []
    for sample_index in range(video.shape[0]):
        sample = {"batch_index": sample_index, "path": "unknown"}
        if sample_info is not None:
            sample["path"] = sample_info["path"][sample_index]
            sample["requested_index"] = int(sample_info["requested_index"][sample_index])
            sample["fallback_attempt"] = int(sample_info["fallback_attempt"][sample_index])
            sample["worker_id"] = int(sample_info["worker_id"][sample_index])
            sample["first_frame"] = int(sample_info["first_frame"][sample_index])
            sample["last_frame"] = int(sample_info["last_frame"][sample_index])
        samples.append(sample)

    def tensor_info(tensor):
        finite = torch.isfinite(tensor)
        finite_values = tensor[finite]
        return {
            "shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
            "nan": int(torch.isnan(tensor).sum().item()),
            "inf": int(torch.isinf(tensor).sum().item()),
            "finite_min": float(finite_values.min()) if finite_values.numel() else None,
            "finite_max": float(finite_values.max()) if finite_values.numel() else None,
        }

    output_info = {name: tensor_info(value) for name, value in out.items()
                   if isinstance(value, torch.Tensor) and value.is_floating_point()}
    nonfinite_parameters = [name for name, value in model.named_parameters()
                            if not bool(torch.isfinite(value).all())]
    nonfinite_buffers = [name for name, value in model.named_buffers()
                         if value.is_floating_point() and not bool(torch.isfinite(value).all())]
    def probe_summary(tensor):
        sampled = tensor.detach().reshape(-1)[::max(1, tensor.numel() // 65536)]
        finite_sample = sampled[torch.isfinite(sampled)]
        return {
            "shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
            "nonfinite": int((~torch.isfinite(tensor)).sum().item()),
            "sample_abs_max": float(finite_sample.abs().max()) if finite_sample.numel() else None,
        }

    probe_info = {name: probe_summary(value) for name, value in probes.items()}
    return (
        f"Non-finite loss or BatchNorm state; skipping batch: epoch={epoch + 1} batch={batch_index} rank={args.rank} "
        f"samples={samples} losses={dict(zip(LOSS_NAMES, loss_values.tolist()))} "
        f"point_mask_count={int(point_mask.sum())} time_mask_count={int(visible_mask.sum())} "
        f"video={tensor_info(video)} tracks={tensor_info(tracks_gt)} "
        f"queries={tensor_info(queries)} outputs={output_info} "
        f"adapter_probes={probe_info} bad_running_buffers_before={bad_running_buffers_before} "
        f"bad_running_buffers_after={bad_running_buffers} "
        f"nonfinite_parameters={nonfinite_parameters[:10]} "
        f"nonfinite_buffers={nonfinite_buffers[:10]}"
    )


def train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, loss_tracker, epoch):
    model.train()
    if getattr(args, "train_trajectory_scorer_only", False):
        # Keep the frozen tracker's running statistics and dropout fixed.
        model.eval()
        model.module.trajectory_scorer.train()

    loss_fn = Loss_Function(args)
    accumulation_steps = int(getattr(args, "grad_accum_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("grad_accum_steps must be a positive integer")
    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0,
                            file=sys.stdout, miniters=accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    skipped_microbatches = 0
    window_loss_sum = 0.0

    def finish_window():
        nonlocal accumulated, skipped_microbatches, window_loss_sum
        if not accumulated:
            if skipped_microbatches:
                loss_tracker.skip(optimizer.param_groups[0]["lr"], count=skipped_microbatches)
                skipped_microbatches = 0
            return
        if args.amp:
            scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.amp:
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updated = scaler.get_scale() >= previous_scale
        else:
            optimizer.step()
            optimizer_updated = True
        if optimizer_updated:
            lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_tracker.log(
            optimizer.param_groups[0]["lr"],
            skipped_microbatches=skipped_microbatches,
            microbatches=accumulated,
            optimizer_updated=optimizer_updated,
            grad_norm=grad_norm,
        )
        train_dataloader.set_description(
            f"Loss(avg {accumulated}): {window_loss_sum / accumulated:.5f}", refresh=False
        )
        accumulated = 0
        skipped_microbatches = 0
        window_loss_sum = 0.0

    for i, batch in enumerate(train_dataloader):
        video, tracks, visibility, k_points = batch[:4]
        sample_info = batch[4] if len(batch) > 4 else None
        video = video.cuda(non_blocking=True)               # (B, T, 3, H, W)
        tracks = tracks.cuda(non_blocking=True)             # (B, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)     # (B, T, N)
        k_points = k_points.cuda(non_blocking=True)         # (B,)

        min_k = k_points.min()
        tracks = tracks[:, :, :min_k]
        visibility = visibility[:, :, :min_k]

        B, T, N, _ = tracks.shape

        # === Scaling ===
        H_in, W_in = video.shape[-2], video.shape[-1]
        H, W = args.input_size[0], args.input_size[1]

        tracks_gt = tracks.clone()
        tracks_gt[..., 0] = tracks_gt[..., 0] * (W / W_in)
        tracks_gt[..., 1] = tracks_gt[..., 1] * (H / H_in)
        # === === ===

        queries = get_queries(tracks, visibility)             # (B, N, 3)
        query_times = queries[:, :, 0].long()        # (B, N)

        # Keep running statistics from any batch that corrupts them out of the model.
        batchnorm_buffers = [
            (f"{name}.{buffer_name}", buffer, buffer.detach().clone())
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked")
            if (buffer := getattr(module, buffer_name)) is not None
        ]
        bad_running_buffers_before = [
            name for name, buffer, _ in batchnorm_buffers
            if buffer.is_floating_point() and not bool(torch.isfinite(buffer).all())
        ]
        probes = {}
        handles = []
        module = model.module if hasattr(model, "module") else model
        encoder = getattr(getattr(module, "backbone", None), "vit_encoder", None)
        if encoder is not None:
            if hasattr(encoder, "spm"):
                handles.append(encoder.spm.register_forward_hook(
                    lambda _module, _inputs, output: probes.update(spm_c1=output[0])))
            if hasattr(encoder, "up"):
                handles.append(encoder.up.register_forward_pre_hook(
                    lambda _module, inputs: probes.update(up_input=inputs[0])))
                handles.append(encoder.up.register_forward_hook(
                    lambda _module, _inputs, output: probes.update(up_output=output)))
            if hasattr(encoder, "norm1"):
                handles.append(encoder.norm1.register_forward_pre_hook(
                    lambda _module, inputs: probes.update(norm1_input=inputs[0])))
                handles.append(encoder.norm1.register_forward_hook(
                    lambda _module, _inputs, output: probes.update(norm1_output=output)))
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(video, queries, p_mask=0.1)
                L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank = loss_fn(
                    out, tracks_gt, visibility, query_times)
        finally:
            for handle in handles:
                handle.remove()

        losses = (L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank)
        loss_values = torch.stack([part.detach().float() for part in losses])
        bad_running_buffers = [
            name for name, buffer, _ in batchnorm_buffers
            if buffer.is_floating_point() and not bool(torch.isfinite(buffer).all())
        ]
        local_bad = bool((~torch.isfinite(loss_values).all()).item()) or bool(bad_running_buffers)
        any_bad = torch.tensor(int(local_bad), device=video.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(any_bad, op=dist.ReduceOp.MAX)
        if any_bad.item():
            if local_bad:
                warnings.warn(nonfinite_loss_details(
                    args, epoch, i, loss_values, loss_fn, visibility, query_times,
                    video, tracks_gt, queries, out, model, sample_info,
                    bad_running_buffers, bad_running_buffers_before, probes,
                ), stacklevel=1)
            elif args.rank == 0:
                warnings.warn(
                    f"Non-finite loss or BatchNorm state on another rank; "
                    f"skipping batch: epoch={epoch + 1} batch={i}",
                    stacklevel=1,
                )
            with torch.no_grad():
                for _name, buffer, previous in batchnorm_buffers:
                    buffer.copy_(previous)
            skipped_microbatches += 1
            del out, losses, loss_values, L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank
            probes.clear()
            continue
        probes.clear()
        loss = sum(losses)
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Log the mean across GPUs, then average those values over the window.
        logged_losses = loss_values.clone()
        if dist.is_initialized():
            dist.all_reduce(logged_losses, op=dist.ReduceOp.SUM)
            logged_losses /= dist.get_world_size()
        loss_tracker.update(*logged_losses.unbind(0))
        window_loss_sum += float(logged_losses.sum().item())
        accumulated += 1
        del out, losses, loss_values, logged_losses, loss
        del L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank
        if accumulated == accumulation_steps:
            finish_window()

    # A short final window is normalized by its actual number of valid batches.
    finish_window()


@torch.no_grad()
def evaluate(args, val_dataloader, model, epoch, writer=None, global_step=None,
             log_to_wandb=False, verbose=False):
    model.eval()

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)       # (1, N, 3)

        # Support grid
        K = 20
        extra_queries = get_points_on_a_grid(K, (H, W), device)             # (1, K ** 2, 2)
        extra_queries = torch.cat([torch.zeros(1, int(K ** 2), 1, device=device), extra_queries], dim=-1).to(device)
        queries = torch.cat([queries, extra_queries], dim=1)

        # out, _ = model.module(video, queries, p_mask=0)
        out = model.module.forward_online(video, queries)

        pred_trajectory = out["P"][:, :, :N]                                           # (1, T, N, P)
        pred_visibility = out["V_logit"][:, :, :N].sigmoid() >= args.delta_v       # (1, T, N)

        # Timer end
        total_time += time.time() - start_time

        # === === ===
        # From CoTracker
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
        # === === ===

        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
        if verbose:
            print(f"Video {j}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / total_time
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results(epoch, log_to_wandb=log_to_wandb)
    if writer is not None:
        step = epoch if global_step is None else global_step
        for name, value in results.items():
            tag = {"aj": "AJ", "oa": "OA"}.get(name, name)
            writer.add_scalar(f"Evaluation/{tag}", value, step)
        writer.add_scalar("Evaluation/epoch", epoch, step)
        writer.flush()
    smaller_delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    return smaller_delta_avg, aj, oa


def build_training_model(args):
    model = Track_On2(args)
    init_checkpoint = getattr(args, "init_checkpoint_path", None)
    # Resuming restores the complete training state below and takes precedence.
    if init_checkpoint and args.checkpoint_path is None:
        allowed = ("trajectory_scorer.",) if model.trajectory_scorer is not None else ()
        if getattr(args, "history_candidate_scoring", False):
            allowed += ("candidate_scorer.", "memory_reliability.")
        load_pretrained_weights(model, init_checkpoint, extra_allowed_missing_prefixes=allowed)
        print("Initialized model weights for a new training run; optimizer and progress start fresh.")
    if getattr(args, "train_trajectory_scorer_only", False):
        if model.trajectory_scorer is None:
            raise ValueError("train_trajectory_scorer_only requires trajectory_history_size > 0")
        model.requires_grad_(False)
        model.trajectory_scorer.requires_grad_(True)
    return model


def main_worker(args, config_path):
    logging_backend = getattr(args, "logging_backend", "tensorboard").lower()
    if logging_backend not in ("tensorboard", "wandb"):
        raise ValueError("logging_backend must be 'tensorboard' or 'wandb'")
    best_metric = getattr(args, "best_metric", "aj")
    if best_metric not in ("aj", "oa", "delta_avg"):
        raise ValueError("best_metric must be 'aj', 'oa', or 'delta_avg'")
    save_every = getattr(args, "save_every", 0)
    if not isinstance(save_every, int) or save_every < 0:
        raise ValueError("save_every must be a non-negative integer number of epochs")
    init_distributed_mode(args)
    prepare_run_directory(args, config_path)
    fix_random_seeds(args.seed)

    start_time = time.time()

    # === Data ===
    train_dataloader, val_dataloader = get_dataloaders(args)
    if train_dataloader is not None:
        accumulation_steps = int(getattr(args, "grad_accum_steps", 1))
        if accumulation_steps < 1:
            raise ValueError("grad_accum_steps must be a positive integer")
        updates_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
        if args.rank == 0:
            print(
                f"Effective batch: {args.bs * args.world_size * accumulation_steps}; "
                f"up to {updates_per_epoch * args.epoch_num} optimizer updates "
                f"({accumulation_steps} microbatches per update)",
                flush=True,
            )
    # === === ===
    
    # === Model & Training ===
    model = build_training_model(args).to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    warmup_ratio = float(getattr(args, "warmup_ratio", 0.0))
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    # Restore the optimizer before constructing a new warmup schedule.
    lr_scheduler = None if warmup_ratio else get_scheduler(args, optimizer, train_dataloader, constant=False)
    scaler = torch.GradScaler() if args.amp else None
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 10**6:.2f}M")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}M")
    print()
    # === === ===

    # === Load from checkpoint ===
    to_restore = {"epoch": 0, "iteration": 0, "best_aj": [-1, -1], "best_oa": [-1, -1], "best_delta_avg": [-1, -1]}

    restored_scheduler_state = {}
    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

        checkpoint_objects = {"model": model, "scaler": scaler, "optimizer": optimizer}
        if lr_scheduler is not None:
            checkpoint_objects["scheduler"] = lr_scheduler
        restored_scheduler_state = restart_from_checkpoint(
            remove_module=False,
            checkpoint_path=args.checkpoint_path,
            run_variables=to_restore,
            **checkpoint_objects,
        )
    if getattr(args, "reset_best", False):
        for metric in ("best_aj", "best_oa", "best_delta_avg"):
            to_restore[metric] = [-1, -1]
    if warmup_ratio:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr
        lr_scheduler = get_scheduler(args, optimizer, train_dataloader, warmup=True)
        # Skipped batches increment the log counter but do not update the model.
        completed_steps = int(restored_scheduler_state.get("last_epoch", to_restore["iteration"]))
        if completed_steps < 0:
            raise ValueError("Checkpoint scheduler step must be non-negative")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step")
            for _ in range(completed_steps):
                lr_scheduler.step()
        if args.rank == 0:
            total_steps = math.ceil(len(train_dataloader) / int(getattr(args, "grad_accum_steps", 1))) * args.epoch_num
            warmup_steps = max(1, int(total_steps * warmup_ratio))
            print(
                f"Learning-rate warmup: {warmup_steps}/{total_steps} optimizer steps; "
                f"resumed at step {completed_steps}; current lr={optimizer.param_groups[0]['lr']:.8g}",
                flush=True,
            )
    # === === ===


    # === Start training ===
    start_epoch = to_restore["epoch"]
    best_models = {
        "aj": to_restore["best_aj"],
        "oa": to_restore["best_oa"],
        "delta_avg": to_restore["best_delta_avg"]
    }
    writer = None
    if logging_backend == "tensorboard":
        writer = init_tensorboard(args, to_restore["iteration"])
        loss_tracker = TensorBoardLossTracker(args.rank, writer, to_restore["iteration"])
    else:
        from utils.log_utils import init_wandb, Loss_Tracker
        init_wandb(args)
        loss_tracker = Loss_Tracker(args.rank)
        loss_tracker.iteration = to_restore["iteration"]

    print("Training starts")
    for epoch in range(start_epoch, args.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===

        train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, loss_tracker, epoch)
        print()
        # === === ===
        
        # === Evaluation ===
        if args.rank == 0:
            best_aj = False
            best_oa = False
            best_delta_avg = False
            eval_every = int(getattr(args, "eval_every_epochs", 1))
            if eval_every < 1:
                raise ValueError("eval_every_epochs must be positive")
            if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch + 1 == args.epoch_num:
                smaller_delta_avg, aj, oa = evaluate(
                    args, val_dataloader, model, epoch, writer=writer,
                    global_step=loss_tracker.iteration,
                    log_to_wandb=logging_backend == "wandb",
                )

                # === Update Best Models ===
                if aj > best_models["aj"][1]:
                    best_models["aj"] = [epoch, aj]
                    best_aj = True

                if oa > best_models["oa"][1]:
                    best_models["oa"] = [epoch, oa]
                    best_oa = True

                if smaller_delta_avg > best_models["delta_avg"][1]:
                    best_models["delta_avg"] = [epoch, smaller_delta_avg]
                    best_delta_avg = True
            # === === ===
            
            # # === Save Model ===
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "best_aj": best_models["aj"],
                "best_oa": best_models["oa"],
                "best_delta_avg": best_models["delta_avg"],
                "iteration": loss_tracker.iteration,
            }
        
            save_dict["best_metric"] = best_metric
            save_dict["best_score"] = best_models[best_metric][1]
            improved = {"aj": best_aj, "oa": best_oa, "delta_avg": best_delta_avg}[best_metric]
            save_training_checkpoints(
                save_dict, args.model_save_path, epoch + 1, improved, save_every,
            )
            if improved:
                print(f"Best {best_metric} checkpoint: {args.model_save_path}/best.pt")
            if save_every and (epoch + 1) % save_every == 0:
                print(f"Periodic checkpoint: {args.model_save_path}/epoch_{epoch + 1:04d}.pt")

        dist.barrier()

        print(f"=== === === === === ===")
        print()

    # print best results
    if args.rank == 0:
        print("Best Results")
        print(f"Best AJ: {best_models['aj'][1]:.3f} at epoch {best_models['aj'][0]}")
        print(f"Best OA: {best_models['oa'][1]:.3f} at epoch {best_models['oa'][0]}")
        print(f"Best Smaller Delta Avg: {best_models['delta_avg'][1]:.3f} at epoch {best_models['delta_avg'][0]}")
    
    if writer is not None:
        writer.close()
    elif args.rank == 0:
        import wandb
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    dist.destroy_process_group()


if __name__ == '__main__':
    # args = get_args()

    parser = argparse.ArgumentParser(description="Training for Track-On2")

    parser.add_argument("--config_path", type=str, required=True, help='Path to the config file')
    parser.add_argument("--resume", type=str, help="Training checkpoint to resume in a new run directory")
    parser.add_argument("--reset-best", action="store_true", help="Reset best metrics when changing validation data")
    args_cmd = parser.parse_args()

    args = load_args_from_yaml(args_cmd.config_path)
    if args_cmd.reset_best and not args_cmd.resume:
        parser.error("--reset-best requires --resume")
    if args_cmd.resume:
        checkpoint_path = Path(args_cmd.resume).resolve()
        if not checkpoint_path.is_file():
            parser.error(f"resume checkpoint does not exist: {checkpoint_path}")
        args.checkpoint_path = str(checkpoint_path)
    args.reset_best = args_cmd.reset_best
    
    args.gpus = torch.cuda.device_count()
    try:
        main_worker(args, args_cmd.config_path)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
