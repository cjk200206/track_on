"""Real-only Frame-Event fine-tuning using offline ERF pseudo labels.

Use --check-data for alignment previews and --max-steps for a short training run.
Use --gpus 0 1 or torchrun for DDP. --resume restores this entry's training state.

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
from torch.utils.data import DataLoader, DistributedSampler

from dataset.frame_event_erf import FrameEventERF, build_erf_index
from model.loss import Loss_Function
from utils.train_utils import get_scheduler, load_args_from_yaml, load_pretrained_weights


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


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)


def make_loaders(args, rank, world_size, run_dir, sequences=None, resume=False):
    payload = [None]
    if rank == 0:
        names = (sorted(sequences) if sequences else
                 sorted(path.name for path in Path(args.erf_root).iterdir() if path.is_dir()))
        if len(names) != len(set(names)) or not names:
            raise ValueError("ERF sequence list must be non-empty and unique")
        val_count = max(1, round(len(names) * args.val_fraction)) if args.val_fraction else 0
        if val_count >= len(names):
            raise ValueError("Sequence split leaves no ERF training sequences")
        permutation = np.random.default_rng(args.split_seed).permutation(names)
        val_names = set(permutation[:val_count].tolist())
        records, summary = build_erf_index(args, names)
        for item in summary:
            item["split"] = "val" if item["sequence"] in val_names else "train"
        split = {"train": [name for name in names if name not in val_names], "val": sorted(val_names)}
        index = {"split": split, "summary": summary, "clips": records}
        index_path = run_dir / "data_index.json"
        if resume:
            if json.loads(index_path.read_text()) != index:
                raise ValueError("ERF data index changed since the saved run")
        else:
            write_json(index_path, index)
        print(f"ERF: {len(records)} / {sum(item['candidates'] for item in summary)} clips retained; "
              f"sequences train={len(split['train'])}, val={len(split['val'])}", flush=True)
        payload[0] = (records, split)
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    records, split = payload[0]
    train_set = FrameEventERF(args, [r for r in records if r["sequence"] in split["train"]], training=True)
    val_set = (FrameEventERF(args, [r for r in records if r["sequence"] in split["val"]], training=False)
               if split["val"] else None)

    def loader(dataset, training):
        workers = args.num_workers if training else args.val_num_workers
        sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed)
                   if training else range(rank, len(dataset), world_size))
        return DataLoader(
            dataset, batch_size=args.bs, sampler=sampler, num_workers=workers,
            pin_memory=True, worker_init_fn=seed_worker, persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(args.seed + rank),
            **({"prefetch_factor": 1} if workers else {}),
        )
    return loader(train_set, True), loader(val_set, False) if val_set is not None else None, split


def inspect_data(loader, run_dir):
    from PIL import Image, ImageDraw

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


@torch.no_grad()
def evaluate(model, loader, args, device):
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
        after = torch.arange(args.T, device=device)[None, :, None] > batch["queries"][:, None, :, 0]
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


def train(model, loader, val_loader, args, cli, device, rank, run_dir, split):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(args, optimizer, loader, warmup=False)
    scaler = torch.GradScaler(enabled=args.amp)
    loss_fn = RealFineTuneLoss(args)
    loss_names = ("patch1", "patch2", "offset", "ranking", "topk_uncertainty")
    start_epoch, start_batch, steps, best_error = 0, 0, 0, float("inf")
    if cli.resume:
        state = torch.load(cli.resume, map_location="cpu", weights_only=False)
        ignored = {"output_dir", "num_workers", "val_num_workers"}
        if ({k: v for k, v in state["config"].items() if k not in ignored}
                != {k: v for k, v in vars(args).items() if k not in ignored} or state["split"] != split):
            raise ValueError("Resume configuration or sequence split differs from the saved run")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch, steps = state["epoch"], state["batch_in_epoch"], state["step"]
        best_error = state["best_pseudo_error_px"]
        del state
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        print(f"Train clips={len(loader.dataset)}, val clips={len(val_loader.dataset) if val_loader else 0}; "
              f"effective batch={args.bs * world_size * args.grad_accum_steps}; "
              f"updates/epoch={math.ceil(len(loader) / args.grad_accum_steps)}", flush=True)
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
            batch = move_batch(raw_batch, device)
            window_start = batch_index // args.grad_accum_steps * args.grad_accum_steps
            window_size = min(args.grad_accum_steps, len(loader) - window_start)
            update_now = batch_index - window_start + 1 == window_size
            sync = model.no_sync() if isinstance(model, DistributedDataParallel) and not update_now else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    out = model(batch["video"], batch["queries"].clone(), events=batch["events"],
                                img_ifnew=batch["img_ifnew"], p_mask=args.memory_mask_probability)
                    parts = loss_fn(out, batch)
                    loss = sum(parts.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite FT loss at epoch={epoch}, batch={batch_index}")
                scaler.scale(loss / window_size).backward()
            values = torch.stack([loss.detach().float(), *(part.detach().float() for part in parts.values())])
            window = values if window is None else window + values
            window_batches += 1
            next_batch = batch_index + 1
            del out, parts, loss, batch
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
            elif rank == 0:
                print(f"AMP skipped update at epoch={epoch + 1}, batch={next_batch}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if dist.is_initialized():
                dist.all_reduce(window)
            means = (window / (window_batches * world_size)).tolist()
            if rank == 0 and (steps % args.log_every == 0 or steps == 1 or next_batch == len(loader)):
                record = {"epoch": epoch + 1, "step": steps, "loss": means[0],
                          "components": dict(zip(loss_names, means[1:])), "lr": optimizer.param_groups[0]["lr"],
                          "optimizer_updated": updated}
                with (run_dir / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"epoch={epoch + 1} step={steps} loss={means[0]:.5f} lr={record['lr']:.8g}", flush=True)
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
                with (run_dir / "eval_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, "step": steps, **score}) + "\n")
                print(f"Pseudo-label validation (not GT): {score}", flush=True)
            state = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch + int(complete), "batch_in_epoch": 0 if complete else next_batch,
                     "step": steps, "best_pseudo_error_px": best_error, "config": vars(args), "split": split}
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
    parser.add_argument("--config", help="Defaults to config/train_frame_event_real_ft.yaml; resume reuses saved settings")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--sequences", nargs="+", help="Restrict data for a small check; saved in the run configuration")
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
    args.sequences = (sorted(cli.sequences) if cli.sequences else
                      saved_config.get("sequences") if saved_config is not None else None)
    if saved_config is not None:
        ignored = {"output_dir", "num_workers", "val_num_workers"}
        if ({k: v for k, v in saved_config.items() if k not in ignored}
                != {k: v for k, v in vars(args).items() if k not in ignored}):
            raise ValueError("Resume configuration differs from the saved run")
    if (args.T < 2 or args.N < 1 or args.bs < 1 or args.grad_accum_steps < 1 or args.epoch_num < 1
            or args.event_native_stride < 1 or args.rgb_native_interval < 1
            or args.rgb_native_interval % args.event_native_stride
            or not 0 < args.min_visible_fraction <= 1 or not 0 <= args.val_fraction < 1
            or min(args.num_workers, args.val_num_workers) < 0 or args.log_every < 1
            or not 0 <= args.memory_mask_probability < 1):
        raise ValueError("Invalid ERF training, frequency, split or trajectory screening configuration")
    if args.event_fusion_type != "recurrent" or args.event_channels != 10:
        raise ValueError("This entry expects the recurrent 10-channel FE model")
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
        train(model, train_loader, val_loader, args, cli, device, rank, run_dir, split)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
