"""Single-GPU feasibility experiment for paired Kubric frames and time surfaces.

Run --check-data first, then --overfit-one-batch, then the default training run.
Use --eval-checkpoint PATH to repeat the three validation modes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset.frame_event_kubric import FrameEventKubric
from model.loss import Loss_Function
from model.trackon import Track_On2
from utils.coord_utils import get_queries
from utils.eval_utils import compute_tapvid_metrics
from utils.train_utils import load_args_from_yaml, load_pretrained_weights


def first_visible_queries(tracks, visibility):
    """Return (t, x, y) queries for tracks visible at least once."""
    batch, _, points, _ = tracks.shape
    first = visibility.long().argmax(dim=1)
    batch_index = torch.arange(batch, device=tracks.device)[:, None]
    point_index = torch.arange(points, device=tracks.device)[None, :]
    xy = tracks[batch_index, first, point_index]
    return torch.cat((first.unsqueeze(-1), xy), dim=-1)


def make_loader(args, training):
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
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=training,
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
    return model


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


def train(model, loader, args, device, steps, overfit_one_batch):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.wd,
    )
    loss_fn = Loss_Function(args)
    iterator = iter(loader)
    fixed_batch = move_batch(next(iterator), device) if overfit_one_batch else None
    fixed_queries = (
        first_visible_queries(fixed_batch["tracks"], fixed_batch["visibility"])
        if fixed_batch is not None else None
    )
    first_loss = None

    for step in range(1, steps + 1):
        if fixed_batch is None:
            try:
                batch = move_batch(next(iterator), device)
            except StopIteration:
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
        if first_loss is None:
            first_loss = loss.item()
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"step={step}/{steps} loss={loss.item():.5f} event_grad_norm={gradient_norm.item():.5f}", flush=True)
    print(f"loss: first={first_loss:.5f}, last={loss.item():.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_frame_event.yaml")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-checkpoint", type=str)
    cli = parser.parse_args()
    args = load_args_from_yaml(cli.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cli.check_data:
        inspect_data(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")

    device = torch.device("cuda")
    model = build_model(args, device)
    val_loader = make_loader(args, training=False)
    if cli.eval_checkpoint:
        checkpoint = torch.load(cli.eval_checkpoint, map_location=device, weights_only=False)
        state = checkpoint["model"]
        # Accept checkpoints saved before event fusion moved into its own module.
        state = {
            key.replace("backbone.event_encoder.layers.", "backbone.event_fusion.encoder.")
               .replace("backbone.event_gates", "backbone.event_fusion.gates"): value
            for key, value in state.items()
        }
        model.load_state_dict(state)
    else:
        train_loader = make_loader(args, training=True)
        steps = cli.max_steps if cli.max_steps is not None else (
            args.overfit_steps if cli.overfit_one_batch else args.max_steps
        )
        if steps < 1:
            raise ValueError("Training steps must be positive")
        train(model, train_loader, args, device, steps, cli.overfit_one_batch)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("overfit.pt" if cli.overfit_one_batch else "frame_event_last.pt")
        torch.save({"model": model.state_dict(), "step": steps}, path)
        print(f"checkpoint: {path}")

    if not cli.overfit_one_batch:
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
