"""TensorBoard logging shared by RGB and Frame-Event training."""

import json
from pathlib import Path
import math
from types import SimpleNamespace
import warnings

from torch.utils.tensorboard import SummaryWriter


LOSS_NAMES = (
    "point_loss",
    "point_rerank_loss",
    "visibility_loss",
    "offset_loss",
    "uncertainty_loss",
    "topk_uncertainty_loss",
    "topk_rank_loss",
)


def init_tensorboard(args, iteration=0, *, flush_secs=120):
    if args.rank != 0:
        return None
    log_dir = Path(args.model_save_path) / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir), purge_step=iteration if iteration else None,
                           flush_secs=flush_secs)
    print(f"TensorBoard log directory: {log_dir.resolve()}")
    return writer


class FrameEventTensorBoard:
    """Use optimizer-attempt steps and the same loss tags as RGB Track-On2."""

    loss_names = dict(zip(
        ("patch1", "patch2", "visibility", "offset", "uncertainty", "topk_uncertainty", "ranking"),
        LOSS_NAMES,
    ))

    def __init__(self, args, run_dir, rank=0):
        self.args = args
        self.context = SimpleNamespace(rank=rank, model_save_path=run_dir)
        self.writer = None
        self.step = 0

    def start(self, step=0):
        self.step = step
        # The checkpoint already includes step; discard only events after it.
        self.writer = init_tensorboard(self.context, step + 1 if step else 0, flush_secs=10)
        if self.writer is not None:
            config = json.dumps(vars(self.args), indent=2, default=str)
            self.writer.add_text("Configuration", f"```json\n{config}\n```", step)
            self.writer.flush()

    def _write(self, values, step):
        self.step = max(self.step, step)
        if self.writer is not None:
            for name, value in values.items():
                if value is not None:
                    self.writer.add_scalar(name, float(value), step)

    def training(self, step, loss=None, components=None, *, lr=None, epoch=None,
                 grad_norm=None, optimizer_updated=None, amp_scale=None, microbatches=None, metrics=None):
        values = {
            "Training/batch_loss": loss, "Training/lr": lr, "Training/epoch": epoch,
            "Training/grad_norm_before_clip": grad_norm, "Training/amp_scale": amp_scale,
            "Training/accumulated_microbatches": microbatches,
            "Training/optimizer_updated": optimizer_updated,
            "Training/skipped_batch": None if optimizer_updated is None else int(not optimizer_updated),
        }
        values.update({f"Training/{self.loss_names[name]}": value
                       for name, value in (components or {}).items()})
        values.update({f"Training/{name}": value for name, value in (metrics or {}).items()})
        self._write(values, step)

    def evaluation(self, metrics, step=None, *, mode="fused", epoch=None):
        prefix = "Evaluation" if mode == "fused" else f"Evaluation/{mode}"
        # Match normal Track-On's percentage units; pixel errors/fractions stay raw.
        values = {f"{prefix}/{name}": value * 100 if name in ("AJ", "OA", "delta_avg") else value
                  for name, value in metrics.items()}
        values[f"{prefix}/epoch"] = epoch
        self._write(values, self.step if step is None else step)
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class TensorBoardLossTracker:
    def __init__(self, rank, writer, iteration=0):
        self.rank = rank
        self.writer = writer
        self.iteration = iteration
        self.losses = [0.0] * len(LOSS_NAMES)
        self.temporal_count = 0

    def update(self, *losses):
        if self.rank != 0:
            return
        if len(losses) != len(LOSS_NAMES):
            raise ValueError(f"Expected {len(LOSS_NAMES)} loss terms, got {len(losses)}")
        for index, loss in enumerate(losses):
            value = float(loss.detach())
            if not math.isfinite(value):
                warnings.warn(f"Non-finite {LOSS_NAMES[index]} at step {self.iteration}; recording zero", stacklevel=1)
                value = 0.0
            self.losses[index] += value
        self.temporal_count += 1

    def log(self, learning_rate=None, skipped_microbatches=0, microbatches=1,
            optimizer_updated=True, grad_norm=None):
        if self.rank != 0:
            return
        values = [value / self.temporal_count for value in self.losses]
        for name, value in zip(LOSS_NAMES, values):
            self.writer.add_scalar(f"Training/{name}", value, self.iteration)
        self.writer.add_scalar("Training/batch_loss", sum(values), self.iteration)
        self.writer.add_scalar("Training/skipped_batch", int(skipped_microbatches > 0), self.iteration)
        self.writer.add_scalar("Training/skipped_microbatches", skipped_microbatches, self.iteration)
        self.writer.add_scalar("Training/accumulated_microbatches", microbatches, self.iteration)
        self.writer.add_scalar("Training/optimizer_updated", int(optimizer_updated), self.iteration)
        if grad_norm is not None:
            self.writer.add_scalar("Training/grad_norm_before_clip", float(grad_norm), self.iteration)
        if learning_rate is not None:
            self.writer.add_scalar("Training/lr", learning_rate, self.iteration)
        self.iteration += 1
        self.losses = [0.0] * len(LOSS_NAMES)
        self.temporal_count = 0

    def skip(self, learning_rate=None, count=1):
        if self.rank != 0:
            return
        self.writer.add_scalar("Training/skipped_batch", 1, self.iteration)
        self.writer.add_scalar("Training/skipped_microbatches", count, self.iteration)
        self.writer.add_scalar("Training/accumulated_microbatches", 0, self.iteration)
        self.writer.add_scalar("Training/optimizer_updated", 0, self.iteration)
        if learning_rate is not None:
            self.writer.add_scalar("Training/lr", learning_rate, self.iteration)
        self.iteration += 1
        self.losses = [0.0] * len(LOSS_NAMES)
        self.temporal_count = 0
