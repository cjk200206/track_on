"""TensorBoard logging for RGB Track-On2 training."""

from pathlib import Path
import math
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


def init_tensorboard(args, iteration=0):
    if args.rank != 0:
        return None
    log_dir = Path(args.model_save_path) / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir), purge_step=iteration if iteration else None)
    print(f"TensorBoard log directory: {log_dir.resolve()}")
    return writer


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
