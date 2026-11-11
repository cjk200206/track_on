"""Optional short backend adaptation followed by joint Frame–Event fine-tuning."""

import torch


def memory_parameter(name):
    return name.startswith(("memory_reliability.", "rgb_memory_reader."))


def build_ft_optimizer(model, args):
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    memory_lr = float(getattr(args, "memory_lr", args.lr))
    if memory_lr == args.lr:
        return torch.optim.AdamW([parameter for _, parameter in parameters],
                                 lr=args.lr, weight_decay=args.wd)
    groups = []
    for is_memory, lr in ((False, args.lr), (True, memory_lr)):
        selected = [parameter for name, parameter in parameters if memory_parameter(name) == is_memory]
        if selected:
            groups.append({"params": selected, "lr": lr})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)


class MemoryTrainingSchedule:
    def __init__(self, model, args):
        self.steps = int(getattr(args, "adaptation_steps", 0))
        self.length = int(getattr(args, "adaptation_T", args.T))
        self.full_length = args.T
        if self.steps < 0 or not 2 <= self.length <= args.T:
            raise ValueError("Require adaptation_steps >= 0 and 2 <= adaptation_T <= T")
        self.frontend = (model.backbone, model.fpn)
        # Capture the original policy before adaptation; the frozen DINO stays frozen.
        self.parameters = [(parameter, parameter.requires_grad)
                           for module in self.frontend for parameter in module.parameters()]

    def phase(self, updates):
        return "adaptation" if updates < self.steps else "joint"

    def apply(self, updates):
        adapting = self.phase(updates) == "adaptation"
        for parameter, trainable in self.parameters:
            parameter.requires_grad_(trainable and not adapting)
        for module in self.frontend:
            module.train(not adapting)
        return self.length if adapting else self.full_length


def training_clip(batch, length):
    """Load/index one full-length dataset; shorten only adaptation forwards."""
    # Synthetic queries can start late. Keep a query and a following step in each
    # sample instead of producing a loss with no initialized points or gradients.
    first_query = batch["queries"][..., 0].amin(dim=1).max().item()
    length = min(batch["video"].shape[1], max(length, int(first_query) + 2))
    temporal = {"video", "events", "tracks", "visibility", "label_valid",
                "img_ifnew", "frame_indices", "timestamps_us"}
    return {key: value[:, :length] if key in temporal else value for key, value in batch.items()}
