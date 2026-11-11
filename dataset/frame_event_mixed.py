"""Real/synthetic Frame-Event samples and distributed source sampling."""

import math

import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler

from dataset.event_kubric_rgb import WeightedDistributedSampler


class MixedFrameEventDataset(Dataset):
    def __init__(self, real_dataset, synthetic_dataset, source_names=()):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.source_names = tuple(source_names)

    def __len__(self):
        return len(self.real_dataset) + len(self.synthetic_dataset)

    def __getitem__(self, index):
        synthetic = index >= len(self.real_dataset)
        if synthetic:
            sample = self.synthetic_dataset[index - len(self.real_dataset)]
            times = sample["visibility"].long().argmax(dim=0)
            xy = sample["tracks"][times, torch.arange(len(times))]
            sample["queries"] = torch.cat((times[:, None], xy), dim=-1).float()
            sample["label_valid"] = torch.ones_like(sample["visibility"], dtype=torch.bool)
            if self.source_names:
                sample["source_id"] = torch.tensor(0)
        else:
            sample = self.real_dataset[index]
            # Native timestamps are ERF metadata, unused by training.
            sample.pop("timestamps_us", None)
        sample["is_synthetic"] = torch.tensor(synthetic)
        return sample


class WeightedSourceSampler(Sampler):
    """Fixed-budget, homogeneous source batches with the same plan on every rank.

    Source quotas use largest-remainder rounding. Samples are drawn uniformly
    with replacement within real sources; synthetic groups retain their weights.
    """

    def __init__(self, ranges, weights, batches_per_epoch, batch_size, rank, world_size, seed,
                 synthetic_group_ids=None, synthetic_group_weights=None):
        self.names = tuple(ranges)
        if set(weights) != set(self.names):
            raise ValueError("sampling_weights must name every active source exactly once")
        probabilities = torch.tensor([weights[name] for name in self.names], dtype=torch.double)
        if (not torch.isfinite(probabilities).all() or (probabilities <= 0).any()
                or batches_per_epoch < 1 or batch_size < 1):
            raise ValueError("Source weights and batch budget must be positive")
        quotas = batches_per_epoch * probabilities / probabilities.sum()
        self.counts = quotas.floor().long()
        remainder = batches_per_epoch - int(self.counts.sum())
        self.counts[torch.argsort(quotas - self.counts, descending=True, stable=True)[:remainder]] += 1
        self.ranges = ranges
        self.batch_size, self.batches_per_epoch = batch_size, batches_per_epoch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0
        self.synthetic_sampler = None
        if "synthetic" in self.names:
            count = int(self.counts[self.names.index("synthetic")])
            if count:
                self.synthetic_sampler = WeightedDistributedSampler(
                    synthetic_group_ids, synthetic_group_weights, count * batch_size * world_size,
                    rank, world_size, seed,
                )

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.synthetic_sampler is not None:
            self.synthetic_sampler.set_epoch(epoch)

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        plan = torch.repeat_interleave(torch.arange(len(self.names)), self.counts)
        plan = plan[torch.randperm(len(plan), generator=generator)].tolist()
        samples = []
        for name, count in zip(self.names, self.counts.tolist()):
            offset, length = self.ranges[name]
            if length < 1:
                raise ValueError(f"Empty training source: {name}")
            if name == "synthetic" and count:
                indices = torch.tensor(list(self.synthetic_sampler), dtype=torch.long)
            else:
                indices = torch.randint(length, (count, self.world_size, self.batch_size), generator=generator)
                indices = indices[:, self.rank].reshape(-1)
            samples.append((indices + offset).reshape(count, self.batch_size).tolist())
        positions = [0] * len(self.names)
        for source in plan:
            yield from samples[source][positions[source]]
            positions[source] += 1


class MixedFrameEventSampler(Sampler):
    """Alternate homogeneous batches; visit each real clip once, with DDP padding.

    All ranks use the same source at each microbatch. Synthetic clips retain
    the source weights used during pretraining. Each epoch contains equal
    numbers of real and synthetic samples.
    """

    def __init__(self, dataset, group_ids, group_weights, batch_size, rank, world_size, seed):
        self.real_sampler = DistributedSampler(
            dataset.real_dataset, num_replicas=world_size, rank=rank, seed=seed,
        )
        self.real_count = math.ceil(len(self.real_sampler) / batch_size) * batch_size
        self.synthetic_sampler = WeightedDistributedSampler(
            group_ids, group_weights, self.real_count * world_size, rank, world_size, seed,
        )
        self.synthetic_offset = len(dataset.real_dataset)
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.real_sampler.set_epoch(epoch)
        self.synthetic_sampler.set_epoch(epoch)

    def __len__(self):
        return 2 * self.real_count

    def __iter__(self):
        real = list(self.real_sampler)
        real = (real * math.ceil(self.real_count / len(real)))[:self.real_count]
        synthetic = [index + self.synthetic_offset for index in self.synthetic_sampler]
        for start in range(0, self.real_count, self.batch_size):
            yield from synthetic[start:start + self.batch_size]
            yield from real[start:start + self.batch_size]
