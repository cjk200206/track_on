"""Compatibility entry for existing ERF fine-tuning configurations."""

from dataset.frame_event_real import (
    FrameEventReal, build_real_index, real_source_args,
    sample_photometric_augmentation, apply_photometric_augmentation,
    matching_rows, resize_labels,
)


def erf_source(args):
    return real_source_args(args, {
        "name": "erf", "dataset": "erf", "root": args.erf_root,
        "event_root": args.erf_event_root, "label_root": args.erf_label_root,
    })


def build_erf_index(args, sequences):
    return build_real_index(erf_source(args), sequences)


class FrameEventERF(FrameEventReal):
    def __init__(self, args, records, training):
        super().__init__(erf_source(args), records, training)
