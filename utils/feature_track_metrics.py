"""TAPFormer EDS/EC feature age metric for sparse tracks."""

import numpy as np


def feature_age_at_threshold(pred_tracks, pred_times, gt_rows, max_distance):
    """Return mean error, FA and EFA using TAPFormer's compareTracks rules.

    Predictions are (T, N, 2). Query index N follows sorted GT track IDs.
    Visibility is deliberately unused, matching TAPFormer's EDS/EC evaluation.
    """
    gt_ids = np.unique(gt_rows[:, 0])
    if pred_tracks.shape != (len(pred_times), len(gt_ids), 2):
        raise ValueError("Prediction shape does not match frame times and GT IDs")
    accepted = []
    last_gt_times = None
    for i, track_id in enumerate(gt_ids):
        gt = gt_rows[gt_rows[:, 0] == track_id, 1:4].copy()
        gt_times = gt[:, 0] - gt[0, 0]
        last_gt_times = gt_times
        if len(gt_times) < 2 or gt_times[-1] <= 0:
            raise ValueError(f"GT track {track_id} needs two increasing timestamps")
        pred_t = np.concatenate(([0.0], pred_times - gt[0, 0]))
        pred_xy = np.concatenate((gt[:1, 1:3], pred_tracks[:, i]), axis=0)
        errors = np.hypot(
            np.interp(gt_times, pred_t, pred_xy[:, 0]) - gt[:, 1],
            np.interp(gt_times, pred_t, pred_xy[:, 1]) - gt[:, 2],
        )
        errors[0] = 0.0
        failures = np.flatnonzero(errors > max_distance)
        if failures.size:
            first_failure = int(failures[0])
            if first_failure <= 1:
                continue
            age = gt_times[first_failure] / gt_times[-1]
            mean_error = float(np.mean(errors[:first_failure]))
        else:
            age = 1.0
            mean_error = float(np.mean(errors))
        accepted.append((mean_error, age))

    if not accepted:
        # TAPFormer's fallback uses the final GT track's first time step.
        fallback_age = float(last_gt_times[1] / last_gt_times[-1])
        return {"mean_error": 0.0, "fa": fallback_age, "efa": 0.0}
    mean_error, fa = np.mean(accepted, axis=0)
    return {"mean_error": float(mean_error), "fa": float(fa),
            "efa": float(len(accepted) / len(gt_ids) * fa)}


def average_feature_age(pred_tracks, pred_times, gt_rows, thresholds=range(1, 32)):
    """Average the per-threshold results over thresholds 1 through 31 pixels."""
    rows = [feature_age_at_threshold(pred_tracks, pred_times, gt_rows, distance)
            for distance in thresholds]
    if not rows:
        raise ValueError("At least one feature threshold is required")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
