"""Flat CSV export shared by TAPFormer benchmark entry points."""

import csv
from statistics import mean


TAPVID_CSV_COLUMNS = {
    name: f"{name}_pct" for name in (
        "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy",
        "pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8", "pts_within_16",
        "jaccard_1", "jaccard_2", "jaccard_4", "jaccard_8", "jaccard_16",
    )
}
FEATURE_CSV_COLUMNS = {"mean_error": "mean_error_px", "fa": "fa", "efa": "efa"}


def write_summary_csv(results, path, metric_columns, metric_scale=1.0, checkpoint=""):
    """Write sequence rows plus an unweighted dataset mean row.

    `metric_columns` maps JSON metric keys to explicit CSV column names.
    """
    fieldnames = [
        "model", "checkpoint", "dataset", "row_type", "sequence", "sequence_count", "frames", "points",
        "seconds_per_frame", "video_path", "preview_path", *metric_columns.values(),
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, data in results.items():
            sequences = data["sequences"]
            for item in sequences:
                row = {
                    "model": "trackon2",
                    "checkpoint": checkpoint,
                    "dataset": dataset,
                    "row_type": "sequence",
                    "sequence": item["sequence"],
                    "frames": item["frames"],
                    "points": item["points"],
                    "seconds_per_frame": round(item["seconds_per_frame"], 6),
                    "video_path": item.get("video_path", ""),
                    "preview_path": item.get("preview_path", ""),
                }
                row.update({column: round(item["metrics"][key] * metric_scale, 6)
                            for key, column in metric_columns.items()})
                writer.writerow(row)
            average = {
                "model": "trackon2",
                "checkpoint": checkpoint,
                "dataset": dataset,
                "row_type": "dataset_mean",
                "sequence": "ALL",
                "sequence_count": len(sequences),
                "seconds_per_frame": round(mean(item["seconds_per_frame"] for item in sequences), 6),
            }
            average.update({column: round(data["mean_metrics"][key] * metric_scale, 6)
                            for key, column in metric_columns.items()})
            writer.writerow(average)
