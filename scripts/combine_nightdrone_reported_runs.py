"""Build the manuscript-reported Night-Drone Vehicle repeated-run tables.

Seed 0 is retained from the archived manuscript comparison (Table 7).
Seeds 1 and 2 are read from the additional-run CSV. The newly rerun seed 0
is deliberately not used in these reported statistics.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


METRICS = ("precision", "recall", "mAP50", "mAP50_95")
ARCHIVED_SEED0 = {
    "yolov11n": {
        "precision": 0.801,
        "recall": 0.735,
        "mAP50": 0.800,
        "mAP50_95": 0.607,
    },
    "ag_yolo": {
        "precision": 0.799,
        "recall": 0.766,
        "mAP50": 0.813,
        "mAP50_95": 0.625,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.additional_results.open(encoding="utf-8-sig", newline="") as handle:
        additional = list(csv.DictReader(handle))

    reported = []
    for model in ("yolov11n", "ag_yolo"):
        reported.append(
            {
                "model": model,
                "seed": 0,
                **ARCHIVED_SEED0[model],
                "source": "archived manuscript Table 7",
            }
        )
        for seed in (1, 2):
            row = next(
                row
                for row in additional
                if row["model"] == model and int(row["seed"]) == seed
            )
            reported.append(
                {
                    "model": model,
                    "seed": seed,
                    **{metric: float(row[metric]) for metric in METRICS},
                    "source": "additional run",
                }
            )

    summary = []
    for model in ("yolov11n", "ag_yolo"):
        model_rows = [row for row in reported if row["model"] == model]
        result = {"model": model, "n": len(model_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in model_rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sample_sd"] = statistics.stdev(values)
        summary.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reported_path = args.output_dir / "nightdrone_reported_runs.csv"
    summary_path = args.output_dir / "nightdrone_reported_summary.csv"

    with reported_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "seed", *METRICS, "source"),
        )
        writer.writeheader()
        writer.writerows(reported)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(reported_path)
    print(summary_path)


if __name__ == "__main__":
    main()
