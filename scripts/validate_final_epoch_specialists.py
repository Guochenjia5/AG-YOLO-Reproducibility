"""Independently validate final-epoch checkpoints used in Table 7.

This script does not train or modify checkpoints. It writes each validation run
to ``validation_last`` and saves a one-row ``final_epoch_results.csv`` beside
the original experiment outputs.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

# The checkpoints contain classes from the locally modified Ultralytics tree.
# Put that checkout ahead of the environment-installed package before import.
LOCAL_ULTRALYTICS = Path(r"D:\gcj\ultralytics-yolo11-main")
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


DATA = Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\datasets\An guang.yaml")
RUNS = (
    {
        "label": "YOLO-CIR independent reimplementation",
        "root": Path(r"D:\gcj\night-drone-yolo-cir-reimpl-b8-fp32"),
        "run_name": "yolo_cir_reimpl_seed0",
        "batch": 8,
    },
    {
        "label": "YOLO-HVS independent reimplementation",
        "root": Path(r"D:\gcj\night-drone-yolo-hvs-reimpl-b32-fp32"),
        "run_name": "yolo_hvs_reimpl_seed0",
        "batch": 32,
    },
)


def metric_value(metrics, dotted_name: str) -> float:
    value = metrics
    for part in dotted_name.split("."):
        value = getattr(value, part)
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric {dotted_name}: {value}")
    return value


def main() -> None:
    for run in RUNS:
        root = run["root"]
        checkpoint = root / run["run_name"] / "weights" / "last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        print(f"Validating {run['label']} from {checkpoint}", flush=True)
        evaluated = YOLO(str(checkpoint)).val(
            data=str(DATA),
            split="val",
            imgsz=640,
            batch=run["batch"],
            workers=4,
            device=0,
            half=False,
            plots=False,
            project=str(root / "validation_last"),
            name=run["run_name"],
            exist_ok=True,
        )
        row = {
            "model": run["label"],
            "seed": 0,
            "precision": metric_value(evaluated, "box.mp"),
            "recall": metric_value(evaluated, "box.mr"),
            "mAP50": metric_value(evaluated, "box.map50"),
            "mAP50_95": metric_value(evaluated, "box.map"),
            "checkpoint": str(checkpoint),
            "checkpoint_policy": "final_epoch",
        }
        output_csv = root / "final_epoch_results.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
