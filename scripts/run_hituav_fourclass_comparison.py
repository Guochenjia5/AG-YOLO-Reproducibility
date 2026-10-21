#!/usr/bin/env python3
"""Orchestrate the corrected six-model HIT-UAV comparison sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKER = HERE / "train_hituav_fourclass_model.py"
EVALUATOR = HERE / "evaluate_hituav_ignore.py"
DATASET = Path(r"D:\gcj\hit-uav-ignore\dataset_ignore.yaml")
DEFAULT_OUTPUT = Path(r"D:\gcj\hituav-revision-4class")

MODELS = [
    {
        "name": "yolov6n",
        "repo": Path(r"D:\gcj\yolo11"),
        "config": Path(r"D:\gcj\yolo11\ultralytics\cfg\models\v6\yolov6.yaml"),
    },
    {
        "name": "yolov8n",
        "repo": Path(r"D:\gcj\yolo11"),
        "config": Path(r"D:\gcj\yolo11\ultralytics\cfg\models\v8\yolov8.yaml"),
    },
    {
        "name": "yolov9t",
        "repo": Path(r"D:\gcj\yolo11"),
        "config": Path(r"D:\gcj\yolo11\ultralytics\cfg\models\v9\yolov9t.yaml"),
    },
    {
        "name": "yolov11n",
        "repo": Path(r"D:\gcj\ultralytics-yolo11-main"),
        "config": Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolo11.yaml"),
    },
    {
        "name": "yolov12n",
        "repo": Path(r"D:\gcj\ultralytics-yolo11-main"),
        "config": Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\12\yolo12.yaml"),
    },
    {
        "name": "ag_yolo",
        "repo": Path(r"D:\gcj\ultralytics-yolo11-main"),
        "config": Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolo11-MAN-CTA-C2CGA.yaml"),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seed0", "core-extra", "all-three", "smoke"], default="seed0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    for required in (WORKER, EVALUATOR, dataset):
        if not required.exists():
            raise FileNotFoundError(required)
    for spec in MODELS:
        if not spec["repo"].exists() or not spec["config"].exists():
            raise FileNotFoundError(spec)

    if args.stage == "core-extra":
        selected = [m for m in MODELS if m["name"] in {"yolov11n", "ag_yolo"}]
        seeds, epochs, batch, imgsz, workers, fraction = [1, 2], 300, 32, 640, 4, 1.0
    elif args.stage == "all-three":
        selected = MODELS
        seeds, epochs, batch, imgsz, workers, fraction = [0, 1, 2], 300, 32, 640, 4, 1.0
    elif args.stage == "smoke":
        selected = MODELS
        seeds, epochs, batch, imgsz, workers, fraction = [0], 1, 2, 320, 0, 0.02
    else:
        selected = MODELS
        seeds, epochs, batch, imgsz, workers, fraction = [0], 300, 32, 640, 4, 1.0

    output = args.output.resolve()
    plan = {
        "stage": args.stage,
        "output": str(output),
        "dataset": str(dataset),
        "models": [m["name"] for m in selected],
        "seeds": seeds,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "fraction": fraction,
        "commands": [],
    }
    commands = []
    for spec in selected:
        command = [
            sys.executable,
            str(WORKER),
            "--repo",
            str(spec["repo"]),
            "--model-config",
            str(spec["config"]),
            "--model-name",
            spec["name"],
            "--dataset",
            str(dataset),
            "--evaluator",
            str(EVALUATOR),
            "--output",
            str(output),
            "--seeds",
            *[str(seed) for seed in seeds],
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
            "--workers",
            str(workers),
            "--device",
            args.device,
            "--fraction",
            str(fraction),
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        commands.append(command)
        plan["commands"].append(command)

    output.mkdir(parents=True, exist_ok=True)
    (output / f"plan_{args.stage}.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
