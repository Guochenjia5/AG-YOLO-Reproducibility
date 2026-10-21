#!/usr/bin/env python3
"""Run paired YOLOv11n and AG-YOLO experiments under one configuration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

MODELS = {
    "yolov11n": "ultralytics/cfg/models/11/yolo11.yaml",
    "ag_yolo": "ultralytics/cfg/models/11/yolo11-MAN-CTA-C2CGA.yaml",
}


def metric_value(metrics, attribute: str) -> float:
    value = metrics
    for part in attribute.split("."):
        value = getattr(value, part)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"D:\gcj\ultralytics-yolo11-main"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\datasets\An guang.yaml"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("runs/revision_multiseed"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.data = args.data.resolve()
    output = args.output if args.output.is_absolute() else args.repo / args.output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pretrained = args.repo / "yolo11n.pt"
    for required in (args.repo, args.data, pretrained):
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from ultralytics import YOLO

    protocol = {
        "data": str(args.data.resolve()),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "patience": 0,
        "workers": args.workers,
        "fraction": args.fraction,
        "amp": True,
        "deterministic": True,
        "pretrained_source": str(pretrained),
        "augmentation_alignment": {
            "close_mosaic": 0,
            "copy_paste": 0.0,
            "copy_paste_mode": "flip",
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    summary_path = output / "paired_results.csv"
    rows = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    for model_name, relative_config in MODELS.items():
        config = args.repo / relative_config
        for seed in args.seeds:
            run_name = f"{model_name}_seed{seed}"
            save_dir = output / run_name
            best = save_dir / "weights" / "best.pt"
            last = save_dir / "weights" / "last.pt"
            results_csv = save_dir / "results.csv"
            completed_epochs = 0
            if results_csv.exists():
                with results_csv.open(encoding="utf-8", newline="") as handle:
                    completed_epochs = sum(1 for _ in csv.DictReader(handle))

            existing_row = next(
                (r for r in rows if r["model"] == model_name and int(r["seed"]) == seed), None
            )
            if args.skip_existing and existing_row and best.exists():
                print(f"Skipping completed result: {run_name}")
                continue

            if completed_epochs >= args.epochs and best.exists():
                print(f"Training already complete; validating {run_name}")
            elif last.exists():
                print(f"Resuming {run_name} from epoch {completed_epochs}")
                model = YOLO(str(last))
                model.train(resume=str(last), device=args.device)
            else:
                model = YOLO(str(config))
                model.load(str(pretrained))
                model.train(
                    data=str(args.data),
                    epochs=args.epochs,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    workers=args.workers,
                    device=args.device,
                    project=str(output),
                    name=run_name,
                    exist_ok=False,
                    optimizer="SGD",
                    lr0=0.01,
                    lrf=0.01,
                    momentum=0.937,
                    weight_decay=0.0005,
                    warmup_epochs=3.0,
                    patience=0,
                    seed=seed,
                    deterministic=True,
                    amp=True,
                    close_mosaic=0,
                    copy_paste=0.0,
                    copy_paste_mode="flip",
                    val=True,
                    plots=True,
                    cache=False,
                    fraction=args.fraction,
                )
                save_dir = Path(model.trainer.save_dir)
                best = save_dir / "weights" / "best.pt"
            if not best.exists():
                raise FileNotFoundError(best)

            evaluated = YOLO(str(best)).val(
                data=str(args.data),
                split="val",
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                half=False,
                plots=False,
                project=str(output / "validation"),
                name=run_name,
            )
            row = {
                "model": model_name,
                "seed": seed,
                "precision": metric_value(evaluated, "box.mp"),
                "recall": metric_value(evaluated, "box.mr"),
                "mAP50": metric_value(evaluated, "box.map50"),
                "mAP50_95": metric_value(evaluated, "box.map"),
                "checkpoint": str(best),
            }
            rows = [r for r in rows if not (r["model"] == model_name and int(r["seed"]) == seed)]
            rows.append(row)
            with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    print(summary_path)


if __name__ == "__main__":
    main()

