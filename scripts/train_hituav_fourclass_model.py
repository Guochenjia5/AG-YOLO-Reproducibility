#!/usr/bin/env python3
"""Train one HIT-UAV model under a documented four-class HIT-UAV protocol.

This worker is launched in a fresh Python process for each model because the
comparison spans two local Ultralytics forks. DontCare boxes are excluded from
the four positive classes during training and filtered from predictions by the
companion evaluator. The original dataset is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def metric_value(metrics, attribute: str) -> float:
    value = metrics
    for part in attribute.split("."):
        value = getattr(value, part)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    config = args.model_config.resolve()
    dataset = args.dataset.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    for required in (repo, config, dataset, evaluator):
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO

    protocol = {
        "model_name": args.model_name,
        "repo": str(repo),
        "model_config": str(config),
        "dataset": str(dataset),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "fraction": args.fraction,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "patience": 0,
        "close_mosaic": 0,
        "amp": True,
        "deterministic": True,
        "pretrained": False,
        "evaluation_split": "test",
        "prediction_confidence_floor": 0.001,
        "prediction_nms_iou": 0.7,
        "dontcare_evaluation_ioa": 0.5,
        "protocol_scope": (
            "four target classes with class-agnostic DontCare filtering; "
            "the dataset manifest records whether the complete official split is used"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / f"protocol_{args.model_name}.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = output / "fourclass_results.csv"
    summary_rows = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))

    for seed in args.seeds:
        run_name = f"{args.model_name}_seed{seed}"
        train_dir = output / "train" / run_name
        metrics_json = output / "metrics" / f"{run_name}.json"
        resume_checkpoint = train_dir / "weights" / "last.pt"
        best = train_dir / "weights" / "best.pt"
        results_csv = train_dir / "results.csv"
        completed_training = False
        if results_csv.exists() and best.exists():
            completed_training = sum(1 for _ in results_csv.open(encoding="utf-8-sig")) >= args.epochs + 1
        if train_dir.exists():
            if args.skip_existing and metrics_json.exists():
                print(f"Skipping completed run: {run_name}")
                continue
            if not args.skip_existing or (not completed_training and not resume_checkpoint.exists()):
                raise FileExistsError(f"Run directory already exists and cannot be resumed: {train_dir}")

        if completed_training:
            print(f"Training already complete; evaluating existing checkpoint: {run_name}")
        elif train_dir.exists():
            print(f"Resuming interrupted run: {run_name} from {resume_checkpoint}")
            model = YOLO(str(resume_checkpoint))
            model.train(resume=True)
        else:
            model = YOLO(str(config))
            model.train(
                data=str(dataset),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                workers=args.workers,
                device=args.device,
                project=str(output / "train"),
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
                pretrained=False,
                close_mosaic=0,
                val=True,
                plots=True,
                cache=False,
                fraction=args.fraction,
            )
        if not best.exists():
            raise FileNotFoundError(best)

        # Keep an ordinary Ultralytics validation record for diagnostics. The
        # paper-facing AP values come from the DontCare-aware evaluator below.
        validated = YOLO(str(best)).val(
            data=str(dataset),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            half=False,
            plots=False,
            project=str(output / "ultralytics_test"),
            name=run_name,
        )

        pred_project = output / "predictions"
        prediction_model = YOLO(str(best))
        prediction_stream = prediction_model.predict(
            source=str(dataset.parent / "images" / "test"),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(pred_project),
            name=run_name,
            exist_ok=False,
            stream=True,
            verbose=False,
        )
        for _ in prediction_stream:
            pass
        pred_labels = pred_project / run_name / "labels"

        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_csv = output / "metrics" / f"{run_name}_per_class.csv"
        subprocess.run(
            [
                sys.executable,
                str(evaluator),
                "--gt-root",
                str(dataset.parent),
                "--pred-root",
                str(pred_labels),
                "--split",
                "test",
                "--ignore-ioa",
                "0.5",
                "--output-json",
                str(metrics_json),
                "--output-csv",
                str(metrics_csv),
            ],
            check=True,
        )
        corrected = json.loads(metrics_json.read_text(encoding="utf-8"))
        row = {
            "model": args.model_name,
            "seed": seed,
            "corrected_precision": corrected["precision"],
            "corrected_recall": corrected["recall"],
            "corrected_mAP50": corrected["mAP50"],
            "corrected_mAP50_95": corrected["mAP50_95"],
            "predictions_before_ignore": corrected["predictions_before_ignore"],
            "predictions_removed_by_ignore": corrected["predictions_removed_by_ignore"],
            "ultralytics_precision": metric_value(validated, "box.mp"),
            "ultralytics_recall": metric_value(validated, "box.mr"),
            "ultralytics_mAP50": metric_value(validated, "box.map50"),
            "ultralytics_mAP50_95": metric_value(validated, "box.map"),
            "checkpoint": str(best),
            "metrics_json": str(metrics_json),
        }
        summary_rows = [r for r in summary_rows if not (r["model"] == args.model_name and int(r["seed"]) == seed)]
        summary_rows.append(row)
        with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
