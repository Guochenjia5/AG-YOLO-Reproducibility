#!/usr/bin/env python3
"""Train the YOLO-HVS independent reimplementation on Night-Drone Vehicle.

The implementation follows the topology disclosed in the YOLO-HVS paper, while
the training and evaluation settings follow the AG-YOLO Night-Drone protocol.
It must be reported as an independent reimplementation, never as the authors'
official implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch


MODEL_LABEL = "YOLO-HVS independent reimplementation"
OFFICIAL_WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(metrics, attribute: str) -> float:
    value = metrics
    for part in attribute.split("."):
        value = getattr(value, part)
    return float(value)


def transfer_compatible_tensors(target: torch.nn.Module, source: torch.nn.Module) -> dict[str, object]:
    """Load only tensors whose names and shapes agree, and return an auditable summary."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    loaded_parameters = sum(value.numel() for value in compatible.values())
    target_parameters = sum(value.numel() for value in target_state.values())
    return {
        "loaded_tensors": len(compatible),
        "loaded_parameters": loaded_parameters,
        "target_state_parameters": target_parameters,
        "loaded_fraction_of_target_state": loaded_parameters / target_parameters,
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "loaded_tensor_names": sorted(compatible),
    }


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def assert_finite_losses(results_csv: Path) -> None:
    """Reject a checkpoint when any recorded training or validation loss is non-finite."""

    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        loss_columns = [name for name in (reader.fieldnames or []) if "loss" in name.lower()]
        if not loss_columns:
            raise RuntimeError(f"No loss columns found in {results_csv}")
        for row_number, row in enumerate(reader, start=2):
            for column in loss_columns:
                raw = (row.get(column) or "").strip()
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Non-numeric loss in {results_csv}, row {row_number}, column {column}: {raw!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite loss in {results_csv}, row {row_number}, column {column}: {raw!r}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"D:\gcj\ultralytics-yolo11-main"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\datasets\An guang.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolov8-hvs-reimpl.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolov8-hvs-baseline.yaml"),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path(r"D:\gcj\pretrained-cache\yolov8n.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\gcj\night-drone-yolo-hvs-reimpl-b32-fp32"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable automatic mixed precision; the approved formal run uses --no-amp.",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    for name in ("repo", "data", "config", "baseline_config", "pretrained", "output"):
        setattr(args, name, getattr(args, name).resolve())
    for required in (args.repo, args.data, args.config, args.baseline_config, args.pretrained):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    baseline_declared = YOLO(str(args.baseline_config)).model
    hvs_declared = YOLO(str(args.config)).model
    baseline_actual = DetectionModel(str(args.baseline_config), ch=3, nc=5, verbose=False)
    hvs_actual = DetectionModel(str(args.config), ch=3, nc=5, verbose=False)
    source_model = YOLO(str(args.pretrained)).model
    transfer = transfer_compatible_tensors(hvs_declared, source_model)

    audit = {
        "model_label": MODEL_LABEL,
        "status": "independent_reimplementation",
        "official_author_code": False,
        "paper": "YOLO-HVS: An Infrared Small Target Detection Algorithm Inspired by the Human Visual System",
        "doi": "10.3390/biomimetics10070451",
        "paper_reported_model": {
            "parameters": 5_400_000,
            "gflops": 8.2,
            "baseline_parameters": 3_100_000,
            "baseline_gflops": 8.1,
        },
        "implemented_topology": {
            "basis": "YOLOv8n",
            "backbone_and_neck": "Every YOLOv8 C2f block is replaced by C2f_DWR",
            "dwr_dilation_rates": [1, 3, 5],
            "dwr_branch_channel_ratio": [2, 1, 1],
            "head": "Three YOLOv8 detection towers with MultiSEAM replacing the second tower convolution",
            "multiseam_patch_sizes": [3, 5, 7],
        },
        "ambiguity_resolution": {
            "multiseam_patch_sizes": (
                "The paper prose states 6/7/8, while the cited YOLO-FaceV2 implementation uses 3/5/7. "
                "The 3/5/7 implementation also agrees with the paper's reported 5.4M parameter scale."
            ),
            "dwr_operator": (
                "The paper calls DConv depthwise/dilated convolution, while the parameter ablation is consistent "
                "with the published common C2f_DWR implementation used here."
            ),
            "head_location": (
                "The paper diagram places three MultiSEAM modules at the end of the neck; the implementation uses "
                "one MultiSEAM in each regression and classification tower at each of the three detection scales, "
                "matching the cited MultiSEAM detector integration and reported parameter scale."
            ),
        },
        "model_complexity_at_640": {
            "baseline_declared_nc80_parameters": sum(p.numel() for p in baseline_declared.parameters()),
            "baseline_declared_nc80_gflops": get_flops(baseline_declared, args.imgsz),
            "hvs_declared_nc80_parameters": sum(p.numel() for p in hvs_declared.parameters()),
            "hvs_declared_nc80_gflops": get_flops(hvs_declared, args.imgsz),
            "baseline_actual_nc5_parameters": sum(p.numel() for p in baseline_actual.parameters()),
            "baseline_actual_nc5_gflops": get_flops(baseline_actual, args.imgsz),
            "hvs_actual_nc5_parameters": sum(p.numel() for p in hvs_actual.parameters()),
            "hvs_actual_nc5_gflops": get_flops(hvs_actual, args.imgsz),
        },
        "initialization": {
            "checkpoint": str(args.pretrained),
            "sha256": sha256(args.pretrained),
            "checkpoint_bytes": args.pretrained.stat().st_size,
            "reference_download_url": OFFICIAL_WEIGHT_URL,
            "checkpoint_model_yaml_nc": getattr(source_model, "yaml", {}).get("nc"),
            **transfer,
        },
        "source_artifacts": {
            "model_config": str(args.config),
            "baseline_config": str(args.baseline_config),
            "runner": str(Path(__file__).resolve()),
        },
    }
    (args.output / "reimplementation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    protocol = {
        "model_label": MODEL_LABEL,
        "data": str(args.data),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "fraction": args.fraction,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "patience": 0,
        "amp": args.amp,
        "deterministic": True,
        "augmentation_alignment": {"close_mosaic": 0, "copy_paste": 0.0, "copy_paste_mode": "flip"},
        "checkpoint_for_independent_validation": "best.pt",
        "evaluation": "Ultralytics validation split, same data YAML and image size as AG-YOLO",
        "comparison_scope": "Night-Drone Vehicle same-data independent reimplementation",
        "differences_from_yolo_hvs_paper": {
            "batch": "32 here versus 8 reported in the paper",
            "momentum": "0.937 here versus 0.9 reported in the paper",
            "amp": "disabled here; the paper does not disclose AMP",
            "epochs": "300 here; the paper does not disclose epoch count",
            "imgsz": "640 here; the paper does not disclose training image size",
            "dataset": "Night-Drone Vehicle subset here versus full DroneVehicle in the paper",
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    summary_path = args.output / "results.csv"
    rows: list[dict[str, object]] = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    for seed in args.seeds:
        run_name = f"yolo_hvs_reimpl_seed{seed}"
        save_dir = args.output / run_name
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        train_csv = save_dir / "results.csv"
        epoch_count = completed_epochs(train_csv)
        existing = next((row for row in rows if int(row["seed"]) == seed), None)
        if args.skip_existing and existing and best.exists() and epoch_count >= args.epochs:
            print(f"Skipping completed result: {run_name}")
            continue

        if epoch_count >= args.epochs and best.exists():
            assert_finite_losses(train_csv)
            print(f"Training already complete; validating {run_name}")
        elif last.exists():
            assert_finite_losses(train_csv)
            print(f"Safely resuming {run_name} from epoch {epoch_count}")
            model = YOLO(str(last))
            model.train(resume=str(last), device=args.device)
        else:
            if epoch_count:
                raise RuntimeError(f"Refusing a fresh overwrite: {train_csv} already contains {epoch_count} epochs")
            model = YOLO(str(args.config))
            source_model = YOLO(str(args.pretrained)).model
            run_transfer = transfer_compatible_tensors(model.model, source_model)
            # Use the public Model.load() path as well as the explicit audit above.
            # Besides loading the same compatible tensors, this sets model.ckpt so
            # YOLO.train() forwards the initialized custom model into the nc=5
            # training model instead of silently rebuilding from random weights.
            model.load(str(args.pretrained))
            print(
                f"Loaded {run_transfer['loaded_tensors']} compatible YOLOv8n tensors "
                f"({run_transfer['loaded_parameters']} state parameters)."
            )
            model.train(
                data=str(args.data),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                workers=args.workers,
                device=args.device,
                project=str(args.output),
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
                amp=args.amp,
                pretrained=False,
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
            train_csv = save_dir / "results.csv"

        assert_finite_losses(train_csv)
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
            project=str(args.output / "validation"),
            name=run_name,
        )
        row = {
            "model": MODEL_LABEL,
            "seed": seed,
            "precision": metric_value(evaluated, "box.mp"),
            "recall": metric_value(evaluated, "box.mr"),
            "mAP50": metric_value(evaluated, "box.map50"),
            "mAP50_95": metric_value(evaluated, "box.map"),
            "checkpoint": str(best),
        }
        rows = [item for item in rows if int(item["seed"]) != seed]
        rows.append(row)
        with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    print(summary_path)


if __name__ == "__main__":
    main()
