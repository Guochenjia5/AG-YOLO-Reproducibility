#!/usr/bin/env python3
"""Run the independent YOLO-CIR reimplementation on Night-Drone Vehicle.

The model follows the components disclosed in the YOLO-CIR paper, while the
dataset, optimizer, schedule, augmentation and evaluator follow the AG-YOLO
Night-Drone protocol.  It must be reported as an independent reimplementation,
not as the authors' official model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


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


def load_convnext_nano_backbone(backbone: torch.nn.Module, checkpoint: Path) -> dict[str, int]:
    """Map a timm ConvNeXt-Nano ImageNet checkpoint into the CIR backbone."""

    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = backbone.state_dict()
    mapped: dict[str, torch.Tensor] = {}

    def copy(source_key: str, target_key: str, squeeze: bool = False) -> None:
        if source_key not in source or target_key not in target:
            return
        value = source[source_key]
        if squeeze and value.ndim == 4 and value.shape[-2:] == (1, 1):
            value = value[:, :, 0, 0]
        if value.shape == target[target_key].shape:
            mapped[target_key] = value

    copy("stem.0.weight", "downsample_layers.0.0.weight")
    copy("stem.0.bias", "downsample_layers.0.0.bias")
    copy("stem.1.weight", "downsample_layers.0.1.weight")
    copy("stem.1.bias", "downsample_layers.0.1.bias")
    for stage in range(1, 4):
        for item in ("weight", "bias"):
            copy(
                f"stages.{stage}.downsample.0.{item}",
                f"downsample_layers.{stage}.0.{item}",
            )
            copy(
                f"stages.{stage}.downsample.1.{item}",
                f"downsample_layers.{stage}.1.{item}",
            )

    source_depths = (2, 2, 8, 2)
    for stage, depth in enumerate(source_depths):
        for block in range(depth):
            source_prefix = f"stages.{stage}.blocks.{block}"
            target_prefix = f"stages.{stage}.{block}"
            copy(f"{source_prefix}.gamma", f"{target_prefix}.gamma")
            copy(f"{source_prefix}.conv_dw.weight", f"{target_prefix}.dwconv.weight")
            copy(f"{source_prefix}.conv_dw.bias", f"{target_prefix}.dwconv.bias")
            copy(f"{source_prefix}.norm.weight", f"{target_prefix}.norm.weight")
            copy(f"{source_prefix}.norm.bias", f"{target_prefix}.norm.bias")
            copy(f"{source_prefix}.mlp.fc1.weight", f"{target_prefix}.expand.weight", squeeze=True)
            copy(f"{source_prefix}.mlp.fc1.bias", f"{target_prefix}.expand.bias")
            copy(f"{source_prefix}.mlp.fc2.weight", f"{target_prefix}.project.weight", squeeze=True)
            copy(f"{source_prefix}.mlp.fc2.bias", f"{target_prefix}.project.bias")

    missing, unexpected = backbone.load_state_dict(mapped, strict=False)
    return {
        "loaded_tensors": len(mapped),
        "loaded_parameters": sum(value.numel() for value in mapped.values()),
        "backbone_parameters": sum(value.numel() for value in target.values()),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }


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
        default=Path(r"D:\gcj\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolo-cir-reimpl.yaml"),
    )
    parser.add_argument(
        "--backbone-pretrained",
        type=Path,
        default=Path(r"D:\gcj\pretrained-cache\convnext_nano.d1h_in1k.pth"),
    )
    parser.add_argument("--output", type=Path, default=Path(r"D:\gcj\night-drone-yolo-cir-reimpl"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision; pass --no-amp for full-precision training.",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.data = args.data.resolve()
    args.config = args.config.resolve()
    args.backbone_pretrained = args.backbone_pretrained.resolve()
    args.output = args.output.resolve()
    for required in (args.repo, args.data, args.config, args.backbone_pretrained):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from ultralytics import YOLO

    audit_model = YOLO(str(args.config))
    model_parameters = sum(parameter.numel() for parameter in audit_model.model.parameters())
    backbone = audit_model.model.model[0]
    transfer = load_convnext_nano_backbone(backbone, args.backbone_pretrained)
    audit = {
        "status": "independent_reimplementation",
        "official_author_code": False,
        "paper": "YOLO-CIR: The network based on YOLO and ConvNeXt for infrared object detection",
        "doi": "10.1016/j.infrared.2023.104703",
        "publicly_disclosed_components": [
            "YOLOv5m basis",
            "modified ConvNeXt backbone",
            "coordinate attention inside ConvNeXt blocks",
            "spatial attention on shallow features",
            "channel attention on deep features",
            "split attention in the neck",
        ],
        "implementation_choices_not_uniquely_specified_by_paper": {
            "convnext_stage_depths": [3, 3, 9, 3],
            "convnext_stage_channels": [80, 160, 320, 640],
            "detection_strides": [8, 16, 32],
            "split_attention_radix": 2,
            "neck_block": "Ultralytics C3",
            "detection_head": "Ultralytics anchor-free Detect",
        },
        "model_parameters": model_parameters,
        "model_gflops_at_640": 83.3,
        "backbone_initialization": {
            "source": "timm convnext_nano.d1h_in1k",
            "checkpoint": str(args.backbone_pretrained),
            "sha256": sha256(args.backbone_pretrained),
            **transfer,
        },
    }
    (args.output / "reimplementation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    protocol = {
        "data": str(args.data),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
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
        "evaluation": "Ultralytics validation split, same data YAML and image size as AG-YOLO",
        "initialization_difference_from_ag_yolo": (
            "ConvNeXt-Nano ImageNet backbone initialization; the custom backbone is incompatible with yolo11n.pt. "
            "The neck, attention modules and detector are randomly initialized."
        ),
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
        run_name = f"yolo_cir_reimpl_seed{seed}"
        save_dir = args.output / run_name
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        train_csv = save_dir / "results.csv"
        completed_epochs = 0
        if train_csv.exists():
            with train_csv.open(encoding="utf-8", newline="") as handle:
                completed_epochs = sum(1 for _ in csv.DictReader(handle))
        existing = next((row for row in rows if int(row["seed"]) == seed), None)
        if args.skip_existing and existing and best.exists() and completed_epochs >= args.epochs:
            print(f"Skipping completed result: {run_name}")
            continue

        if completed_epochs >= args.epochs and best.exists():
            print(f"Training already complete; validating {run_name}")
        elif last.exists():
            print(f"Resuming {run_name} from epoch {completed_epochs}")
            model = YOLO(str(last))
            model.train(resume=str(last), device=args.device)
        else:
            model = YOLO(str(args.config))
            transfer = load_convnext_nano_backbone(model.model.model[0], args.backbone_pretrained)
            print(f"Loaded ConvNeXt tensors: {transfer}")
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
            "model": "YOLO-CIR reimplementation",
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
