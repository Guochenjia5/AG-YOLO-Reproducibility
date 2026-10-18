#!/usr/bin/env python3
"""Evaluate four HIT-UAV classes after class-agnostic DontCare filtering.

Ground truth must come from ``prepare_hituav_ignore_dataset.py``. Prediction
files use Ultralytics ``save_txt=True, save_conf=True`` format:
``class x_center y_center width height confidence``.

When ``official_protocol_manifest.json`` and ``official_protocol_audit.json``
are present, their provenance is embedded in the metrics.  The authors' AP.py
is explicitly marked as unfinished, so this transparent evaluator is described
as reviewer-aligned rather than as an author-supplied official evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle"]
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def xywh_to_xyxy(values: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = values
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa_prediction(pred, ignore) -> float:
    pred_area = area(pred)
    return intersection(pred, ignore) / pred_area if pred_area > 0 else 0.0


def read_targets(path: Path) -> list[Box]:
    boxes = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed target line in {path}: {raw!r}")
        cls = int(float(parts[0]))
        if cls not in range(4):
            raise ValueError(f"Unexpected target class {cls} in {path}")
        boxes.append(Box(cls, xywh_to_xyxy([float(v) for v in parts[1:5]])))
    return boxes


def read_ignore(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        coords = parts[-4:]
        boxes.append(xywh_to_xyxy([float(v) for v in coords]))
    return boxes


def read_predictions(path: Path) -> list[Box]:
    boxes = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"Expected class xywh confidence in {path}: {raw!r}")
        cls = int(float(parts[0]))
        if cls not in range(4):
            continue
        boxes.append(Box(cls, xywh_to_xyxy([float(v) for v in parts[1:5]]), float(parts[5])))
    return boxes


def ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    values = []
    for target_recall in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= target_recall]
        values.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(values))


def evaluate(gt_root: Path, pred_root: Path, split: str, ignore_ioa: float) -> dict:
    label_dir = gt_root / "labels" / split
    ignore_dir = gt_root / "ignore" / split
    if not label_dir.is_dir() or not ignore_dir.is_dir():
        raise FileNotFoundError("Prepared labels/ignore directories were not found")

    image_records = {}
    ignored_predictions = 0
    total_predictions = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        targets = read_targets(label_path)
        ignores = read_ignore(ignore_dir / label_path.name)
        predictions = read_predictions(pred_root / label_path.name)
        total_predictions += len(predictions)
        retained = []
        for pred in predictions:
            if any(ioa_prediction(pred.xyxy, region) >= ignore_ioa for region in ignores):
                ignored_predictions += 1
            else:
                retained.append(pred)
        image_records[label_path.stem] = (targets, retained)

    per_class = []
    pr50_data = []
    for cls, name in enumerate(CLASS_NAMES):
        gt_count = sum(sum(t.cls == cls for t in targets) for targets, _ in image_records.values())
        preds = []
        for image_id, (_targets, image_preds) in image_records.items():
            preds.extend((p.confidence, image_id, p) for p in image_preds if p.cls == cls)
        preds.sort(key=lambda item: item[0], reverse=True)

        aps = []
        class_pr50 = None
        for threshold in IOU_THRESHOLDS:
            matched = {image_id: set() for image_id in image_records}
            tp = np.zeros(len(preds), dtype=float)
            fp = np.zeros(len(preds), dtype=float)
            for index, (_confidence, image_id, pred) in enumerate(preds):
                targets = [t for t in image_records[image_id][0] if t.cls == cls]
                candidates = [
                    (iou(pred.xyxy, target.xyxy), target_index)
                    for target_index, target in enumerate(targets)
                    if target_index not in matched[image_id]
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_iou >= threshold:
                    tp[index] = 1.0
                    matched[image_id].add(best_index)
                else:
                    fp[index] = 1.0
            if gt_count == 0:
                aps.append(math.nan)
                continue
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / gt_count
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            if math.isclose(float(threshold), 0.5):
                class_pr50 = {
                    "scores": np.asarray([item[0] for item in preds], dtype=float),
                    "cumulative_tp": cumulative_tp,
                    "cumulative_fp": cumulative_fp,
                    "gt_count": gt_count,
                }
            aps.append(ap_101(recall, precision))

        has_ground_truth = gt_count > 0
        per_class.append(
            {
                "class_id": cls,
                "class_name": name,
                "ground_truth_boxes": gt_count,
                "predictions_after_ignore": len(preds),
                "AP50": aps[0] if has_ground_truth else None,
                "AP50_95": float(np.nanmean(aps)) if has_ground_truth else None,
            }
        )
        pr50_data.append(class_pr50)

    valid = [row for row in per_class if row["ground_truth_boxes"] > 0]
    # Match the usual detector-table convention by selecting one shared
    # confidence threshold that maximizes mean per-class F1 at IoU 0.50.
    confidence_grid = np.linspace(0.0, 1.0, 1001)
    precision_grid = np.zeros((len(per_class), confidence_grid.size), dtype=float)
    recall_grid = np.zeros_like(precision_grid)
    for class_index, data in enumerate(pr50_data):
        if data is None or data["gt_count"] == 0:
            continue
        scores = data["scores"]
        cumulative_tp = data["cumulative_tp"]
        cumulative_fp = data["cumulative_fp"]
        for threshold_index, confidence in enumerate(confidence_grid):
            count = int(np.searchsorted(-scores, -confidence, side="right")) if scores.size else 0
            if count == 0:
                continue
            tp_count = cumulative_tp[count - 1]
            fp_count = cumulative_fp[count - 1]
            precision_grid[class_index, threshold_index] = tp_count / max(tp_count + fp_count, 1e-12)
            recall_grid[class_index, threshold_index] = tp_count / data["gt_count"]

    valid_indices = [index for index, row in enumerate(per_class) if row["ground_truth_boxes"] > 0]
    if valid_indices:
        f1_grid = 2 * precision_grid * recall_grid / np.maximum(precision_grid + recall_grid, 1e-12)
        best_threshold_index = int(np.argmax(f1_grid[valid_indices].mean(axis=0)))
        operating_confidence = float(confidence_grid[best_threshold_index])
        mean_precision = float(precision_grid[valid_indices, best_threshold_index].mean())
        mean_recall = float(recall_grid[valid_indices, best_threshold_index].mean())
        for class_index, row in enumerate(per_class):
            row["precision_at_operating_point"] = float(precision_grid[class_index, best_threshold_index])
            row["recall_at_operating_point"] = float(recall_grid[class_index, best_threshold_index])
    else:
        operating_confidence = None
        mean_precision = None
        mean_recall = None
    provenance = {
        "dataset_scope": "prepared dataset without an official manifest",
        "author_evaluator_status": "not used",
    }
    manifest_path = gt_root / "official_protocol_manifest.json"
    audit_path = gt_root / "official_protocol_audit.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance["dataset_scope"] = manifest.get("protocol")
        provenance["official_repository"] = manifest.get("official_repository")
        provenance["official_json_sha256"] = manifest["splits"][split]["json_sha256"]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        provenance["geometry_audit_status"] = audit.get("status")
        provenance["geometry_audit_split"] = audit.get("splits", {}).get(split)
    provenance["author_evaluator_status"] = (
        "repository AP.py is marked unfinished and does not implement a complete AP calculation; "
        "the rule below is an explicit reviewer-aligned implementation"
    )
    return {
        "protocol": {
            "classes": CLASS_NAMES,
            "iou_thresholds": [round(float(v), 2) for v in IOU_THRESHOLDS],
            "ap_interpolation": "101-point",
            "precision_recall_rule": "one shared confidence threshold maximizing mean per-class F1 at IoU 0.50",
            "ignore_rule": f"drop predictions with intersection/prediction-area >= {ignore_ioa} against any DontCare box",
            "scope": "complete official split when confirmed by provenance; reviewer-aligned evaluator",
        },
        "provenance": provenance,
        "images": len(image_records),
        "predictions_before_ignore": total_predictions,
        "predictions_removed_by_ignore": ignored_predictions,
        "operating_confidence": operating_confidence,
        "precision": mean_precision,
        "recall": mean_recall,
        "mAP50": float(np.mean([row["AP50"] for row in valid])) if valid else None,
        "mAP50_95": float(np.mean([row["AP50_95"] for row in valid])) if valid else None,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ignore-ioa", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=Path("hituav_ignore_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("hituav_ignore_per_class.csv"))
    args = parser.parse_args()
    result = evaluate(args.gt_root, args.pred_root, args.split, args.ignore_ioa)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_class"][0]))
        writer.writeheader()
        writer.writerows(result["per_class"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

