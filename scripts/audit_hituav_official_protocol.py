#!/usr/bin/env python3
"""Audit a prepared HIT-UAV tree against the authors' original JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_boxes(path: Path, ignore: bool) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        boxes.append((4 if ignore else int(float(parts[0])), *map(float, parts[-4:])))
    return boxes


def audit(dataset: Path, json_root: Path, tolerance: float) -> dict:
    report = {
        "dataset": str(dataset.resolve()),
        "official_json_root": str(json_root.resolve()),
        "coordinate_tolerance": tolerance,
        "splits": {},
        "status": "passed",
    }
    for split in ("train", "val", "test"):
        document = json.loads((json_root / f"{split}.json").read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in document["images"]}
        expected_targets = defaultdict(list)
        expected_ignore = defaultdict(list)
        annotations = document.get("annotation", document.get("annotations", []))
        for ann in annotations:
            image = images[int(ann["image_id"])]
            x, y, width, height = map(float, ann["bbox"])
            normalized = (
                int(ann["category_id"]),
                (x + width / 2) / float(image["width"]),
                (y + height / 2) / float(image["height"]),
                width / float(image["width"]),
                height / float(image["height"]),
            )
            stem = Path(image.get("filename", image.get("file_name"))).stem
            (expected_ignore if normalized[0] == 4 else expected_targets)[stem].append(normalized)

        mismatches = []
        max_error = 0.0
        for image in images.values():
            filename = Path(image.get("filename", image.get("file_name")))
            stem = filename.stem
            if not (dataset / "images" / split / filename.name).exists():
                mismatches.append({"file": filename.name, "kind": "missing_image"})
            for folder, ignore, expected in (
                ("labels", False, expected_targets[stem]),
                ("ignore", True, expected_ignore[stem]),
            ):
                path = dataset / folder / split / f"{stem}.txt"
                if not path.exists():
                    mismatches.append({"file": str(path), "kind": "missing_sidecar"})
                    continue
                actual = read_boxes(path, ignore)
                if len(actual) != len(expected):
                    mismatches.append(
                        {"file": str(path), "kind": "box_count", "actual": len(actual), "expected": len(expected)}
                    )
                    continue
                remaining = list(actual)
                for wanted in expected:
                    best_index = min(
                        range(len(remaining)),
                        key=lambda index: (
                            max(abs(remaining[index][axis] - wanted[axis]) for axis in range(1, 5))
                            if remaining[index][0] == wanted[0]
                            else float("inf")
                        ),
                    )
                    found = remaining.pop(best_index)
                    error = max(abs(found[axis] - wanted[axis]) for axis in range(1, 5))
                    max_error = max(max_error, error)
                    if found[0] != wanted[0] or error > tolerance:
                        mismatches.append(
                            {"file": str(path), "kind": "box_geometry", "actual": found, "expected": wanted}
                        )

        split_report = {
            "official_images": len(images),
            "official_annotations": len(annotations),
            "maximum_normalized_coordinate_error": max_error,
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:20],
            "status": "passed" if not mismatches else "failed",
        }
        report["splits"][split] = split_report
        if mismatches:
            report["status"] = "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\gcj\hit-uav-official-ignore"))
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "official-hituav",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset.resolve(), args.json_root.resolve(), args.tolerance)
    output = args.output or args.dataset / "official_protocol_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
