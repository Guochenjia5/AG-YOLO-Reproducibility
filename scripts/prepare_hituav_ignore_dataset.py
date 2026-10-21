#!/usr/bin/env python3
"""Prepare a four-class HIT-UAV view while preserving DontCare regions.

The source tree is never modified. Class 4 is removed from ordinary YOLO target
files and written to ``ignore/<split>``. Images are linked or copied so that
Ultralytics can resolve ``images`` to the transformed ``labels`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle"]


def transfer_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def find_image(image_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidate_upper = image_dir / f"{stem}{suffix.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    raise FileNotFoundError(f"No image found for label stem {stem!r} in {image_dir}")


def prepare(source: Path, output: Path, splits: list[str], image_mode: str) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source dataset")
    manifest = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "ignore_rule": "source class 4 is stored separately and excluded from four-class targets",
        "splits": {},
    }
    for split in splits:
        src_labels = source / "labels" / split
        src_images = source / "images" / split
        if not src_labels.is_dir() or not src_images.is_dir():
            raise FileNotFoundError(f"Missing images/labels split: {split}")

        out_labels = output / "labels" / split
        out_ignore = output / "ignore" / split
        out_images = output / "images" / split
        out_labels.mkdir(parents=True, exist_ok=True)
        out_ignore.mkdir(parents=True, exist_ok=True)
        out_images.mkdir(parents=True, exist_ok=True)

        stats = {"images": 0, "target_boxes": 0, "dontcare_boxes": 0, "empty_target_files": 0}
        for label_path in sorted(src_labels.glob("*.txt")):
            target_lines: list[str] = []
            ignore_lines: list[str] = []
            for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = raw.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    raise ValueError(f"Malformed line {line_no} in {label_path}: {raw!r}")
                class_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:5]]
                if any(v < 0.0 or v > 1.0 for v in coords):
                    raise ValueError(f"Non-normalized box at {label_path}:{line_no}")
                normalized = " ".join(f"{v:.8f}" for v in coords)
                if class_id == 4:
                    ignore_lines.append(normalized)
                    stats["dontcare_boxes"] += 1
                elif 0 <= class_id < 4:
                    target_lines.append(f"{class_id} {normalized}")
                    stats["target_boxes"] += 1
                else:
                    raise ValueError(f"Unexpected class {class_id} at {label_path}:{line_no}")

            (out_labels / label_path.name).write_text(
                "\n".join(target_lines) + ("\n" if target_lines else ""), encoding="utf-8"
            )
            (out_ignore / label_path.name).write_text(
                "\n".join(ignore_lines) + ("\n" if ignore_lines else ""), encoding="utf-8"
            )
            if not target_lines:
                stats["empty_target_files"] += 1

            image_path = find_image(src_images, label_path.stem)
            transfer_image(image_path, out_images / image_path.name, image_mode)
            stats["images"] += 1

        manifest["splits"][split] = stats

    yaml_lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
        "nc: 4",
    ]
    (output / "dataset_ignore.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output / "ignore_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"D:\gcj\hit-uav"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\gcj\hit-uav-ignore"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--image-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.splits, args.image_mode), indent=2))


if __name__ == "__main__":
    main()

