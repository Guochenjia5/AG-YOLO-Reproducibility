#!/usr/bin/env python3
"""Build the complete official HIT-UAV split for four-class evaluation.

The published JSON files contain 2,029/290/579 image records.  The commonly
converted local YOLO copy omits 32 zero-annotation background images.  This
script preserves the already verified four-class labels and DontCare sidecars,
adds those background images from the authors' repository, and records hashes
and count checks in a manifest.  The input dataset is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")
CLASS_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle"]
RAW_BASE = (
    "https://raw.githubusercontent.com/suojiashun/"
    "HIT-UAV-Infrared-Thermal-Dataset/main/normal_json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def annotations(document: dict) -> list[dict]:
    # The authors' files use the singular key ``annotation``.
    return document.get("annotation", document.get("annotations", []))


def image_name(record: dict) -> str:
    value = record.get("file_name", record.get("filename"))
    if not value:
        raise ValueError(f"Image record has no filename: {record}")
    return Path(value).name


def build(prepared: Path, json_root: Path, output: Path, raw_base: str) -> dict:
    prepared = prepared.resolve()
    json_root = json_root.resolve()
    output = output.resolve()
    if output == prepared:
        raise ValueError("Output must differ from the prepared input dataset")

    manifest: dict = {
        "protocol": "complete official HIT-UAV split; four target classes; DontCare stored as ignore regions",
        "official_repository": "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset",
        "prepared_source": str(prepared),
        "official_json_root": str(json_root),
        "output": str(output),
        "splits": {},
    }

    for split in SPLITS:
        json_path = json_root / f"{split}.json"
        document = json.loads(json_path.read_text(encoding="utf-8"))
        images = document["images"]
        anns = annotations(document)
        image_ids = {int(item["id"]): item for item in images}
        ann_counts = Counter(int(item["image_id"]) for item in anns)
        category_counts = Counter(int(item["category_id"]) for item in anns)

        names = [image_name(item) for item in images]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate filenames in {json_path}")

        out_images = output / "images" / split
        out_labels = output / "labels" / split
        out_ignore = output / "ignore" / split
        for directory in (out_images, out_labels, out_ignore):
            directory.mkdir(parents=True, exist_ok=True)

        downloaded: list[str] = []
        copied: list[str] = []
        for name in names:
            stem = Path(name).stem
            src_image = prepared / "images" / split / name
            src_label = prepared / "labels" / split / f"{stem}.txt"
            src_ignore = prepared / "ignore" / split / f"{stem}.txt"
            dst_image = out_images / name
            dst_label = out_labels / f"{stem}.txt"
            dst_ignore = out_ignore / f"{stem}.txt"

            if src_image.exists():
                if not src_label.exists() or not src_ignore.exists():
                    raise FileNotFoundError(f"Missing prepared sidecar for {src_image}")
                transfer(src_image, dst_image)
                transfer(src_label, dst_label)
                transfer(src_ignore, dst_ignore)
                copied.append(name)
            else:
                record = next(item for item in images if image_name(item) == name)
                if ann_counts[int(record["id"])] != 0:
                    raise ValueError(f"Refusing to synthesize labels for annotated missing image: {name}")
                if not dst_image.exists():
                    url = f"{raw_base.rstrip('/')}/{split}/{name}"
                    urllib.request.urlretrieve(url, dst_image)
                dst_label.write_text("", encoding="utf-8")
                dst_ignore.write_text("", encoding="utf-8")
                downloaded.append(name)

        expected_names = set(names)
        actual_images = {item.name for item in out_images.iterdir() if item.is_file()}
        actual_labels = {item.stem for item in out_labels.glob("*.txt")}
        actual_ignore = {item.stem for item in out_ignore.glob("*.txt")}
        expected_stems = {Path(name).stem for name in names}
        if actual_images != expected_names:
            raise RuntimeError(f"Image set mismatch for {split}")
        if actual_labels != expected_stems or actual_ignore != expected_stems:
            raise RuntimeError(f"Sidecar set mismatch for {split}")

        target_boxes = 0
        dontcare_boxes = 0
        empty_targets = 0
        for path in out_labels.glob("*.txt"):
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            target_boxes += len(lines)
            empty_targets += int(not lines)
        for path in out_ignore.glob("*.txt"):
            dontcare_boxes += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        expected_targets = sum(category_counts[index] for index in range(4))
        expected_dontcare = category_counts[4]
        if target_boxes != expected_targets or dontcare_boxes != expected_dontcare:
            raise RuntimeError(
                f"Box count mismatch for {split}: targets {target_boxes}/{expected_targets}, "
                f"DontCare {dontcare_boxes}/{expected_dontcare}"
            )

        manifest["splits"][split] = {
            "official_images": len(images),
            "official_annotations": len(anns),
            "four_class_boxes": target_boxes,
            "dontcare_boxes": dontcare_boxes,
            "empty_target_images": empty_targets,
            "reused_images": len(copied),
            "downloaded_zero_annotation_images": len(downloaded),
            "downloaded_files": downloaded,
            "json_sha256": sha256(json_path),
        }

    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
        "nc: 4",
    ]
    (output / "dataset_ignore.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output / "official_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path(r"D:\gcj\hit-uav-ignore"))
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "official-hituav",
    )
    parser.add_argument("--output", type=Path, default=Path(r"D:\gcj\hit-uav-official-ignore"))
    parser.add_argument("--raw-base", default=RAW_BASE)
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.json_root, args.output, args.raw_base), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
