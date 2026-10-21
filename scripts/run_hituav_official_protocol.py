#!/usr/bin/env python3
"""Run every manuscript HIT-UAV comparison on the complete official split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE = HERE / "run_hituav_fourclass_comparison.py"
REMAINING = HERE / "run_hituav_fourclass_remaining.py"
DATASET = Path(r"D:\gcj\hit-uav-official-ignore\dataset_ignore.yaml")
DEFAULT_OUTPUT = Path(r"D:\gcj\hituav-official-4class")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for required in (CORE, REMAINING, args.dataset):
        if not required.exists():
            raise FileNotFoundError(required)

    common = [
        "--dataset", str(args.dataset.resolve()),
        "--output", str(args.output.resolve()),
        "--device", args.device,
        "--skip-existing",
    ]
    commands = [
        [sys.executable, str(CORE), "--stage", "seed0", *common],
        [sys.executable, str(REMAINING), "--stage", "seed0", *common],
        [sys.executable, str(CORE), "--stage", "core-extra", *common],
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": "complete official HIT-UAV split, four classes, DontCare ignore regions",
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "expected_runs": 13,
        "commands": commands,
    }
    (args.output / "official_protocol_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
