import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import select_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatable pure-forward latency benchmark")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device, batch=args.batch)
    model = attempt_load_weights(args.weights, device=device, fuse=True).eval()
    inputs = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    if args.half:
        model = model.half()
        inputs = inputs.half()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        synchronize(device)

        runs = []
        for repeat_index in range(args.repeats):
            samples_ms = []
            for _ in range(args.iterations):
                synchronize(device)
                start = time.perf_counter()
                model(inputs)
                synchronize(device)
                samples_ms.append((time.perf_counter() - start) * 1000.0 / args.batch)

            runs.append(
                {
                    "repeat": repeat_index + 1,
                    "latency_ms_mean": float(np.mean(samples_ms)),
                    "latency_ms_sd": float(np.std(samples_ms, ddof=1)),
                    "latency_ms_median": float(np.median(samples_ms)),
                    "latency_ms_p95": float(np.percentile(samples_ms, 95)),
                    "fps": 1000.0 / float(np.mean(samples_ms)),
                }
            )

    run_means = [row["latency_ms_mean"] for row in runs]
    fps_values = [row["fps"] for row in runs]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "weights": str(Path(args.weights).resolve()),
        "protocol": {
            "scope": "fused PyTorch model forward pass only",
            "disk_io_included": False,
            "preprocessing_included": False,
            "postprocessing_nms_included": False,
            "cuda_synchronization": device.type == "cuda",
            "batch_size": args.batch,
            "input_size": [args.imgsz, args.imgsz],
            "precision": "FP16" if args.half else "FP32",
            "warmup_iterations": args.warmup,
            "timed_repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "summary": {
            "latency_ms_mean_across_repeats": statistics.mean(run_means),
            "latency_ms_sd_across_repeats": statistics.stdev(run_means) if len(run_means) > 1 else 0.0,
            "fps_mean_across_repeats": statistics.mean(fps_values),
            "fps_sd_across_repeats": statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0,
        },
        "runs": runs,
    }

    stem = f"{args.model_name.lower().replace(' ', '_')}_fp{'16' if args.half else '32'}_bs{args.batch}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)

    print(json.dumps(result["summary"], indent=2))
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
