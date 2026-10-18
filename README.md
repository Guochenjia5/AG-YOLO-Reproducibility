# AG-YOLO reproducibility code

This repository contains the essential source files required to build AG-YOLO,
run the principal Night-Drone Vehicle and HIT-UAV experiments, apply the
DontCare-aware HIT-UAV evaluator, and repeat the latency benchmark.

## Included files

- `model/`: AG-YOLO model configuration and the required Ultralytics module/parser changes.
- `scripts/`: dataset preparation, training, evaluation, and timing scripts.
- `results/`: compact result tables reported in the revised manuscript.
- `requirements.txt`: recorded Python package versions.

Datasets, pretrained weights, trained checkpoints, prediction folders, detailed
per-image metrics, and third-party repositories are not redistributed.

## Environment

The recorded experiments used Python 3.9, PyTorch 2.1.2 with CUDA 11.8, and
Ultralytics 8.3.9.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Clone the compatible Ultralytics source tree and overlay the files in `model/`:

```bash
git clone --branch v8.3.9 https://github.com/ultralytics/ultralytics.git third_party/ultralytics
cp -R model/ultralytics/. third_party/ultralytics/ultralytics/
python -m pip install -e third_party/ultralytics
```

PowerShell equivalent:

```powershell
git clone --branch v8.3.9 https://github.com/ultralytics/ultralytics.git third_party/ultralytics
Copy-Item -Recurse -Force model\ultralytics\* third_party\ultralytics\ultralytics\
python -m pip install -e third_party/ultralytics
```

## Night-Drone Vehicle

Prepare an Ultralytics detection YAML for the dataset, then run:

```bash
python scripts/run_multiseed_night_drone.py \
  --repo third_party/ultralytics \
  --data data/night-drone/dataset.yaml \
  --output runs/night-drone-multiseed \
  --seeds 0 1 2 --device 0 --epochs 300 --batch 32 --imgsz 640 --workers 4
```

The compact reported-run and mean/standard-deviation tables are under
`results/night_drone/`.

## HIT-UAV four-class plus DontCare protocol

Prepare the dataset from the official HIT-UAV images and JSON annotations:

```bash
python scripts/prepare_hituav_ignore_dataset.py \
  --source data/hit-uav --output data/hit-uav-ignore

python scripts/prepare_hituav_official_fourclass.py \
  --prepared data/hit-uav-ignore \
  --json-root official-hituav \
  --output data/hit-uav-official-ignore
```

Run the comparison launcher:

```bash
python scripts/run_hituav_official_protocol.py \
  --dataset data/hit-uav-official-ignore/dataset_ignore.yaml \
  --output runs/hituav-official-4class --device 0
```

The evaluator removes predictions whose intersection-over-prediction-area is at
least 0.5 against a DontCare region and then computes four-class 101-point AP:

```bash
python scripts/evaluate_hituav_ignore.py \
  --gt-root data/hit-uav-official-ignore \
  --pred-root runs/example/predictions/labels \
  --split test --ignore-ioa 0.5
```

The 13-run aggregate table is `results/hit_uav/fourclass_results.csv`.

## Latency benchmark

```bash
python scripts/benchmark_latency_repeat.py \
  --weights path/to/best.pt --model-name AG-YOLO \
  --output runs/timing/ag-yolo_fp32_bs1.json
```

The benchmark covers the fused PyTorch forward pass only; preprocessing,
decoding, NMS, and disk I/O are excluded. A compact summary is provided in
`results/timing/timing_summary.csv`.

## License

This repository is released under AGPL-3.0. Files derived from Ultralytics retain
the same license obligations. Dataset licenses remain with the original
publishers; see `THIRD_PARTY_NOTICES.md`.
