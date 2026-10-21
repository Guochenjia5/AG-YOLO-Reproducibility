# AG-YOLO revision experiment helpers

These files address evidence that was absent from the archived project. They do
not create replacement numbers without running the required experiments.

## 1. Paired multi-seed Night-Drone Vehicle runs

Run from the AG-YOLO repository with the CUDA environment:

```powershell
$env:PYTHONPATH='D:\gcj\ultralytics-yolo11-main'
& 'D:\Anaconda3\envs\learn_cuda\python.exe' `
  '.\run_multiseed_night_drone.py' --seeds 0 1 2 --device 0
```

The script reruns both models with identical optimizer, AMP, worker, early-stop,
input-size, and validation settings. For consistency with the original
manuscript tables, the reported revision retains the archived Table 7 seed-0
values and uses only seeds 1 and 2 from the new launcher. Generate that explicit
reporting view with:

```powershell
& 'D:\Anaconda3\envs\learn_cuda\python.exe' `
  '.\combine_nightdrone_reported_runs.py' `
  --additional-results 'D:\gcj\night-drone-revision-multiseed\paired_results.csv' `
  --output-dir '.\'
```

The resulting `nightdrone_reported_runs.csv` labels the source of every row,
and `nightdrone_reported_summary.csv` contains the manuscript means and sample
standard deviations. Because seed 0 came from the archived manuscript run while
seeds 1 and 2 came from the additional launcher, the dispersion can include
minor configuration effects as well as random-seed variation. The manuscript
and response letter state this limitation explicitly.

## 2. HIT-UAV four-class plus DontCare workflow

For the complete author-published split, first download the official JSON files
and run:

```powershell
D:\Anaconda3\envs\learn_cuda\python.exe prepare_hituav_official_fourclass.py `
  --prepared 'D:\gcj\hit-uav-ignore' `
  --json-root '..\official-hituav' `
  --output 'D:\gcj\hit-uav-official-ignore'

D:\Anaconda3\envs\learn_cuda\python.exe audit_hituav_official_protocol.py `
  --dataset 'D:\gcj\hit-uav-official-ignore'
```

The builder adds only the 32 zero-annotation images absent from the local YOLO
conversion and records the official JSON SHA-256 values. The audit verifies
every class and box coordinate against the original JSON files. The completed
audit covered all 2,898 images and 24,899 annotations, found zero mismatches,
and recorded a maximum normalized-coordinate error of 5.0e-9. Run all 13
manuscript comparisons with:

```powershell
D:\Anaconda3\envs\learn_cuda\python.exe run_hituav_official_protocol.py `
  --dataset 'D:\gcj\hit-uav-official-ignore\dataset_ignore.yaml' `
  --output 'D:\gcj\hituav-official-4class' --device 0
```

Prepare a separate dataset view:

```powershell
python .\prepare_hituav_ignore_dataset.py `
  --source 'D:\gcj\hit-uav' --output 'D:\gcj\hit-uav-ignore'
```

Train against `D:\gcj\hit-uav-ignore\dataset_ignore.yaml`. Generate normalized
prediction text files with confidence scores, one file per evaluated image, then
evaluate them:

```powershell
python .\evaluate_hituav_ignore.py `
  --gt-root 'D:\gcj\hit-uav-ignore' `
  --pred-root 'D:\path\to\predictions\labels' `
  --split test --ignore-ioa 0.5
```

The evaluator removes detections whose intersection divided by prediction area
is at least 0.5 against any DontCare box, then reports four-class 101-point AP at
IoU 0.50 and at 0.50:0.95. This rule is explicit and class agnostic. The
included `official_protocol_manifest.json` records the official JSON hashes,
and `official_protocol_audit.json` records the passed coordinate-level audit.
The HIT-UAV repository's `AP.py` is marked unfinished, so the manuscript names
this explicit reviewer-aligned evaluator instead of claiming output from a
completed author-supplied reference evaluator.

All 13 manuscript runs reached 300 epochs. The package includes the aggregate
`fourclass_results.csv`, per-run metric JSON files, and per-class CSV files.
Large checkpoints and 579-image prediction-label directories are retained in
the experiment archive and are intentionally not duplicated in this compact
review package.

## 3. Timing

`benchmark_latency_repeat.py` and the saved JSON/CSV files use batch 1, FP32,
640 by 640 synthetic inputs, 200 warm-up iterations, and five repetitions of
1,000 synchronized forward passes. Disk I/O, preprocessing, decoding, and NMS
are excluded.

## 4. Model files and package layout

The `model/` directory contains the exact AG-YOLO YAML and the modified
Ultralytics 8.3.9 parser and extra-module files used by the experiments. The
`scripts/` directory contains the launch, conversion, audit, evaluation,
timing, and figure-generation helpers. The `results/` directory contains the
compact CSV/JSON evidence used in the revised manuscript. Dataset images,
large checkpoints, and prediction-label directories are excluded from this
review package; their locations and completion checks are recorded in the
result manifests and manuscript evidence log.

## 5. Night-Drone Vehicle specialist applicability tests

The revised manuscript includes explicitly labeled independent reimplementations
of YOLO-CIR and YOLO-HVS. These are paper-guided, same-data applicability tests,
not author-official implementations or exact reproductions. No executable author
code or weights were available for either method.

Run the two launchers from the compatible Ultralytics source tree:

```powershell
python scripts/run_yolo_cir_night_drone.py --output runs/yolo-cir --seeds 0 `
  --device 0 --epochs 300 --batch 8 --imgsz 640 --workers 4 --no-amp

python scripts/run_yolo_hvs_night_drone.py --output runs/yolo-hvs --seeds 0 `
  --device 0 --epochs 300 --batch 32 --imgsz 640 --workers 4 --no-amp
```

Table 7 uses an independent validation pass on each final-epoch `last.pt`
checkpoint. The saved `final_epoch_results.csv` files report YOLO-CIR at
0.7814 mAP@0.5 and 0.5931 mAP@0.5:0.95, and YOLO-HVS at 0.7881 and 0.6008.
The original best-checkpoint validation summaries are retained separately for
provenance and are not the values used in Table 7. YOLO-CIR uses physical batch
8 because larger FP32 batches were not stable on the available Windows GPU
memory-management path; YOLO-HVS uses physical batch 32. The protocol and audit
JSON files record initialization choices, architectural ambiguities, source
paper DOIs, and model-complexity measurements.

The per-epoch CSV files are included so that Figure 8 can be regenerated with
the updated `generate_vector_training_curves.py` script. The two specialist
curves are drawn with dashed lines in the manuscript figure.
