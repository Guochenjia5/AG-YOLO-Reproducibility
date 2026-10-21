from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


RUNS_DIR = Path(r"D:\gcj\hituav-official-4class\train")
OUTPUT_DIR = Path(
    r"C:\Users\lenovo\Documents\Codex\2026-10-14\zh\outputs\revised_figures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("YOLOv5n", "yolov5n_seed0"),
    ("YOLOv6n", "yolov6n_seed0"),
    ("YOLOv8n", "yolov8n_seed0"),
    ("YOLOv9-t", "yolov9t_seed0"),
    ("YOLOv10n", "yolov10n_seed0"),
    ("YOLOv11n", "yolov11n_seed0"),
    ("YOLOv12n", "yolov12n_seed0"),
    ("YOLOv13n", "yolov13n_seed0"),
    ("AG-YOLO", "ag_yolo_seed0"),
]

METRICS = [
    ("metrics/precision(B)", "Precision", (0.0, 1.0)),
    ("metrics/recall(B)", "Recall", (0.0, 1.0)),
    ("metrics/mAP50(B)", "mAP@0.5", (0.0, 1.0)),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95", (0.0, 0.70)),
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def load_results(run_name: str) -> pd.DataFrame:
    path = RUNS_DIR / run_name / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    if len(frame) != 300:
        raise ValueError(f"Expected 300 epochs for {run_name}, found {len(frame)}")
    required = {"epoch", *(column for column, _, _ in METRICS)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for {run_name}: {missing}")
    return frame


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2), constrained_layout=False)
    axes = axes.ravel()
    handles = []
    labels = []

    for index, (label, run_name) in enumerate(RUNS):
        frame = load_results(run_name)
        epoch = frame["epoch"]
        color = COLORS[index]
        linewidth = 2.5 if label == "AG-YOLO" else (2.0 if label == "YOLOv11n" else 1.35)
        zorder = 5 if label == "AG-YOLO" else (4 if label == "YOLOv11n" else 2)
        alpha = 1.0 if label in {"AG-YOLO", "YOLOv11n"} else 0.88

        for axis, (column, title, limits) in zip(axes, METRICS):
            line = axis.plot(
                epoch,
                frame[column],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=zorder,
            )[0]
            axis.set_title(title, pad=7, weight="bold")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Validation score")
            axis.set_xlim(float(epoch.min()), float(epoch.max()))
            axis.set_ylim(*limits)
            axis.grid(True, color="#D9D9D9", linewidth=0.65, linestyle="--", alpha=0.8)
            axis.tick_params(direction="out", length=3.5, width=0.8)

        handles.append(line)
        labels.append(label)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.205, hspace=0.34, wspace=0.26)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.5,
        handlelength=2.6,
    )

    stem = OUTPUT_DIR / "Fig12_HITUAV_training_curves_revised"
    exports = {
        "svg": {},
        "pdf": {},
        "tif": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "png": {"dpi": 300},
    }
    for extension, options in exports.items():
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **options)
    plt.close(fig)
    print(f"Generated HIT-UAV seed-0 training curves at {stem}.*")


if __name__ == "__main__":
    main()
