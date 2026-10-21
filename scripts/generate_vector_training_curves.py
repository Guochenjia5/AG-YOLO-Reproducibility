from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(r"C:\Users\lenovo\Documents\Codex\2026-10-14\zh\outputs\revised_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN = Path(r"D:\gcj\ultralytics-yolo11-main")
HIST = Path(r"D:\gcj\ultralytics-yolo11\ultralytics-yolo11-main")

METRICS = [
    ("metrics/precision(B)", "precision"),
    ("metrics/recall(B)", "recall"),
    ("metrics/mAP50(B)", "mAP_0.5"),
    ("metrics/mAP50-95(B)", "mAP_0.5:0.95"),
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
    "#17becf",
    "#000000",
]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def plot_curves(
    name: str,
    runs: list[tuple[str, Path]],
    *,
    shared_legend: bool = False,
    legends_below_axes: bool = False,
    legend_columns: int = 2,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_height = 12.0 if legends_below_axes else (10.8 if shared_legend else 10.0)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, figure_height))
    axes = axes.ravel()

    for run_index, (label, path) in enumerate(runs):
        frame = load(path)
        epoch = frame["epoch"]
        color = COLORS[run_index % len(COLORS)]
        specialist = "reimpl." in label
        for axis, (column, title) in zip(axes, METRICS):
            axis.plot(
                epoch,
                frame[column],
                color=color,
                linewidth=1.8 if specialist else 1.5,
                linestyle="--" if specialist else "-",
                label=label,
            )

    for axis, (_, title) in zip(axes, METRICS):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, linestyle="--", alpha=0.35)
        if legends_below_axes:
            # Long ablation labels need a dedicated band.  A fixed-width
            # legend below each axis keeps every box inside its subplot width
            # and leaves the data area unobstructed.
            axis.legend(
                loc="upper left",
                bbox_to_anchor=(0.0, -0.16, 1.0, 0.0),
                mode="expand",
                borderaxespad=0.0,
                fontsize=6.2,
                ncol=legend_columns,
                frameon=True,
                handlelength=1.6,
                handletextpad=0.45,
                columnspacing=0.8,
                labelspacing=0.3,
                borderpad=0.35,
            )
        elif not shared_legend:
            axis.legend(loc="lower right", fontsize=7.5, ncol=2)

    if shared_legend:
        # Figure 10 has long ablation labels. A single legend in a reserved
        # strip prevents the box from covering curves or crossing an axis.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=legend_columns,
            fontsize=8.5,
            frameon=True,
            handlelength=2.7,
            columnspacing=1.8,
        )
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    elif legends_below_axes:
        fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.085, hspace=0.52, wspace=0.18)
    else:
        fig.tight_layout()

    for extension, kwargs in {
        "svg": {},
        "pdf": {},
        "tif": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(OUTPUT_DIR / f"{name}_revised.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


figure8_runs = [
    ("YOLOv5n", HIST / r"runs\train\An guang(v5)\results.csv"),
    ("YOLOv6n", HIST / r"runs\train\An guang(v6)\results.csv"),
    ("YOLOv8n", HIST / r"runs\train\An guang(v8)\results.csv"),
    ("YOLOv9-t", HIST / r"runs\train\An guang(v9t)\results.csv"),
    ("YOLOv10n", HIST / r"runs\train\An guang(v10)\results.csv"),
    ("YOLOv11n", MAIN / r"runs\train\An guang（yolo11）2\results.csv"),
    ("YOLOv12n", MAIN / r"runs\train\An guang（yolo12）\results.csv"),
    ("YOLOv13n", MAIN / r"runs\train\An guang（yolo13）\results.csv"),
    ("AG-YOLO", MAIN / r"runs\An guang\train\yolo11-MAN-CTA-C2CGA\results.csv"),
    (
        "YOLO-CIR (reimpl.)",
        Path(r"D:\gcj\night-drone-yolo-cir-reimpl-b8-fp32\yolo_cir_reimpl_seed0\results.csv"),
    ),
    (
        "YOLO-HVS (reimpl.)",
        Path(r"D:\gcj\night-drone-yolo-hvs-reimpl-b32-fp32\yolo_hvs_reimpl_seed0\results.csv"),
    ),
]

figure10_runs = [
    ("YOLOv11n", MAIN / r"runs\train\An guang（yolo11）2\results.csv"),
    ("YOLOv11n+MANet", MAIN / r"runs\Anguang\train\An guang（yolo11-MAN）\results.csv"),
    ("YOLOv11n+C3k2_FCA_CTA", MAIN / r"runs\train\An guang（yolo11-C3k2-CTA）\results.csv"),
    ("YOLOv11n+C2CGA", MAIN / r"runs\train\An guang（yolo11-C2CGA）\results.csv"),
    ("YOLOv11n+MANet+C3k2_FCA_CTA", MAIN / r"runs\An guang\train\yolo11-MAN-CTA6\results.csv"),
    ("YOLOv11n+MANet+C2CGA", MAIN / r"runs\Anguang\train\An guang（yolo11-MAN-C2CGA）2\results.csv"),
    ("YOLOv11n+C3k2_FCA_CTA+C2CGA", MAIN / r"runs\An guang\train\yolo11-CTA-C2CGA\results.csv"),
    ("AG-YOLO", MAIN / r"runs\An guang\train\yolo11-MAN-CTA-C2CGA\results.csv"),
]

plot_curves("Fig8", figure8_runs)
plot_curves("Fig10", figure10_runs, legends_below_axes=True, legend_columns=2)
print("Generated vector and raster exports for Fig8 and Fig10.")
