from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(r"C:\Users\lenovo\Documents\Codex\2026-10-14\zh\outputs\revised_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "conv": "#E8EEF7",
    "channel": "#CFE8FF",
    "manet": "#FFE5B4",
    "context": "#D9F0D3",
    "fusion": "#FCE1D0",
    "detect": "#E1F2C4",
    "neutral": "#F4F4F4",
    "accent": "#63D7E2",
    "attention": "#FFF1A8",
    "line": "#202020",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def canvas(figsize=(10, 6), xlim=(0, 12), ylim=(0, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="white", ec=COLORS["line"], lw=1.15, fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(ax, start, end, color=COLORS["line"], lw=1.2, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        shrinkA=1.5,
        shrinkB=1.5,
    )
    ax.add_patch(patch)
    return patch


def group(ax, x, y, w, h, title, ec="#6F6F6F"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="none",
        edgecolor=ec,
        linewidth=1.0,
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h - 0.18, title, ha="left", va="top", fontsize=10, weight="bold", color=ec)


def save(fig, stem):
    for extension, kwargs in {
        "svg": {},
        "pdf": {},
        "tif": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(OUTPUT_DIR / f"{stem}_revised.{extension}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def figure1():
    fig, ax = canvas((10.2, 7.25), (0, 13), (0, 10.8))
    group(ax, 0.15, 0.35, 3.25, 9.25, "Backbone")
    group(ax, 3.65, 0.35, 6.2, 9.25, "FPN--PAN neck")
    group(ax, 10.1, 0.35, 2.7, 9.25, "Detection")

    backbone = [
        ("Input\n640$\\times$640", COLORS["neutral"]),
        ("Conv /2", COLORS["conv"]),
        ("C3k2_FCA_CTA\nP2/4", COLORS["channel"]),
        ("C3k2_FCA_CTA\nP3/8", COLORS["channel"]),
        ("MANet\nP4/16", COLORS["manet"]),
        ("MANet\nP5/32", COLORS["manet"]),
        ("SPPF", COLORS["attention"]),
        ("C2CGA", COLORS["context"]),
    ]
    ys = [8.75, 7.70, 6.65, 5.60, 4.55, 3.50, 2.45, 1.40]
    for (label, color), y in zip(backbone, ys):
        box(ax, 0.72, y, 2.1, 0.62, label, fc=color, fontsize=9.5, weight="bold" if "CTA" in label or "MANet" in label or "C2CGA" in label else "normal")
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (1.77, y1), (1.77, y2 + 0.62))

    box(ax, 4.05, 1.40, 1.65, 0.62, "Upsample +\nConcat P4", fc=COLORS["fusion"], fontsize=9)
    box(ax, 6.00, 2.65, 1.65, 0.62, "MANet\nTD-P4", fc=COLORS["manet"], fontsize=9.5, weight="bold")
    box(ax, 4.05, 4.75, 1.65, 0.62, "Upsample +\nConcat P3", fc=COLORS["fusion"], fontsize=9)
    box(ax, 6.00, 5.95, 1.65, 0.62, "MANet\nTD-P3", fc=COLORS["manet"], fontsize=9.5, weight="bold")
    box(ax, 7.95, 4.35, 1.65, 0.62, "Downsample +\nConcat P4", fc=COLORS["fusion"], fontsize=9)
    box(ax, 7.95, 3.20, 1.65, 0.62, "MANet\nBU-P4", fc=COLORS["manet"], fontsize=9.5, weight="bold")
    box(ax, 7.95, 1.55, 1.65, 0.62, "Downsample +\nConcat P5", fc=COLORS["fusion"], fontsize=9)
    box(ax, 7.95, 0.55, 1.65, 0.62, "MANet\nBU-P5", fc=COLORS["manet"], fontsize=9.5, weight="bold")

    arrow(ax, (2.82, 1.71), (4.05, 1.71))
    arrow(ax, (5.70, 1.71), (6.00, 2.96))
    arrow(ax, (6.83, 3.27), (4.88, 4.75))
    arrow(ax, (5.70, 5.06), (6.00, 6.26))
    arrow(ax, (7.65, 6.26), (8.78, 4.97))
    arrow(ax, (8.78, 4.35), (8.78, 3.82))
    arrow(ax, (8.78, 3.20), (8.78, 2.17))
    arrow(ax, (8.78, 1.55), (8.78, 1.17))

    # Lateral feature connections.
    arrow(ax, (2.82, 4.86), (4.05, 5.06), color="#555555", connection="angle3,angleA=0,angleB=90")
    arrow(ax, (2.82, 5.91), (4.05, 5.06), color="#555555", connection="angle3,angleA=0,angleB=-90")
    arrow(ax, (2.82, 4.86), (7.95, 4.66), color="#777777", connection="angle3,angleA=0,angleB=90")
    arrow(ax, (2.82, 1.71), (7.95, 1.86), color="#777777", connection="angle3,angleA=0,angleB=90")

    for y, level in [(6.26, "P3/8"), (3.51, "P4/16"), (0.86, "P5/32")]:
        box(ax, 10.55, y - 0.31, 1.75, 0.62, f"Detect {level}", fc=COLORS["detect"], fontsize=10, weight="bold")
    arrow(ax, (7.65, 6.26), (10.55, 6.26))
    arrow(ax, (9.60, 3.51), (10.55, 3.51))
    arrow(ax, (9.60, 0.86), (10.55, 0.86))

    ax.text(6.5, 10.55, "AG-YOLO stage-specific architecture", ha="center", va="top", fontsize=15, weight="bold")
    ax.text(1.78, 0.48, "High-resolution channel enhancement", ha="center", fontsize=8.5, color="#235E91")
    ax.text(6.5, 10.05, "MANet is placed at selected multiscale fusion stages", ha="center", fontsize=9.5, color="#8A5600")
    save(fig, "Fig1")


def figure2():
    fig, ax = canvas((10.0, 5.6), (0, 12), (0, 7))
    ax.text(6, 6.7, "MANet: mixed aggregation of parallel and hierarchical features", ha="center", va="top", fontsize=14, weight="bold")
    box(ax, 0.4, 3.0, 1.1, 0.62, "Input $X$", fc=COLORS["neutral"], weight="bold")
    box(ax, 1.9, 3.0, 1.35, 0.62, "1$\\times$1 Conv", fc=COLORS["conv"])
    ax.text(2.58, 2.73, "$Y$", ha="center", fontsize=10)
    arrow(ax, (1.5, 3.31), (1.9, 3.31))

    branch_y = [5.25, 4.10, 2.70, 1.30]
    for y in branch_y:
        arrow(ax, (3.25, 3.31), (4.05, y + 0.31), color="#555555", connection="angle3,angleA=0,angleB=90")

    box(ax, 4.05, 5.25, 1.35, 0.62, "1$\\times$1 Conv", fc=COLORS["conv"])
    ax.text(4.72, 6.04, "Channel mapping", ha="center", va="bottom", fontsize=9.1, weight="bold")
    ax.text(5.62, 5.56, "$Y_0$", fontsize=10, weight="bold")

    box(ax, 4.05, 4.10, 1.25, 0.62, "1$\\times$1 Conv", fc=COLORS["conv"], fontsize=9)
    box(ax, 5.60, 4.10, 1.35, 0.62, "DWConv $k$", fc=COLORS["attention"], fontsize=9)
    box(ax, 7.25, 4.10, 1.25, 0.62, "1$\\times$1 Conv", fc=COLORS["conv"], fontsize=9)
    arrow(ax, (5.30, 4.41), (5.60, 4.41))
    arrow(ax, (6.95, 4.41), (7.25, 4.41))
    ax.text(6.28, 4.92, "Local filtering", ha="center", va="bottom", fontsize=9.1, weight="bold")
    ax.text(8.67, 4.41, "$Y_1$", va="center", fontsize=10, weight="bold")

    box(ax, 4.05, 2.70, 1.15, 0.62, "Split", fc=COLORS["fusion"])
    ax.text(4.62, 3.50, "Split bypass", ha="center", va="bottom", fontsize=9.1, weight="bold")
    ax.text(5.40, 3.01, "$Y_2$", va="center", fontsize=10, weight="bold")

    box(ax, 4.05, 1.30, 1.15, 0.62, "Split", fc=COLORS["fusion"])
    box(ax, 5.55, 1.30, 1.15, 0.62, "$B_1$", fc=COLORS["manet"])
    box(ax, 7.05, 1.30, 1.15, 0.62, "$B_2$", fc=COLORS["manet"])
    ax.text(8.62, 1.61, "$\\cdots$", ha="center", va="center", fontsize=18)
    box(ax, 9.05, 1.30, 1.15, 0.62, "$B_n$", fc=COLORS["manet"])
    ax.text(7.15, 0.78, "Hierarchical bottlenecks", ha="center", va="top", fontsize=9.1, weight="bold")
    arrow(ax, (5.20, 1.61), (5.55, 1.61))
    arrow(ax, (6.70, 1.61), (7.05, 1.61))
    arrow(ax, (8.20, 1.61), (9.05, 1.61))

    box(ax, 9.05, 3.00, 1.35, 0.62, "Concat", fc=COLORS["fusion"], weight="bold")
    for start in [(5.40, 5.56), (8.50, 4.41), (5.20, 3.01), (10.20, 1.61)]:
        arrow(ax, start, (9.05, 3.31), color="#555555", connection="angle3,angleA=0,angleB=90")
    box(ax, 10.65, 3.00, 1.05, 0.62, "1$\\times$1\nConv", fc=COLORS["conv"], fontsize=9)
    arrow(ax, (10.40, 3.31), (10.65, 3.31))
    ax.text(11.82, 3.31, "$Y_{out}$", va="center", fontsize=10, weight="bold")
    save(fig, "Fig2")


def figure3():
    fig, ax = canvas((11.0, 6.2), (0, 14), (0, 8))
    ax.text(7, 7.72, "C3k2_FCA_CTA", ha="center", va="top", fontsize=15, weight="bold", color="#008FC7")
    group(ax, 0.3, 5.15, 13.4, 2.05, "CSP wrapper")
    box(ax, 0.65, 5.72, 1.0, 0.62, "Input", fc=COLORS["neutral"])
    box(ax, 2.0, 5.72, 1.1, 0.62, "Conv", fc=COLORS["conv"])
    box(ax, 3.45, 5.72, 1.1, 0.62, "Split", fc=COLORS["fusion"])
    box(ax, 6.05, 5.72, 1.55, 0.62, "FCA_CTA", fc=COLORS["accent"], weight="bold")
    box(ax, 8.35, 5.72, 1.2, 0.62, "Concat", fc=COLORS["fusion"])
    box(ax, 10.1, 5.72, 1.1, 0.62, "Conv", fc=COLORS["conv"])
    box(ax, 11.65, 5.72, 1.0, 0.62, "Output", fc=COLORS["neutral"])
    for x1, x2 in [(1.65, 2.0), (3.1, 3.45), (4.55, 6.05), (7.60, 8.35), (9.55, 10.1), (11.2, 11.65)]:
        arrow(ax, (x1, 6.03), (x2, 6.03))
    arrow(ax, (4.0, 6.34), (8.95, 6.34), color="#555555", connection="angle3,angleA=0,angleB=90")

    group(ax, 0.3, 0.35, 13.4, 4.45, "FCA_CTA residual unit")
    box(ax, 0.65, 2.25, 0.9, 0.60, "$X$", fc=COLORS["neutral"])
    box(ax, 1.85, 2.25, 0.9, 0.60, "LN", fc="#FFD6D6")
    box(ax, 3.05, 2.25, 1.0, 0.60, "Linear", fc="#FFD6D6")
    arrow(ax, (1.55, 2.55), (1.85, 2.55))
    arrow(ax, (2.75, 2.55), (3.05, 2.55))

    box(ax, 4.55, 3.35, 1.65, 0.66, "Channel\nself-attention", fc=COLORS["attention"], fontsize=9.5)
    box(ax, 4.55, 1.25, 1.65, 0.66, "1$\\times$1 Conv +\n3$\\times$3 DWConv", fc="#FFC7C7", fontsize=9.2)
    box(ax, 6.70, 3.35, 1.55, 0.66, "Channel\nprojection", fc=COLORS["channel"], fontsize=9.5)
    box(ax, 6.70, 1.25, 1.55, 0.66, "Spatial\nprojection", fc="#FFD45C", fontsize=9.5)
    box(ax, 8.75, 2.25, 1.25, 0.66, "Cross-gated\nfusion", fc=COLORS["accent"], fontsize=9.5, weight="bold")
    box(ax, 10.45, 2.25, 1.0, 0.66, "Linear", fc="#FFD6D6")
    box(ax, 11.85, 2.25, 0.85, 0.66, "+", fc=COLORS["neutral"], fontsize=15)
    for start, end in [
        ((4.05, 2.55), (4.55, 3.68)),
        ((4.05, 2.55), (4.55, 1.58)),
        ((6.20, 3.68), (6.70, 3.68)),
        ((6.20, 1.58), (6.70, 1.58)),
        ((8.25, 3.68), (8.75, 2.70)),
        ((8.25, 1.58), (8.75, 2.45)),
        ((10.0, 2.58), (10.45, 2.58)),
        ((11.45, 2.58), (11.85, 2.58)),
    ]:
        arrow(ax, start, end)
    arrow(ax, (1.10, 2.25), (12.28, 2.25), color="#777777", connection="angle3,angleA=0,angleB=90")
    box(ax, 12.88, 2.22, 0.72, 0.72, "LN +\nDFFN +", fc=COLORS["context"], fontsize=8.0)
    arrow(ax, (12.70, 2.58), (12.88, 2.58))
    ax.text(13.72, 2.58, "$Z_{out}$", va="center", fontsize=10, weight="bold")
    save(fig, "Fig3")


def figure4():
    fig, ax = canvas((10.0, 3.2), (0, 12), (0, 4))
    ax.text(6, 3.72, "C2CGA split--transform--fuse structure", ha="center", va="top", fontsize=14, weight="bold")
    box(ax, 0.45, 1.55, 1.1, 0.68, "Input", fc=COLORS["neutral"], weight="bold")
    box(ax, 1.95, 1.55, 1.25, 0.68, "1$\\times$1 Conv", fc=COLORS["conv"])
    box(ax, 3.60, 1.55, 1.05, 0.68, "Split", fc=COLORS["fusion"])
    box(ax, 5.15, 2.45, 1.45, 0.68, "Bypass $a$", fc=COLORS["neutral"])
    box(ax, 5.15, 0.65, 1.45, 0.68, "CGA Block 1", fc=COLORS["context"])
    box(ax, 7.10, 0.65, 1.45, 0.68, "CGA Block $N$", fc=COLORS["context"])
    ax.text(6.85, 0.99, "$\\cdots$", ha="center", va="center", fontsize=18)
    box(ax, 8.95, 1.55, 1.15, 0.68, "Concat", fc=COLORS["fusion"], weight="bold")
    box(ax, 10.45, 1.55, 1.15, 0.68, "1$\\times$1 Conv", fc=COLORS["conv"], fontsize=9)
    for start, end in [
        ((1.55, 1.89), (1.95, 1.89)),
        ((3.20, 1.89), (3.60, 1.89)),
        ((4.65, 2.05), (5.15, 2.79)),
        ((4.65, 1.73), (5.15, 0.99)),
        ((6.60, 0.99), (7.10, 0.99)),
        ((8.55, 0.99), (8.95, 1.73)),
        ((6.60, 2.79), (8.95, 2.05)),
        ((10.10, 1.89), (10.45, 1.89)),
    ]:
        arrow(ax, start, end)
    ax.text(11.78, 1.89, "Output", va="center", fontsize=10, weight="bold")
    save(fig, "Fig4")


def figure5():
    fig, ax = canvas((10.0, 5.4), (0, 12), (0, 7))
    ax.text(6, 6.68, "Cascaded group attention within one local window", ha="center", va="top", fontsize=14, weight="bold")
    box(ax, 0.35, 2.95, 1.05, 0.70, "Input\nwindow", fc=COLORS["neutral"], fontsize=9.5, weight="bold")
    box(ax, 1.85, 2.95, 1.15, 0.70, "Channel\nsplit", fc=COLORS["fusion"], fontsize=9.5)
    arrow(ax, (1.40, 3.30), (1.85, 3.30))

    ys = [5.0, 3.0, 1.0]
    names = ["Group 1", "Group 2", "Group $h$"]
    for y, name in zip(ys, names):
        box(ax, 3.55, y, 1.1, 0.64, name, fc=COLORS["channel"], fontsize=9.5)
        box(ax, 5.05, y, 1.5, 0.64, "Q, K, V\nprojections", fc=COLORS["conv"], fontsize=9)
        box(ax, 6.95, y, 1.25, 0.64, "DWConv\non Q", fc=COLORS["attention"], fontsize=9)
        box(ax, 8.60, y, 1.45, 0.64, "Scaled dot-\nproduct attention", fc=COLORS["context"], fontsize=8.8)
        arrow(ax, (4.65, y + 0.32), (5.05, y + 0.32))
        arrow(ax, (6.55, y + 0.32), (6.95, y + 0.32))
        arrow(ax, (8.20, y + 0.32), (8.60, y + 0.32))
        arrow(ax, (3.00, 3.30), (3.55, y + 0.32), color="#555555", connection="angle3,angleA=0,angleB=90")

    ax.text(5.95, 2.15, "$\\vdots$", ha="center", va="center", fontsize=20)
    arrow(ax, (10.05, 5.32), (3.55, 3.32), color="#C27500", lw=1.5, connection="arc3,rad=-0.20")
    arrow(ax, (10.05, 3.32), (3.55, 1.32), color="#C27500", lw=1.5, connection="arc3,rad=-0.20")
    ax.text(7.4, 4.30, "previous group output", fontsize=8.8, color="#9A5A00", rotation=-11)
    ax.text(7.4, 2.30, "previous group output", fontsize=8.8, color="#9A5A00", rotation=-11)

    box(ax, 10.55, 2.95, 1.05, 0.70, "Concat +\nprojection", fc=COLORS["fusion"], fontsize=9.2, weight="bold")
    for y in ys:
        arrow(ax, (10.05, y + 0.32), (10.55, 3.30), color="#555555", connection="angle3,angleA=0,angleB=90")
    ax.text(11.76, 3.30, "Output", va="center", fontsize=10, weight="bold")
    save(fig, "Fig5")


figure1()
figure2()
figure3()
figure4()
figure5()
print("Generated vector and raster exports for Fig1--Fig5.")
