from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(r"C:\Users\lenovo\Desktop\AG-YOLO\论文投稿\Figures")
OUTPUT_DIR = Path(r"C:\Users\lenovo\Documents\Codex\2026-10-14\zh\outputs\revised_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def fit_label(draw: ImageDraw.ImageDraw, text: str, box_width: int, max_size: int = 30) -> ImageFont.FreeTypeFont:
    for size in range(max_size, 15, -1):
        candidate = font(size, bold=True)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=candidate)
        if right - left <= box_width:
            return candidate
    return font(16, bold=True)


def reflow(
    source_name: str,
    body_height: int,
    source_rows: int,
    source_columns: int,
    row_labels: list[str],
    left_margin: int,
    top_margin: int,
) -> None:
    source_path = SOURCE_DIR / source_name
    image = Image.open(source_path).convert("RGB")
    body = image.crop((0, 0, image.width, body_height))
    cell_width = image.width // source_columns
    cell_height = body_height // source_rows

    output_width = left_margin + source_rows * cell_width
    output_height = top_margin + source_columns * cell_height
    output = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(output)

    header_font = font(30, bold=True)
    for scene in range(source_rows):
        label = f"Scene {scene + 1}"
        x0 = left_margin + scene * cell_width
        bbox = draw.textbbox((0, 0), label, font=header_font)
        draw.text(
            (x0 + (cell_width - (bbox[2] - bbox[0])) / 2, (top_margin - (bbox[3] - bbox[1])) / 2 - 3),
            label,
            fill="black",
            font=header_font,
        )

    for model in range(source_columns):
        label = row_labels[model]
        y0 = top_margin + model * cell_height
        label_font = fit_label(draw, label, left_margin - 24)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (12, y0 + (cell_height - (bbox[3] - bbox[1])) / 2),
            label,
            fill="black",
            font=label_font,
        )

        for scene in range(source_rows):
            sx0 = model * cell_width
            sx1 = (model + 1) * cell_width if model < source_columns - 1 else body.width
            sy0 = scene * cell_height
            sy1 = (scene + 1) * cell_height if scene < source_rows - 1 else body.height
            panel = body.crop((sx0, sy0, sx1, sy1))
            if panel.size != (cell_width, cell_height):
                panel = panel.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            output.paste(panel, (left_margin + scene * cell_width, y0))

    draw = ImageDraw.Draw(output)
    line_color = (235, 235, 235)
    for c in range(source_rows + 1):
        x = left_margin + c * cell_width
        draw.line((x, top_margin, x, output.height), fill=line_color, width=2)
    for r in range(source_columns + 1):
        y = top_margin + r * cell_height
        draw.line((left_margin, y, output.width, y), fill=line_color, width=2)

    stem = Path(source_name).stem
    output.save(OUTPUT_DIR / f"{stem}_revised.tif", compression="tiff_lzw", dpi=(300, 300))
    output.save(OUTPUT_DIR / f"{stem}_revised.png", dpi=(300, 300), optimize=True)
    print(f"{stem}: {image.size} -> {output.size}")


reflow(
    "Fig9.tif",
    body_height=1148,
    source_rows=5,
    source_columns=9,
    row_labels=[
        "(a) YOLOv5n",
        "(b) YOLOv6n",
        "(c) YOLOv8n",
        "(d) YOLOv9-t",
        "(e) YOLOv10n",
        "(f) YOLOv11n",
        "(g) YOLOv12n",
        "(h) YOLOv13n",
        "(i) AG-YOLO",
    ],
    left_margin=250,
    top_margin=72,
)

reflow(
    "Fig11.tif",
    body_height=1480,
    source_rows=5,
    source_columns=6,
    row_labels=[
        "(a) Input image",
        "(b) Baseline",
        "(c) Baseline + MANet",
        "(d) Baseline + C3k2_FCA_CTA",
        "(e) Baseline + C2CGA",
        "(f) AG-YOLO",
    ],
    left_margin=360,
    top_margin=80,
)
