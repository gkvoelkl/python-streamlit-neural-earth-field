"""Creates English versions of tutorial figures that only exist as PNG.

Some figures were drawn with matplotlib in earlier sessions and only the PNG is
left. This script replaces their texts instead of redrawing them: for each text
listed in ``png-en.json`` it finds the ink inside the given box, works out the
font size by setting the German original in DejaVu Sans (matplotlib's default
font) and comparing it with the ink, paints the old text over with the
background colour and writes the English text on the same baseline.

    uv run --with matplotlib python docs/tutorial/tools/translate_png.py

Each entry: ``box`` [x0, y0, x1, y1] around one text (generous, but not touching
other ink), ``de`` the German text, ``en`` the replacement, optional ``align``
(left, centre, right), ``bold``, ``rotate`` (90 for text that reads upwards) and
``plate`` (paint a light box behind the text, for labels that sit on an image).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOOLS = Path(__file__).resolve().parent
IMG = TOOLS.parent / "img"
FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def ink_box(text: str, size: int, bold: bool) -> tuple[int, int, int, int]:
    """Bounding box of the ink when ``text`` is drawn at the origin (left, top, right, bottom)."""
    return font(size, bold).getbbox(text)


def background(region: np.ndarray) -> np.ndarray:
    border = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]])
    values, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    return values[counts.argmax()]


def replace(img: Image.Image, item: dict) -> Image.Image:
    rotate = item.get("rotate", 0)
    x0, y0, x1, y1 = item["box"]
    crop = img.crop((x0, y0, x1, y1))
    if rotate:
        crop = crop.rotate(-rotate, expand=True)
    a = np.asarray(crop.convert("RGB")).astype(int)
    bg = background(a)
    dist = np.abs(a - bg).sum(axis=2)
    ys, xs = np.where(dist > 60)
    if len(xs) == 0:
        raise ValueError(f"no ink found for {item['de']!r}")
    L, T, R, B = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    strong = a[ys, xs][dist[ys, xs] >= np.percentile(dist[ys, xs], 80)]
    ink = tuple(int(v) for v in np.median(strong, axis=0))
    bold = item.get("bold", False)

    # font size: the German text set in DejaVu should cover the same ink width and height
    best, size = None, item.get("size", 10)
    for s in ([item["size"]] if "size" in item else range(6, 80)):
        l, t, r, b = ink_box(item["de"], s, bold)
        score = abs((r - l) - (R - L)) / (R - L) + abs((b - t) - (B - T)) / (B - T)
        if best is None or score < best:
            best, size = score, s
    l_de, t_de, r_de, _ = ink_box(item["de"], size, bold)
    l_en, _, r_en, _ = ink_box(item["en"], size, bold)
    oy = T - t_de  # same baseline as the German text
    align = item.get("align", "left")
    if align == "centre":
        ox = (L + R) / 2 - (l_en + r_en) / 2
    elif align == "right":
        ox = R - r_en
    else:
        ox = L - l_de

    if os.environ.get("VERBOSE"):
        print(f"  {item['de'][:30]!r}: ink {L},{T}-{R},{B} size {size}")
    if rotate:  # text reading upwards: work in the rotated crop and paste it back
        out = Image.fromarray(a.astype(np.uint8))
        draw = ImageDraw.Draw(out)
        draw.rectangle((L - 2, T - 2, R + 2, B + 2), fill=tuple(int(v) for v in bg))
        draw.text((ox, oy), item["en"], font=font(size, bold), fill=ink)
        img = img.copy()
        img.paste(out.rotate(rotate, expand=True), (x0, y0))
        return img
    # otherwise draw on the whole figure, so a longer English text may leave the box
    img = img.copy()
    draw = ImageDraw.Draw(img)
    L, R, ox = L + x0, R + x0, ox + x0
    T, B, oy = T + y0, B + y0, oy + y0
    if item.get("plate"):
        pad = item.get("pad", 4)
        draw.rounded_rectangle((min(L, ox + l_en) - pad, T - pad, max(R, ox + r_en) + pad, B + pad),
                               radius=5, fill=(255, 255, 255))
        ink = tuple(item.get("colour", (31, 41, 51)))
    else:
        draw.rectangle((L - 2, T - 2, R + 2, B + 2), fill=tuple(int(v) for v in bg))
    draw.text((ox, oy), item["en"], font=font(size, bold), fill=ink)
    return img


def main() -> None:
    spec = json.loads((TOOLS / "png-en.json").read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    (IMG / "en").mkdir(exist_ok=True)
    for name, items in spec.items():
        if only and name not in only:
            continue
        img = Image.open(IMG / name).convert("RGB")
        for item in items:
            img = replace(img, item)
        img.save(IMG / "en" / name, optimize=True)
        print(f"{name}: {len(items)} texts")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
