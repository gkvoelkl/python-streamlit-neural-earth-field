"""Creates the English versions of the tutorial diagrams.

The German SVGs in ``docs/tutorial/img/`` are the originals. This script copies
each one listed in ``svg-en.json`` to ``docs/tutorial/img/en/`` and replaces its
texts: first the exact translations from the JSON file, then decimal commas
(``12,5`` -> ``12.5``) and the compass letter ``O`` (Ost) -> ``E``.

Texts that still contain German umlauts afterwards are reported, so a missing
translation is easy to spot.

    uv run python docs/tutorial/tools/translate_svg.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
IMG = TOOLS.parent / "img"
OUT = IMG / "en"

TEXT = re.compile(r"(<text\b[^>]*>)([^<]*)(</text>)")
ARIA = re.compile(r'aria-label="([^"]*)"')
DECIMAL = re.compile(r"(\d),(\d)")
EAST = re.compile(r"(\d)° O\b")
GERMAN = re.compile(r"[ÄÖÜäöüß]")


def translate(text: str, table: dict[str, str]) -> str:
    plain = html.unescape(text)
    if plain in table:
        return html.escape(table[plain], quote=False)
    plain = EAST.sub(r"\1° E", DECIMAL.sub(r"\1.\2", plain))
    return html.escape(plain, quote=False)


def main() -> None:
    config = json.loads((TOOLS / "svg-en.json").read_text(encoding="utf-8"))
    common: dict[str, str] = config.get("_common", {})
    OUT.mkdir(exist_ok=True)
    for name, own in config.items():
        if name.startswith("_"):
            continue
        table = {**common, **own}
        svg = (IMG / name).read_text(encoding="utf-8")
        svg = TEXT.sub(lambda m: m.group(1) + translate(m.group(2), table) + m.group(3), svg)
        svg = ARIA.sub(lambda m: f'aria-label="{html.escape(translate(m.group(1), table))}"', svg, count=1)
        (OUT / name).write_text(svg, encoding="utf-8")
        left = sorted({html.unescape(t) for _, t, _ in TEXT.findall(svg) if GERMAN.search(html.unescape(t))})
        print(f"{name}: ok" if not left else f"{name}: still German: {left}")


if __name__ == "__main__":
    main()
