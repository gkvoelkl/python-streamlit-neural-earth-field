"""Figure for chapter 4: the seam at the date line, with lon/lat and with xyz input.

Trains the preset *Plain MLP* twice on the standard grid 1080 x 2160 (4 000 steps,
seed 0) – once with ``lonlat`` and once with ``xyz`` input – measures the jump at
±180° and draws the Pacific around the date line in German and English.

    uv run --with matplotlib python docs/tutorial/tools/fig_naht_pazifik.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from python_streamlit_neural_earth_field import viz
from python_streamlit_neural_earth_field.data import load_etopo
from python_streamlit_neural_earth_field.model import PRESETS, build_field
from python_streamlit_neural_earth_field.training import TrainConfig, pick_device, train

IMG = Path(__file__).resolve().parent.parent / "img"
LON0, LON1, LAT0, LAT1 = 160.0, -160.0, -50.0, 10.0  # across the date line
INK, MUTED = "#1f2933", "#5f6b7a"

TEXT = {
    "de": dict(
        title="Pazifik um die Datumsgrenze (gestrichelt): die Naht entsteht nur mit Länge und Breite",
        sub="Vorlage Plain MLP, 4 versteckte Schichten, Gitter 1080 × 2160, 4 000 Schritte – nur die Eingabe ist verschieden",
        panels=["Original (ETOPO 2022)", "Gelernt mit Länge und Breite", "Gelernt mit Kugelkoordinaten"],
        jump="Sprung an der Naht: im Mittel {} m", east="O", west="W", south="S", north="N",
        unit="Höhe [m]", file="naht-pazifik.png"),
    "en": dict(
        title="Pacific around the date line (dashed): the seam only appears with longitude and latitude",
        sub="Preset Plain MLP, 4 hidden layers, grid 1080 × 2160, 4 000 steps – only the input differs",
        panels=["Original (ETOPO 2022)", "Learned with longitude and latitude", "Learned with spherical coordinates"],
        jump="Jump at the seam: {} m on average", east="E", west="W", south="S", north="N",
        unit="Elevation [m]", file="en/naht-pazifik.png"),
}


def fmt(value: float, lang: str) -> str:
    text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
    return text.replace(".", ",") if lang == "de" else text


def window(lon: np.ndarray) -> np.ndarray:
    """Column indices from 160° E across ±180° to 160° W, in map order."""
    east = np.where(lon >= LON0)[0]
    west = np.where(lon <= LON1)[0]
    return np.concatenate([east, west])


def seam_jump(field, device) -> float:
    """Mean |f(180° − ε) − f(−180° + ε)| along the date line from pole to pole."""
    lat = np.linspace(-89.9, 89.9, 1800, dtype=np.float32)
    eps = 1e-3
    a = np.stack([np.full_like(lat, 180.0 - eps), lat], 1)
    b = np.stack([np.full_like(lat, -180.0 + eps), lat], 1)
    with torch.no_grad():
        fa = field(torch.from_numpy(a).to(device)).cpu().numpy()
        fb = field(torch.from_numpy(b).to(device)).cpu().numpy()
    return float(np.abs(fa - fb).mean())


def main() -> None:
    grid = load_etopo(1080)
    device = pick_device()
    results = {}
    for mode in ("lonlat", "xyz"):
        torch.manual_seed(0)
        cfg = replace(PRESETS["Plain MLP (fails on purpose)"], input_mode=mode)
        field = build_field(cfg, float(grid.z.mean()), float(grid.z.std()))
        for _ in train(field, grid, TrainConfig(steps=4000, seed=0, eval_every=4000), device):
            pass
        field.eval()
        recon = field.predict_grid(grid.lon, grid.lat, device=device)
        results[mode] = (recon, seam_jump(field, device), field.n_parameters())
        print(mode, results[mode][2], "weights, seam jump", round(results[mode][1], 1), "m")

    cols = window(grid.lon)
    rows = np.where((grid.lat >= LAT0) & (grid.lat <= LAT1))[0]
    panels = [grid.z[np.ix_(rows, cols)], results["lonlat"][0][np.ix_(rows, cols)],
              results["xyz"][0][np.ix_(rows, cols)]]
    zmin, zmax = float(panels[0].min()), float(panels[0].max())
    scale = viz.hypsometric_colorscale(zmin, zmax)
    cmap = LinearSegmentedColormap.from_list("hyps", [(p, c) for p, c in scale])
    seam_x = int(np.sum(grid.lon[cols] >= LON0))  # column where 180° lies

    for lang, t in TEXT.items():
        fig = plt.figure(figsize=(15.6, 6.3), dpi=100)
        fig.text(0.06, 0.94, t["title"], fontsize=17, fontweight="bold", color=INK)
        fig.text(0.06, 0.87, t["sub"], fontsize=12.5, color=MUTED)
        axes = [fig.add_axes([0.06 + i * 0.28, 0.2, 0.265, 0.58]) for i in range(3)]
        for i, (ax, data) in enumerate(zip(axes, panels)):
            im = ax.imshow(data[::-1], cmap=cmap, vmin=zmin, vmax=zmax, aspect="auto",
                           extent=(0, len(cols), LAT0, LAT1), interpolation="nearest")
            ax.axvline(seam_x, color="#c0622a", linestyle="--", linewidth=1.5)
            ax.set_title(t["panels"][i], loc="left", fontsize=14, fontweight="bold", color=INK)
            ticks = [np.argmin(np.abs(np.where(grid.lon[cols] < 0, grid.lon[cols] + 360, grid.lon[cols]) - v))
                     for v in (165, 180, 195)]
            ax.set_xticks(ticks, [f"165° {t['east']}", "180°", f"165° {t['west']}"], color=MUTED)
            ax.set_yticks([10, -10, -30, -50],
                          [f"10° {t['north']}", f"10° {t['south']}", f"30° {t['south']}", f"50° {t['south']}"]
                          if i == 0 else [], color=MUTED)
            for side in ax.spines.values():
                side.set_color("#b8c1cc")
            if i:
                mode = "lonlat" if i == 1 else "xyz"
                ax.text(0, -0.13, t["jump"].format(fmt(results[mode][1], lang)), transform=ax.transAxes,
                        fontsize=13, fontweight="bold", color="#c0622a" if i == 1 else "#2a86b0")
        cax = fig.add_axes([0.905, 0.2, 0.012, 0.58])
        bar = fig.colorbar(im, cax=cax)
        bar.set_label(t["unit"], color=MUTED)
        bar.ax.tick_params(colors=MUTED)
        fig.savefig(IMG / t["file"], facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
