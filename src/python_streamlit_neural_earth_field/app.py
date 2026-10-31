"""Streamlit interface: a neural field as a compressed Earth."""

from __future__ import annotations

import io
import os
import signal
import threading
from dataclasses import replace

import httpx
import numpy as np
import streamlit as st
import torch

from . import viz
from .coast import (
    CoastScores,
    download_natural_earth,
    load_coast_distance,
    load_coastline,
    natural_earth_available,
)
from .embedding import (
    ENCODERS,
    NEEDS_ELEVATION_NETWORK,
    SATCLIP_MB,
    SATCLIP_URL,
    TEMPERATURE_MB,
    TEMPERATURE_URL,
    HeadConfig,
    download,
    encode,
    fit_and_predict,
    load_temperature,
    sample_places,
    satclip_path,
    temperature_path,
    weighted_rmse,
)
from .data import (
    ETOPO_BYTES,
    ETOPO_URL,
    ElevationGrid,
    cloud_mode,
    download_etopo,
    load_etopo,
    source_available,
    source_path,
    thin,
)
from .metrics import Compression, evaluate, human_bytes
from .ortho import (
    ATTRIBUTION,
    ORTHO_PRESETS,
    PLACES,
    TILE,
    TILE_MB,
    OrthoTrainConfig,
    build_ortho_field,
    download_tile,
    image_psnr,
    jpeg_at_size,
    load_orthophoto,
    tile_path,
    train_ortho,
)
from .ortho import RESOLUTIONS as ORTHO_RESOLUTIONS
from . import buildings as lod2
from .model import PRESETS, FieldConfig, NeuralField, build_field
from .training import TrainConfig, pick_device, train

RESOLUTIONS = {
    "180 × 360 · 1.0° · ~111 km": 180,
    "360 × 720 · 0.5° · ~56 km": 360,
    "540 × 1080 · 0.33° · ~37 km": 540,
    "720 × 1440 · 0.25° · ~28 km": 720,
    "1080 × 2160 · 0.17° · ~19 km": 1080,
    "2160 × 4320 · 0.083° · ~9 km": 2160,
}
DEFAULT_RESOLUTION = "1080 × 2160 · 0.17° · ~19 km"
# Online (shared CPU, throttled when busy for long): coarser grids and lighter training
CLOUD_RESOLUTIONS = list(RESOLUTIONS)[:3]
CLOUD_DEFAULT_RESOLUTION = "360 × 720 · 0.5° · ~56 km"
ELEVATION = "Elevation (ETOPO 2022)"
COAST = "Distance to the coast (Natural Earth)"


def fmt_ratio(r: float) -> str:
    return f"{r:.2f}×" if r < 10 else f"{r:,.0f}×"


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


@st.cache_data(show_spinner="Loading elevation grid …")
def get_grid(target_ny: int) -> ElevationGrid:
    return load_etopo(target_ny)


@st.cache_data(show_spinner="Computing the distance to the coast …")
def get_coast_grid(target_ny: int) -> ElevationGrid:
    return load_coast_distance(target_ny)


@st.cache_resource(show_spinner="Loading the coastline …")
def get_coastline():
    return load_coastline()


def require_natural_earth() -> None:
    """The coastline is small (about 3 MB): online it is fetched directly, locally the app asks."""
    if natural_earth_available():
        return
    if not cloud_mode():
        st.info(
            "**The coastline is missing.** This dataset is computed from the land polygons and "
            "coastlines of Natural Earth (1:50 million, public domain, about **3 MB**). "
            "May the app download them now?"
        )
        if not st.button("Download Natural Earth (3 MB)", type="primary"):
            st.stop()
    try:
        with st.spinner("Downloading Natural Earth …"):
            download_natural_earth()
    except (httpx.HTTPError, OSError) as error:
        st.error(f"The download failed: {error}")
        st.stop()


def make_field(cfg: FieldConfig, grid: ElevationGrid) -> NeuralField:
    coarse = grid.coarsen(cfg.context_ny).z if cfg.use_context else None
    return build_field(cfg, float(grid.z.mean()), float(grid.z.std()), coarse)


# --------------------------------------------------------------------------- UI


def sidebar() -> tuple[ElevationGrid, FieldConfig, TrainConfig, str, bool]:
    st.sidebar.title("🌍 Neural Elevation Field")

    if cloud_mode():
        st.sidebar.warning(
            "**Online version with much less data.** It runs on a shared CPU without GPU, so "
            "it uses grids of at most 540 × 1080 cells (default 360 × 720 – a ninth of the "
            "2.3 million cells of the full 1080 × 2160 grid) and lighter training. File sizes, "
            "compression ratios and errors therefore differ from the tutorial. For the full "
            "data, run the app on your own computer – see the "
            "[README](https://github.com/gkvoelkl/python-streamlit-neural-earth-field)."
        )
    st.sidebar.subheader("1 · Data")
    dataset = st.sidebar.radio(
        "Dataset", [ELEVATION, COAST],
        help="The distance to the coast turns vector data – the coastline – into a field: "
             "positive on land, negative at sea, zero exactly on the coast.",
    )
    if dataset == ELEVATION:
        st.sidebar.caption("Source: ETOPO 2022 global relief model (NOAA NCEI), 1 arc-minute")
    else:
        st.sidebar.caption(
            "Source: Natural Earth 1:50 million, land polygons and coastlines. "
            "Distance in km to the nearest coast, land +, sea −."
        )

    res_label = st.sidebar.select_slider(
        "Grid resolution",
        options=CLOUD_RESOLUTIONS if cloud_mode() else list(RESOLUTIONS),
        value=CLOUD_DEFAULT_RESOLUTION if cloud_mode() else DEFAULT_RESOLUTION,
    )
    if dataset == ELEVATION:
        grid = get_grid(RESOLUTIONS[res_label])
    else:
        require_natural_earth()
        grid = get_coast_grid(RESOLUTIONS[res_label])
        limit = st.sidebar.select_slider(
            "Limit the distance to", ["no limit", 1000, 500, 200, 100, 50, 25, 10, 1],
            value="no limit",
            format_func=lambda v: v if isinstance(v, str) else (
                "± 1 km (≈ land/sea mask)" if v == 1 else f"± {v} km"
            ),
            help="Far from the coast, the exact distance hardly matters for the coastline. "
                 "Cut off there, the network can spend its weights near the coast – but it "
                 "no longer knows the distance beyond the limit.",
        )
        if not isinstance(limit, str):
            grid = replace(
                grid, z=np.clip(grid.z, -limit, limit), source=f"{grid.source} · limited to ± {limit} km"
            )

    st.sidebar.subheader("2 · Network")
    preset_name = st.sidebar.selectbox("Preset", list(PRESETS))
    cfg = replace(PRESETS[preset_name])

    with st.sidebar.expander("Architecture details"):
        cfg.input_mode = st.selectbox(
            "Coordinate input", ["xyz", "lonlat"],
            index=["xyz", "lonlat"].index(cfg.input_mode),
            help="xyz = point on the unit sphere: no seam at ±180°, no pole distortion.",
        )
        cfg.encoding = st.selectbox(
            "Positional encoding", ["fourier", "spherical", "none"],
            index=["fourier", "spherical", "none"].index(cfg.encoding),
            help="fourier = random plane waves, spherical = spherical harmonics "
                 "(the Fourier basis of the sphere).",
        )
        if cfg.encoding == "spherical":
            cfg.sh_degree = st.slider(
                "Maximum degree L", 2, 40, cfg.sh_degree,
                help="(L+1)² features. The finest wavelength is roughly 40,000 km / L.",
            )
            st.caption(
                f"{(cfg.sh_degree + 1) ** 2} features · finest wavelength "
                f"≈ {40_000 / cfg.sh_degree:,.0f} km"
            )
            if cfg.input_mode != "xyz":
                cfg.input_mode = "xyz"
                st.caption("Spherical harmonics live on the sphere – using `xyz` input.")
        if cfg.encoding == "fourier":
            cfg.n_features = st.slider("Fourier frequencies", 16, 512, cfg.n_features, 16)
            cfg.sigma = st.slider(
                "σ (bandwidth)", 0.5, 20.0, cfg.sigma, 0.5,
                help="Small = blurry, large = grainy noise.",
            )
        cfg.layers = st.slider(
            "Hidden layers", 0, 8, cfg.layers,
            help="0 = no MLP at all: a linear model, i.e. a weighted sum of the encoded features.",
        )
        if cfg.layers > 0:
            cfg.activation = st.selectbox(
                "Activation", ["gelu", "relu", "siren"],
                index=["gelu", "relu", "siren"].index(cfg.activation),
            )
            cfg.hidden = st.slider("Neurons per layer", 8, 512, cfg.hidden, 8)
        else:
            st.caption("Linear model: elevation = weighted sum of the encoded features.")
        cfg.use_context = st.checkbox(
            "Provide coarse context field", value=cfg.use_context,
            help="The network then only learns the residual. The bytes of the coarse "
                 "grid are counted in the compression ratio.",
        )
        if cfg.use_context:
            cfg.context_ny = int(
                st.select_slider("Context field resolution", [45, 90, 180], value=cfg.context_ny)
            )

    st.sidebar.subheader("3 · Training")
    tcfg = TrainConfig()
    if cloud_mode():
        tcfg.steps = st.sidebar.slider("Steps", 200, 2000, 1000, 200)
    else:
        tcfg.steps = st.sidebar.slider("Steps", 200, 30000, 4000, 200)
    tcfg.batch_size = int(
        st.sidebar.select_slider(
            "Points per step",
            [4096, 8192, 16384] if cloud_mode() else [4096, 8192, 16384, 32768, 65536, 131072],
            value=8192 if cloud_mode() else 32768,
        )
    )
    tcfg.lr = st.sidebar.select_slider(
        "Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=3e-3, format_func=lambda v: f"{v:g}"
    )
    tcfg.area_weighted_sampling = st.sidebar.checkbox(
        "Area-weighted sampling (∝ cos lat)", value=True,
        help="Without it, the polar regions are heavily over-represented in training.",
    )
    tcfg.eval_every = max(20, tcfg.steps // 40)

    device_label = st.sidebar.selectbox("Device", ["auto", "cpu", "mps"])
    start = st.sidebar.button("🚀 Train", type="primary", use_container_width=True)
    st.sidebar.caption("Progress appears in the **📉 Training** tab.")
    return grid, cfg, tcfg, device_label, start


def budget_panel(grid: ElevationGrid, field: NeuralField) -> Compression:
    comp = Compression(grid.raw_bytes, field.stored_bytes(4), grid.n_points, field.n_parameters())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid points", fmt_int(grid.n_points))
    c1.caption(f"int16 → **{human_bytes(comp.raw_bytes)}**")
    c2.metric("Parameters", fmt_int(comp.n_parameters))
    extra = comp.model_bytes - comp.n_parameters * 4
    c2.caption(f"float32 + {human_bytes(extra)} overhead → **{human_bytes(comp.model_bytes)}**")
    c3.metric("Compression ratio", fmt_ratio(comp.ratio))
    c3.caption(f"with float16 weights: {fmt_ratio(grid.raw_bytes / field.stored_bytes(2))}")
    c4.metric("Bits per grid point", f"{comp.bits_per_point:.3f}")
    c4.caption(f"{fmt_int(comp.points_per_parameter)} points per parameter")
    if comp.ratio < 1:
        st.error(
            "The network is **larger** than the raw data. Pick a finer grid or a smaller "
            "network on the left – otherwise there is no compression."
        )
    return comp


def tab_idea(grid: ElevationGrid, field: NeuralField) -> None:
    if grid.unit != "m":
        st.info(
            "**Dataset: distance to the coast.** The coastline is vector data – a list of "
            "points. To store it in a neural field, it first becomes a field: for every cell "
            "the distance to the nearest coast in km, **positive on land, negative at sea**. "
            "The coast is the line where this field is 0. The *Coastline* tab recovers it "
            "from the trained network and compares it with the original."
        )
    st.markdown(
        """
### The idea in one sentence

Instead of storing **elevations at grid points**, you store a small network
`f(lon, lat) → elevation` and treat **its weights as the file**.

This is ordinary machine learning turned upside down:

| ordinary ML | here |
|---|---|
| generalise to new data | memorise **exactly one** dataset |
| overfitting is the failure | overfitting is the goal |
| small model, big data | the model **is** the data |
| success = test error | success = reconstruction error at a small file size |
        """
    )
    st.divider()
    st.subheader("Your budget – before training")
    comp = budget_panel(grid, field)
    st.caption(
        f"Source: {grid.source} · cell size ≈ {grid.km_per_cell:.0f} km · "
        f"values from {grid.z.min():,.0f} {grid.unit} to {grid.z.max():,.0f} {grid.unit}"
    )
    st.info(
        f"This network has to explain **{fmt_int(comp.points_per_parameter)} grid points per "
        "parameter**. That is exactly where the smoothing comes from: the field lacks the "
        "capacity for every detail. The compression ratio is the dial that controls it."
    )

    with st.expander("Two tricks it doesn't work without", expanded=True):
        st.markdown(
            """
**1 · Put the coordinates on the sphere.** `(lon, lat)` has a seam at ±180°
and two degenerate poles; the network sees jumps there that don't exist in the
data. The point `(x, y, z)` on the unit sphere doesn't have that problem.
Switch to `lonlat` on the left and look at the error map along the date line.

**2 · Fourier features.** An MLP with ReLU/GELU has a strong *spectral bias*:
it learns low frequencies quickly and high ones practically never. So instead of
feeding in `x`, you feed in `[sin(2π B x), cos(2π B x)]` with a fixed random
matrix `B ~ N(0, σ²)`. σ determines how fine the structures it can represent are.
The *Plain MLP* preset omits both and visibly fails.

`B` does **not** need to be stored – the seed is enough (4 bytes) to reproduce
`B`. Otherwise the encoding would be larger than the whole network.
            """
        )
    with st.expander("Spherical harmonics instead of random Fourier features"):
        st.markdown(
            """
Random plane waves are a makeshift on the sphere. The Fourier basis that actually
belongs to it are the **spherical harmonics** `Y_lm` – the same functions geodesy
uses for gravity and topography models. The encoding `spherical` feeds all of them
up to degree `L` into the network: complete, orthogonal, no seed needed.

The catch is the price: degree `L` needs **(L+1)²** features, and every one of
them costs a full row of weights in the first layer. `L = 16` gives 289 features
but only resolves wavelengths of about 2,500 km directly – everything finer has
to be built by the network from products of these functions. Fourier features
reach high frequencies with far fewer inputs because they don't have to cover
every direction systematically. Compare both on the error map.
            """
        )
    with st.expander("Context field (context-aware INR)"):
        st.markdown(
            """
Instead of producing the elevation out of nothing, you give the network a **very
coarse elevation grid** as an additional input. The network then only predicts
the deviation from it – a much easier task, because the large-scale structure
(continents, ocean basins) comes for free.

Important for honest numbers: this coarse grid is part of the file.
The app counts it as int16 – a 90 × 180 context costs 31.6 KB.
            """
        )
        if field.context is not None:
            st.success(
                f"Active: {field.context.ny} × {field.context.nx} context points = "
                f"{human_bytes(field.context.stored_bytes)} of "
                f"{human_bytes(field.stored_bytes(4))} file size."
            )
    with st.expander("How realistic are the 300× to 3000× from the paper?"):
        st.markdown(
            """
Huang & Hoefler (ICLR 2023) achieve these ratios on **climate fields**:
temperature, geopotential, humidity. These are spatially smooth and come as long
time series – so a network `f(lon, lat, t)` additionally exploits the enormous
redundancy *between* time steps.

**Topography is the harder case.** Coastlines and mountain ridges are nearly
fractal, there is no time axis to exploit, and every detail has to come from the
weights themselves. Expect ratios in the tens to a few hundred here – at an error
you can read off yourself in the *Rate & Error* tab. That is exactly what makes
the example useful for learning: the trade-off is visible rather than hidden.
            """
        )


def tab_training(grid, cfg, tcfg, device_label, start) -> None:
    if start:
        field = make_field(cfg, grid)
        device = pick_device(device_label)
        st.caption(f"Device: **{device}** · {fmt_int(field.n_parameters())} parameters")

        bar = st.progress(0.0, text="Starting …")
        metric_slot = st.empty()
        chart_slot = st.empty()
        steps: list[int] = []
        rmses: list[float] = []

        for p in train(field, grid, tcfg, device):
            steps.append(p.step)
            rmses.append(p.scores.rmse)
            bar.progress(
                p.step / p.total,
                text=f"Step {p.step}/{p.total} · RMSE {p.scores.rmse:,.1f} {grid.unit} · "
                     f"{p.elapsed_s:.0f} s",
            )
            with metric_slot.container():
                m = p.scores
                a, b, c, d = st.columns(4)
                a.metric("RMSE", f"{m.rmse:,.1f} {grid.unit}")
                b.metric("weighted RMSE", f"{m.weighted_rmse:,.1f} {grid.unit}")
                c.metric("MAE", f"{m.mae:,.1f} {grid.unit}")
                d.metric("R²", f"{m.r2:.4f}")
            chart_slot.plotly_chart(
                viz.loss_curve(steps, rmses, unit=grid.unit), use_container_width=True,
                key=f"loss{p.step}",
            )

        with st.spinner("Reconstructing the full grid …"):
            recon = field.predict_grid(grid.lon, grid.lat, device=device)
            scores = evaluate(grid.z, recon, grid.cell_weights())
        st.session_state.update(
            field=field, recon=recon, scores=scores, history=(steps, rmses),
            grid_shape=(grid.ny, grid.nx), grid_source=grid.source, trained_cfg=cfg,
        )
        st.success(
            "Done – the *Globe*, *Maps*, *Super-Resolution* and *Coastline* tabs show the result."
        )
        return

    if "field" not in st.session_state:
        st.info("Choose settings on the left and press **🚀 Train**.")
        st.markdown(
            "During training, the error is measured on a fixed sample of "
            "200,000 grid points; the full grid is only evaluated at the end – "
            "otherwise measuring would take longer than learning."
        )
        return

    st.caption("Result of the last training run (full grid)")
    scores = st.session_state.scores
    items = list(scores.to_dict(grid.unit).items())
    for row in (items[:4], items[4:]):
        cols = st.columns(4)
        for col, (name, value) in zip(cols, row):
            col.metric(name, f"{value:,.4f}" if abs(value) < 10 else f"{value:,.1f}")
    steps, rmses = st.session_state.history
    st.plotly_chart(viz.loss_curve(steps, rmses, unit=grid.unit), use_container_width=True)
    budget_panel(grid, st.session_state.field)


def _require_result(grid: ElevationGrid) -> np.ndarray | None:
    if "field" not in st.session_state:
        st.info("Train first.")
        return None
    if st.session_state.get("grid_source", grid.source) != grid.source:
        st.warning("The dataset has changed since training – please retrain or switch back.")
        return None
    if st.session_state.grid_shape != (grid.ny, grid.nx):
        st.warning(
            "The grid resolution has changed since training – please retrain "
            f"or switch back to {st.session_state.grid_shape[0]} rows."
        )
        return None
    return st.session_state.recon


def tab_globe(grid: ElevationGrid) -> None:
    recon = _require_result(grid)
    if recon is None:
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    if grid.unit == "m":
        exaggeration = c1.slider(
            "Vertical exaggeration", 1, 200, 40, help="1 = true to scale (and flat)."
        )
    else:
        exaggeration = 0
        c1.caption("A distance is not a height – the globes are shown without relief.")
    max_rows = c2.select_slider("Display grid", [90, 180, 240, 360], value=180)
    camera_lon = c3.slider("View direction (longitude)", -180, 180, 10)

    display = grid.coarsen(max_rows)
    recon_display = ElevationGrid(grid.lon, grid.lat, recon, "recon").coarsen(max_rows)
    zmin, zmax = float(grid.z.min()), float(grid.z.max())

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            viz.globe(display.lon, display.lat, display.z, title="Original (grid data)",
                      exaggeration=exaggeration, zmin=zmin, zmax=zmax, camera_lon=camera_lon,
                      unit=grid.unit),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.globe(recon_display.lon, recon_display.lat, recon_display.z,
                      title="Neural field", exaggeration=exaggeration,
                      zmin=zmin, zmax=zmax, camera_lon=camera_lon, unit=grid.unit),
            use_container_width=True,
        )
    st.caption(
        f"Both spheres show {display.ny} × {display.nx} points. The right one comes from "
        f"{fmt_int(st.session_state.field.n_parameters())} numbers "
        f"({human_bytes(st.session_state.field.stored_bytes(4))}). Dragging rotates only "
        "that sphere – use the *View direction* slider for a synchronised comparison."
    )


def tab_maps(grid: ElevationGrid) -> None:
    recon = _require_result(grid)
    if recon is None:
        return
    err = recon - grid.z
    zmin, zmax = float(grid.z.min()), float(grid.z.max())

    # The browser receives every map value as JSON -- with 9 M points that gets sluggish.
    max_rows = 480
    truth_view = grid.coarsen(max_rows)
    recon_view = ElevationGrid(grid.lon, grid.lat, recon, "recon").coarsen(max_rows)
    err_lon, err_lat, err_view = thin(grid.lon, grid.lat, err, max_rows)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            viz.elevation_map(truth_view.lon, truth_view.lat, truth_view.z,
                              title="Original", zmin=zmin, zmax=zmax, unit=grid.unit),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.elevation_map(recon_view.lon, recon_view.lat, recon_view.z,
                              title="Reconstruction", zmin=zmin, zmax=zmax, unit=grid.unit),
            use_container_width=True,
        )

    a, b = st.columns([3, 2])
    with a:
        st.plotly_chart(viz.error_map(err_lon, err_lat, err_view, unit=grid.unit),
                        use_container_width=True)
    with b:
        st.plotly_chart(viz.error_histogram(err, unit=grid.unit), use_container_width=True)
    if truth_view.ny < grid.ny:
        st.caption(
            f"The maps are thinned to {truth_view.ny} × {truth_view.nx} for drawing "
            f"(elevation maps averaged, error map subsampled). All metrics come from the "
            f"full {grid.ny} × {grid.nx} grid."
        )

    if grid.unit == "m":
        st.markdown(
            "**What to look for:** the error almost never sits in the open ocean, but along "
            "coasts, ocean trenches and mountain ridges – wherever the elevation changes "
            "sharply over a short distance. Those high frequencies are exactly what compression costs."
        )
    else:
        st.markdown(
            "**What to look for:** the distance has kinks – along the coast, where the sign "
            "changes, and on the lines exactly halfway between two coasts. Small islands far "
            "out at sea create whole circles in the field. The error sits on these kinks."
        )
    with st.expander("Detailed numbers"):
        scores = st.session_state.scores.to_dict(grid.unit)
        st.dataframe(
            {"Metric": list(scores), "Value": [f"{v:,.3f}" for v in scores.values()]},
            use_container_width=True, hide_index=True,
        )


def tab_superres(grid: ElevationGrid) -> None:
    if _require_result(grid) is None:
        return
    st.markdown(
        "The field is a **continuous** function: you can query it at any "
        "coordinate – even finer than it was trained on."
    )
    c1, c2, c3, c4 = st.columns(4)
    lon_c = c1.number_input("Longitude (centre)", -180.0, 180.0, 10.0, 1.0)
    lat_c = c2.number_input("Latitude (centre)", -89.0, 89.0, 46.0, 1.0)
    span = float(c3.select_slider("Window [°]", [2, 5, 10, 20, 45, 90], value=20))
    n = int(c4.select_slider("Points per axis", [128, 256, 384, 512], value=256))

    lon_q = np.linspace(lon_c - span / 2, lon_c + span / 2, n, dtype=np.float32)
    lat_q = np.linspace(lat_c - span / 2, lat_c + span / 2, n, dtype=np.float32)

    field: NeuralField = st.session_state.field
    with st.spinner("Querying the field …"):
        fine = field.predict_grid(lon_q, lat_q)

    i0 = int(np.searchsorted(grid.lat, lat_q[0]))
    i1 = max(i0 + 2, int(np.searchsorted(grid.lat, lat_q[-1])))
    j0 = int(np.searchsorted(grid.lon, lon_q[0]))
    j1 = max(j0 + 2, int(np.searchsorted(grid.lon, lon_q[-1])))
    patch = grid.z[i0:i1, j0:j1]
    zmin, zmax = float(patch.min()), float(patch.max())

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            viz.elevation_map(grid.lon[j0:j1], grid.lat[i0:i1], patch,
                              title=f"Training grid ({patch.shape[0]} × {patch.shape[1]})",
                              zmin=zmin, zmax=zmax, unit=grid.unit),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.elevation_map(lon_q, lat_q, fine, title=f"Field queried ({n} × {n})",
                              zmin=zmin, zmax=zmax, unit=grid.unit),
            use_container_width=True,
        )
    st.warning(
        "**To be honest:** this is not real super-resolution. The field interpolates "
        "smoothly between what it has learned – it doesn't invent details that aren't in "
        "the training data. It is useful because you can evaluate it at any coordinate "
        "without re-gridding (e.g. as a data loader that delivers grids of arbitrary "
        "resolution)."
    )


def tab_coastline(grid: ElevationGrid) -> None:
    recon = _require_result(grid)
    if recon is None:
        return
    if not natural_earth_available():
        st.info(
            "The comparison needs the coastline of Natural Earth (1:50 million, about 3 MB)."
        )
        if st.button("Download Natural Earth (3 MB)", key="ne-coast-tab"):
            with st.spinner("Downloading Natural Earth …"):
                download_natural_earth()
            st.rerun()
        return
    truth = get_coast_grid(grid.ny)
    coastline = get_coastline()
    scores = CoastScores.compare(truth, recon > 0)
    field: NeuralField = st.session_state.field

    st.markdown(
        "Where the field is exactly **0**, the network puts the coast – the 0 m line for "
        "elevations, the line of zero distance for the distance to the coast. Here it is "
        "compared with the true coastline of Natural Earth."
    )
    a, b, c, d = st.columns(4)
    a.metric("Correct side", f"{scores.correct_side:.1%}",
             help="Share of the Earth's area where the network says land on land and sea at sea.")
    b.metric("… in coastal cells", f"{scores.correct_side_at_coast:.1%}",
             help="The same, only for the cells through which the true coastline runs.")
    c.metric("Coast offset (median)", f"{scores.offset_median_km:,.1f} km",
             help="How far the cells of the recovered coast are from the true coastline.")
    d.metric("Coast offset (90 %)", f"{scores.offset_p90_km:,.0f} km",
             help="90 % of the recovered coast lies at most this far from the true coastline.")
    st.caption(
        f"For comparison: even the grid itself puts its coast cells up to half a cell "
        f"(≈ {grid.km_per_cell / 2:.0f} km) away from the true line."
    )

    e, f, g = st.columns(3)
    e.metric("Coastline as vector data", human_bytes(coastline.raw_bytes),
             help="Every point of the coastline as a pair of float32 numbers.")
    e.caption(f"{fmt_int(coastline.n_points)} points × 2 numbers × 4 bytes")
    f.metric("Network", human_bytes(field.stored_bytes(4)))
    f.caption(f"{fmt_int(field.n_parameters())} parameters · float32")
    g.metric("Grid of the field (int16)", human_bytes(grid.raw_bytes))
    g.caption(f"{grid.ny} × {grid.nx} cells × 2 bytes")

    c1, c2, c3 = st.columns(3)
    lon_c = c1.number_input("Longitude (centre)", -180.0, 180.0, 10.0, 1.0, key="coast-lon")
    lat_c = c2.number_input("Latitude (centre)", -89.0, 89.0, 46.0, 1.0, key="coast-lat")
    span = float(c3.select_slider("Window [°]", [5, 10, 20, 45, 90], value=20, key="coast-span"))
    i0 = int(np.searchsorted(grid.lat, lat_c - span / 2))
    i1 = max(i0 + 2, int(np.searchsorted(grid.lat, lat_c + span / 2)))
    j0 = int(np.searchsorted(grid.lon, lon_c - span))
    j1 = max(j0 + 2, int(np.searchsorted(grid.lon, lon_c + span)))
    lon, lat = grid.lon[j0:j1], grid.lat[i0:i1]
    lines = [
        line for line in coastline.coast
        if line[:, 0].max() >= lon[0] and line[:, 0].min() <= lon[-1]
        and line[:, 1].max() >= lat[0] and line[:, 1].min() <= lat[-1]
    ]
    st.plotly_chart(
        viz.coastline_map(lon, lat, grid.z[i0:i1, j0:j1], recon[i0:i1, j0:j1], lines,
                          unit=grid.unit),
        use_container_width=True,
    )
    st.caption(
        "Background: the training grid. Black: the coastline of Natural Earth. "
        "Orange: where the network's field is 0."
    )


@st.cache_data(show_spinner="Loading the temperatures …")
def get_temperature():
    return load_temperature()


def tab_encoder() -> None:
    st.markdown(
        "A **location encoder** turns a place into a list of numbers. It is not meant to store "
        "one field but to be **transferred**: a small second network learns a new task from these "
        "numbers. The task here is the **mean annual air temperature** on land, learned from only a "
        "few places and checked on all the others."
    )
    if cloud_mode():
        st.info("This tab needs two downloads and only runs in the local version of the app.")
        return
    missing = [(p, url, mb, label) for p, url, mb, label in (
        (temperature_path(), TEMPERATURE_URL, TEMPERATURE_MB,
         "temperatures of the University of Delaware (no restrictions)"),
        (satclip_path(), SATCLIP_URL, SATCLIP_MB, "pretrained SatCLIP encoder (MIT licence)"),
    ) if not p.exists()]
    if missing:
        total = sum(m[2] for m in missing)
        st.info("**Missing data:** " + " and ".join(f"the {m[3]}, about {m[2]} MB" for m in missing) + ".")
        if not st.button(f"Download ({total} MB)", type="primary", key="encoder-download"):
            return
        try:
            for path, url, _, label in missing:
                with st.spinner(f"Downloading the {label} …"):
                    download(path, url)
        except (httpx.HTTPError, OSError) as error:
            st.error(f"The download failed: {error}")
            return
        st.rerun()

    data = get_temperature()
    field = st.session_state.get("field")
    has_elevation = field is not None and st.session_state.get("grid_source", "").startswith("ETOPO")
    available = [e for e in ENCODERS if has_elevation or e not in NEEDS_ELEVATION_NETWORK]

    c1, c2, c3 = st.columns([1, 5, 1])
    n = c1.select_slider("Training places", [100, 300, 1000, 3000, 10000], value=1000, key="encoder-n")
    # a new key once the elevation network is there, so that its two inputs are selected as well
    chosen = c2.multiselect("Inputs", available, default=available, key=f"encoder-inputs-{len(available)}")
    seed = int(c3.number_input("Seed", 0, 99, 0, key="encoder-seed"))
    if not has_elevation:
        st.caption(
            "Train an elevation network (dataset *Elevation (ETOPO 2022)*) to add its hidden layer "
            "and its elevation as inputs."
        )
    if st.button("Compare", type="primary", key="encoder-run", disabled=not chosen):
        train_idx = sample_places(data, n, seed)
        results = {}
        bar = st.progress(0.0)
        for k, name in enumerate(chosen):
            features = encode(name, data.lonlat, field)
            results[name] = fit_and_predict(features, data.celsius, train_idx, HeadConfig(seed=seed))
            bar.progress((k + 1) / len(chosen), text=name)
        bar.empty()
        st.session_state.encoder_results = (train_idx, results)

    if "encoder_results" not in st.session_state:
        return
    train_idx, results = st.session_state.encoder_results
    if any(len(v) != data.n for v in results.values()):  # from an older version of the data
        del st.session_state.encoder_results
        return
    test = np.ones(data.n, dtype=bool)
    test[train_idx] = False
    w = data.weights
    rmse = {k: weighted_rmse(data.celsius[test], v[test], w[test]) for k, v in results.items()}
    baseline = weighted_rmse(data.celsius[test], np.full(test.sum(), np.average(data.celsius[train_idx])), w[test])
    st.plotly_chart(viz.encoder_bars(list(rmse), list(rmse.values()), baseline), use_container_width=True)
    st.caption(
        f"{fmt_int(len(train_idx))} training places, {fmt_int(test.sum())} places to check. "
        "Each input feeds the same small network (2 hidden layers of 64 neurons, 2 000 steps)."
    )

    shown = st.selectbox("Map for", list(results), key="encoder-map")
    lon, lat, truth = data.grid()
    _, _, pred = data.grid(results[shown])
    a, b = st.columns(2)
    a.plotly_chart(viz.temperature_map(lon, lat, truth, title="Measured (University of Delaware, 1981–2010)",
                                       points=data.lonlat[train_idx]), use_container_width=True)
    b.plotly_chart(viz.temperature_map(lon, lat, pred, title=f"Predicted from {shown}"),
                   use_container_width=True)
    _, _, err = data.grid(results[shown] - data.celsius)
    st.plotly_chart(viz.error_map(lon, lat, err, title="Error (prediction − measured)", unit="°C"),
                    use_container_width=True)


@st.cache_data(show_spinner="Loading the orthophoto …")
def get_orthophoto(n: int):
    return load_orthophoto(n)


def _error_image(truth: np.ndarray, pred: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Mean absolute error over the three colours, brightened: black = exact."""
    err = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(axis=-1) * gain
    return np.clip(err, 0, 255).astype(np.uint8)


def tab_ortho() -> None:
    st.markdown(
        "An **aerial photo** as a neural field: f(E, N) → (R, G, B). The tile covers 1 km × 1 km "
        "of the old town of **Regensburg**. A kilometre is flat for all practical purposes, so the "
        "network gets map coordinates in metres (ETRS89 / UTM 32N) instead of a point on the globe."
    )
    if cloud_mode():
        st.info("This tab needs a download and a lot of computing time; it only runs in the local version of the app.")
        return
    if not tile_path().exists():
        st.info(f"**Missing data:** the orthophoto tile {TILE} (DOP40), about {TILE_MB} MB.")
        if not st.button(f"Download ({TILE_MB} MB)", type="primary", key="ortho-download"):
            return
        try:
            with st.spinner("Downloading the orthophoto …"):
                download_tile()
        except (httpx.HTTPError, OSError) as error:
            st.error(f"The download failed: {error}")
            return
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    n = c1.selectbox("Pixels", list(ORTHO_RESOLUTIONS), index=1, key="ortho-n",
                     format_func=lambda k: f"{k} × {k} · {ORTHO_RESOLUTIONS[k]} m")
    preset = c2.selectbox("Network", list(ORTHO_PRESETS), index=1, key="ortho-preset")
    steps = int(c3.select_slider("Steps", [300, 1000, 3000, 6000], value=1000, key="ortho-steps"))
    device_label = c4.selectbox("Device", ["auto", "cpu", "mps"], key="ortho-device")
    photo = get_orthophoto(n)
    cfg = ORTHO_PRESETS[preset]
    field = build_ortho_field(cfg, photo)
    st.caption(
        f"{fmt_int(photo.n_pixels)} pixels × 3 colours = {human_bytes(photo.raw_bytes)} uncompressed · "
        f"network: {fmt_int(field.n_parameters())} weights = {human_bytes(field.stored_bytes())} in float16 · "
        f"Data: {ATTRIBUTION}"
    )

    if st.button("Train", type="primary", key="ortho-train"):
        device = pick_device(device_label)
        bar = st.progress(0.0, text="Starting …")
        for p in train_ortho(field, photo, OrthoTrainConfig(steps=steps), device):
            bar.progress(p.step / p.total,
                         text=f"Step {p.step}/{p.total} · PSNR {p.psnr_db:.1f} dB · {p.elapsed_s:.0f} s")
        bar.empty()
        with st.spinner("Drawing the image from the network …"):
            image = field.render(photo)
        st.session_state.ortho = dict(field=field, image=image, n=n, preset=preset, steps=steps,
                                      seconds=p.elapsed_s)

    result = st.session_state.get("ortho")
    if result is None:
        return
    if result["n"] != n:
        st.warning(f"The resolution has changed since training – please retrain or switch back to {result['n']} pixels.")
        return
    field, image = result["field"], result["image"]
    size = field.stored_bytes()
    jpeg, jpeg_bytes, quality = jpeg_at_size(photo.rgb, size)

    a, b, c, d = st.columns(4)
    a.metric("PSNR of the network", f"{image_psnr(photo.rgb, image):.1f} dB")
    b.metric("File size (float16)", human_bytes(size))
    c.metric("Compression ratio", f"{photo.raw_bytes / size:,.0f} : 1")
    d.metric(f"JPEG of the same size (quality {quality})", f"{image_psnr(photo.rgb, jpeg):.1f} dB")
    st.caption(
        f"{result['preset']} network, {fmt_int(result['steps'])} steps, {result['seconds']:.0f} s. "
        "PSNR: the higher, the closer to the original; +6 dB halves the error."
    )

    a, b, c = st.columns(3)
    a.image(photo.rgb, caption="Original (DOP40)", use_container_width=True)
    b.image(image, caption="Network", use_container_width=True)
    c.image(jpeg, caption=f"JPEG, {human_bytes(jpeg_bytes)}", use_container_width=True)

    st.markdown("**Zoom in:** 100 m × 100 m around a landmark")
    place = st.selectbox("Detail", list(PLACES), key="ortho-place")
    rows, cols = photo.crop(*PLACES[place], 100)
    a, b, c, d = st.columns(4)
    a.image(photo.rgb[rows, cols], caption="Original", use_container_width=True)
    b.image(image[rows, cols], caption="Network", use_container_width=True)
    c.image(jpeg[rows, cols], caption="JPEG", use_container_width=True)
    d.image(_error_image(photo.rgb[rows, cols], image[rows, cols]),
            caption="Error of the network (×4, black = exact)", use_container_width=True)


@st.cache_data(show_spinner="Reading the building models and computing the distance field …")
def get_building_grid(res: float):
    return lod2.load_building_grid(res)


@st.cache_data(show_spinner="Building the surface …")
def _building_surface(field: np.ndarray, res: float, step: int):
    return lod2.surface(field, res, step)


def tab_buildings() -> None:
    st.markdown(
        "**Buildings** as a neural field in three dimensions: f(E, N, H) → distance to the nearest wall or "
        "roof, negative inside (like the coast in the Coastline tab). The area is 300 m × 300 m around the "
        "**cathedral of Regensburg**, 120 m high. The surface of the buildings is where the distance is 0."
    )
    if cloud_mode():
        st.info("This tab needs a large download and a lot of computing time; it only runs in the local version of the app.")
        return
    if not lod2.tile_path().exists():
        st.info(f"**Missing data:** the LoD2 building models of tile {lod2.TILE} (2 km × 2 km), about {lod2.TILE_MB} MB.")
        if not st.button(f"Download ({lod2.TILE_MB} MB)", type="primary", key="lod2-download"):
            return
        try:
            with st.spinner("Downloading the building models …"):
                lod2.download_tile()
        except (httpx.HTTPError, OSError) as error:
            st.error(f"The download failed: {error}")
            return
        st.rerun()

    c1, c2, c3, c4, c5 = st.columns(5)
    res = c1.selectbox("Voxel size", [2.0, 1.0], index=1, key="lod2-res", format_func=lambda r: f"{r:g} m")
    preset = c2.selectbox("Network", list(lod2.BUILDING_PRESETS), index=1, key="lod2-preset")
    limit = c3.selectbox("Limit the distance to", [2.0, 5.0, 10.0, 20.0], index=1, key="lod2-limit",
                         format_func=lambda v: f"±{v:g} m")
    steps = int(c4.select_slider("Steps", [300, 1000, 3000, 6000], value=1000, key="lod2-steps"))
    device_label = c5.selectbox("Device", ["auto", "cpu", "mps"], key="lod2-device")
    grid = get_building_grid(res)
    cfg = replace(lod2.BUILDING_PRESETS[preset], limit_m=limit)
    field = lod2.build_building_field(cfg, grid)
    nz, ny, nx = grid.shape
    st.caption(
        f"{grid.n_buildings} buildings · {fmt_int(grid.n_points)} corner points · CityGML {human_bytes(grid.gml_bytes)} · "
        f"grid {nx} × {ny} × {nz} = {fmt_int(grid.n_voxels)} voxels · network: {fmt_int(field.n_parameters())} weights = "
        f"{human_bytes(field.stored_bytes())} in float16 · Data: {lod2.ATTRIBUTION}"
    )

    if st.button("Train", type="primary", key="lod2-train"):
        device = pick_device(device_label)
        bar = st.progress(0.0, text="Starting …")
        for p in lod2.train_building_field(field, grid, lod2.BuildingTrainConfig(steps=steps), device):
            bar.progress(p.step / p.total,
                         text=f"Step {p.step}/{p.total} · right side {100 * p.correct_side:.1f} % · {p.elapsed_s:.0f} s")
        bar.empty()
        with st.spinner("Evaluating the network on every voxel …"):
            predicted = field.render(grid)
            scores = lod2.BuildingScores.compare(grid, predicted, limit)
        st.session_state.lod2 = dict(field=field, predicted=predicted, scores=scores, res=res, preset=preset,
                                     steps=steps, limit=limit, seconds=p.elapsed_s)

    result = st.session_state.get("lod2")
    if result is None:
        return
    if result["res"] != res:
        st.warning(f"The voxel size has changed since training – please retrain or switch back to {result['res']:g} m.")
        return
    field, predicted, scores = result["field"], result["predicted"], result["scores"]
    size = field.stored_bytes()
    a, b, c, d = st.columns(4)
    a.metric("Volume overlap (IoU)", f"{scores.iou:.2f}")
    b.metric("Surface offset (median)", f"{scores.offset_median_m:.1f} m")
    c.metric("Surface offset (90 %)", f"{scores.offset_p90_m:.1f} m")
    d.metric("Network vs CityGML", f"{human_bytes(size)} · {grid.gml_bytes / size:,.0f} : 1")
    st.caption(
        f"{result['preset']} network, ±{result['limit']:g} m, {fmt_int(result['steps'])} steps, {result['seconds']:.0f} s. "
        f"Right side of the surface: {100 * scores.correct_side:.1f} % of all voxels. IoU: the volume both see as "
        "building, divided by the volume at least one of them sees as building – 1 is perfect. Surface offset: "
        "how far the network's surface lies from the true one."
    )

    step = 2 if res < 2 else 1
    ground = float(grid.height[0])
    zmax = 60.0  # houses are 10 to 25 m high; the cathedral's towers stay dark above 60 m
    a, b = st.columns(2)
    with a:
        verts, faces = _building_surface(grid.sdf, res, step)
        st.plotly_chart(viz.building_mesh(lod2.mesh_enh(grid, verts), faces, ground=ground, zmax=zmax,
                                          title="LoD2 building models"), use_container_width=True)
    with b:
        verts, faces = _building_surface(predicted, res, step)
        if len(verts):
            st.plotly_chart(viz.building_mesh(lod2.mesh_enh(grid, verts), faces, ground=ground, zmax=zmax,
                                              title="Network"), use_container_width=True)
        else:
            st.info("The network has no surface yet – train longer.")

    st.markdown("**Horizontal slice** through the distance field")
    level = st.slider("Height above the lowest ground", 1, int(grid.height[-1] - grid.height[0]) - 1, 10,
                      key="lod2-level", format="%d m")
    k = int(np.argmin(np.abs(grid.height - (grid.height[0] - res / 2 + level))))
    a, b = st.columns(2)
    a.plotly_chart(viz.distance_slice(grid.east, grid.north, grid.sdf[k], limit=result["limit"], title="LoD2"),
                   use_container_width=True)
    b.plotly_chart(viz.distance_slice(grid.east, grid.north, predicted[k], limit=result["limit"], title="Network"),
                   use_container_width=True)
    st.caption("Red inside, blue outside, white near a wall or a roof; the black line is the 0-level, i.e. the walls "
               "at this height. Small black rings in the network's slice are crumbs: places just above a low roof "
               "where the network's noise crosses 0.")


def tab_rate(grid: ElevationGrid) -> None:
    st.markdown(
        "Compression is always a trade-off: **file size versus error**. This "
        "rate-distortion curve trains several network sizes on the same grid and "
        "shows what you get at each point."
    )
    c1, c2, c3 = st.columns(3)
    cloud = cloud_mode()
    steps = int(c1.select_slider("Steps per network",
                                 [300, 600] if cloud else [300, 600, 1000, 2000], value=600))
    widths = c2.multiselect("Network widths",
                            [16, 32, 64, 128] if cloud else [16, 24, 32, 48, 64, 96, 128, 192, 256],
                            default=[16, 64] if cloud else [16, 32, 64, 128])
    with_context = c3.checkbox("Also with context field (90 × 180)", value=not cloud)
    run = st.button("Compute curve", type="primary")

    if run and widths:
        device = pick_device()
        variants = [("without context", False)] + ([("with context", True)] if with_context else [])
        total = len(widths) * len(variants)
        bar = st.progress(0.0, text="…")
        rows: list[dict] = []
        done = 0
        for name, ctx in variants:
            for width in sorted(widths):
                cfg = FieldConfig(
                    hidden=width, layers=4, n_features=min(128, max(32, width)), sigma=8.0,
                    use_context=ctx, context_ny=90,
                )
                field = make_field(cfg, grid)
                tcfg = TrainConfig(steps=steps, batch_size=32768, eval_every=max(50, steps))
                last = None
                for p in train(field, grid, tcfg, device):
                    last = p
                    bar.progress(
                        (done + p.step / p.total) / total,
                        text=f"{name}, width {width}: step {p.step}/{p.total}",
                    )
                done += 1
                comp = Compression(grid.raw_bytes, field.stored_bytes(4),
                                   grid.n_points, field.n_parameters())
                rows.append({
                    "Variant": name, "Width": width, "Parameters": field.n_parameters(),
                    "File size": field.stored_bytes(4), "Ratio": comp.ratio,
                    f"weighted RMSE [{grid.unit}]": last.scores.weighted_rmse,
                    "PSNR [dB]": last.scores.psnr_db,
                })
        bar.empty()
        st.session_state.rd_rows = rows
        st.session_state.rd_unit = grid.unit

    rows = st.session_state.get("rd_rows")
    if not rows:
        return
    unit = st.session_state.get("rd_unit", "m")
    key = f"weighted RMSE [{unit}]"
    st.plotly_chart(viz.rate_distortion(rows, unit=unit), use_container_width=True)
    st.dataframe(
        [{**r, "File size": human_bytes(r["File size"]), "Ratio": fmt_ratio(r["Ratio"]),
          key: round(r[key], 1), "PSNR [dB]": round(r["PSNR [dB]"], 1)}
         for r in rows],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The error here comes from the 200,000-point sample and, for short runs, has "
        "not fully converged yet – that is enough to show the shape of the curve."
    )


def tab_export(grid: ElevationGrid) -> None:
    if _require_result(grid) is None:
        return
    field: NeuralField = st.session_state.field

    def dump(dtype: torch.dtype) -> bytes:
        state = {
            k: (v.to(dtype) if v.is_floating_point() else v) for k, v in field.state_dict().items()
        }
        buffer = io.BytesIO()
        torch.save(
            {"config": field.cfg.to_dict(), "state_dict": state,
             "grid": {"ny": grid.ny, "nx": grid.nx}}, buffer
        )
        return buffer.getvalue()

    f32, f16 = dump(torch.float32), dump(torch.float16)
    a, b, c = st.columns(3)
    a.metric("Raw grid (int16)", human_bytes(grid.raw_bytes))
    b.metric("Model float32", human_bytes(len(f32)), fmt_ratio(grid.raw_bytes / len(f32)))
    c.metric("Model float16", human_bytes(len(f16)), fmt_ratio(grid.raw_bytes / len(f16)))

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Weights float32 (.pt)", f32, "earth_field_fp32.pt",
                       use_container_width=True)
    d2.download_button("⬇️ Weights float16 (.pt)", f16, "earth_field_fp16.pt",
                       use_container_width=True)
    st.caption(
        "The actual files are slightly larger than the bare parameter count: the "
        "`.pt` format adds a pickle header and tensor names. The budget display only "
        "counts the payload."
    )
    st.info(
        "**float16 halves the file almost for free** – for a Fourier MLP the elevations "
        "typically shift by well under a metre on average and by up to about ten metres "
        "in single cells; the weighted RMSE does not change. For a **SIREN** it is "
        "noticeably more expensive: the sine activation with `w0 = 30` amplifies every "
        "rounding error in the weights, so single cells can move by a hundred metres or "
        "more. If you want to quantise there, do it during training."
    )
    st.code(
        "import torch\n"
        "from python_streamlit_neural_earth_field.model import FieldConfig, NeuralField\n\n"
        "blob = torch.load('earth_field_fp32.pt', weights_only=False)\n"
        "cfg = FieldConfig(**blob['config'])\n"
        "state = {k: v.float() if v.is_floating_point() else v\n"
        "         for k, v in blob['state_dict'].items()}\n"
        "coarse = state['context.coarse'].numpy() if cfg.use_context else None\n"
        "field = NeuralField(cfg, 0.0, 1.0, coarse)\n"
        "field.load_state_dict(state)\n"
        "field.eval()\n\n"
        "# Query the elevation at Mount Everest\n"
        "print(field(torch.tensor([[86.925, 27.988]])).item(), 'm')",
        language="python",
    )
    st.json(field.cfg.to_dict())


def _manual_download_hint() -> str:
    return f"```\ncurl -L -o {source_path()} \\\n  {ETOPO_URL}\n```"


def _shut_down(delay_s: float = 2.0) -> None:
    """Ends the Streamlit server shortly after the current page has been sent."""
    threading.Timer(delay_s, lambda: os.kill(os.getpid(), signal.SIGINT)).start()


def require_elevation_data() -> None:
    """Makes sure the ETOPO file is there – asks before downloading it, quits otherwise."""
    if source_available() or cloud_mode():  # online, the averaged grids come from the release
        return

    st.title("The Earth as a Neural Field")
    if st.session_state.get("download_declined"):
        st.error(
            "**This app needs the ETOPO 2022 elevation data and cannot run without it.** "
            "The app is shutting down now. Start it again to download the data, "
            "or download it yourself:\n\n" + _manual_download_hint()
        )
        print(f"ETOPO 2022 elevation data missing ({source_path()}) – the app needs it. Shutting down.")
        _shut_down()
        st.stop()

    st.info(
        "**The elevation data is missing.** This app works with the global relief model "
        "ETOPO 2022 of NOAA (1 arc-minute, about **456 MB**). It is not yet in "
        f"`{source_path().parent}`.\n\nMay the app download it now? This happens only once "
        "and can take several minutes, depending on the connection."
    )
    if not st.session_state.get("downloading"):
        yes, no, _ = st.columns([1, 1, 3])
        if no.button("No, quit"):
            st.session_state.download_declined = True
            st.rerun()
        if yes.button("Download (456 MB)", type="primary"):
            st.session_state.downloading = True  # hides the buttons, so a second click cannot restart it
            st.rerun()
        st.stop()

    bar = st.progress(0.0, text="Connecting …")

    def report(done: int, total: int | None) -> None:
        expected = total or ETOPO_BYTES  # NOAA sends no size, the file's size is known
        mb = 1 << 20
        bar.progress(
            min(done / expected, 1.0),
            text=f"{done / mb:,.0f} of {expected / mb:,.0f} MB – please keep this page open",
        )

    try:
        download_etopo(report)
    except (httpx.HTTPError, OSError) as error:
        st.session_state.downloading = False
        bar.empty()
        st.error(
            f"The download failed: {error}\n\nPlease try again later or download the file "
            "yourself:\n\n" + _manual_download_hint()
        )
        st.stop()
    bar.empty()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Neural Elevation Field of the Earth", page_icon="🌍", layout="wide")
    require_elevation_data()
    grid, cfg, tcfg, device_label, start = sidebar()

    st.title("The Earth as a Neural Field")
    st.caption(f"f(lon, lat) → {grid.quantity.lower()} · the weights are the compressed data")

    labels = ["💡 Idea & Budget", "📉 Training", "🌐 Globe 3D", "🗺️ Maps & Error",
              "🔍 Super-Resolution", "🏖️ Coastline", "🧭 Location Encoder", "🛩️ Orthophoto", "🏛️ Buildings 3D",
              "📐 Rate & Error",
              "💾 Export"]
    # Each press of Train gives the tab bar a new key, so it is created afresh and
    # opens on the Training tab. Otherwise the tabs keep the user's choice and don't
    # rerun the app – a rerun would abort a training that is still running.
    if start:
        st.session_state.train_runs = st.session_state.get("train_runs", 0) + 1
    runs = st.session_state.get("train_runs", 0)
    tabs = st.tabs(labels, default=labels[1] if runs else None, key=f"tabs-{runs}")
    with tabs[0]:
        tab_idea(grid, make_field(cfg, grid))
    with tabs[1]:
        tab_training(grid, cfg, tcfg, device_label, start)
    with tabs[2]:
        tab_globe(grid)
    with tabs[3]:
        tab_maps(grid)
    with tabs[4]:
        tab_superres(grid)
    with tabs[5]:
        tab_coastline(grid)
    with tabs[6]:
        tab_encoder()
    with tabs[7]:
        tab_ortho()
    with tabs[8]:
        tab_buildings()
    with tabs[9]:
        tab_rate(grid)
    with tabs[10]:
        tab_export(grid)
