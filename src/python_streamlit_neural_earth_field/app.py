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
from .data import (
    ETOPO_BYTES,
    ETOPO_URL,
    ElevationGrid,
    download_etopo,
    load_etopo,
    source_available,
    source_path,
    thin,
)
from .metrics import Compression, evaluate, human_bytes
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


def fmt_ratio(r: float) -> str:
    return f"{r:.2f}×" if r < 10 else f"{r:,.0f}×"


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


@st.cache_data(show_spinner="Loading elevation grid …")
def get_grid(target_ny: int) -> ElevationGrid:
    return load_etopo(target_ny)


def make_field(cfg: FieldConfig, grid: ElevationGrid) -> NeuralField:
    coarse = grid.coarsen(cfg.context_ny).z if cfg.use_context else None
    return build_field(cfg, float(grid.z.mean()), float(grid.z.std()), coarse)


# --------------------------------------------------------------------------- UI


def sidebar() -> tuple[ElevationGrid, FieldConfig, TrainConfig, str, bool]:
    st.sidebar.title("🌍 Neural Elevation Field")

    st.sidebar.subheader("1 · Data")
    st.sidebar.caption("Source: ETOPO 2022 global relief model (NOAA NCEI), 1 arc-minute")

    res_label = st.sidebar.select_slider(
        "Grid resolution", options=list(RESOLUTIONS), value=DEFAULT_RESOLUTION
    )
    grid = get_grid(RESOLUTIONS[res_label])

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
    tcfg.steps = st.sidebar.slider("Steps", 200, 30000, 4000, 200)
    tcfg.batch_size = int(
        st.sidebar.select_slider(
            "Points per step", [4096, 8192, 16384, 32768, 65536, 131072], value=32768
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
        f"elevations from {grid.z.min():.0f} m to {grid.z.max():.0f} m"
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
                text=f"Step {p.step}/{p.total} · RMSE {p.scores.rmse:,.1f} m · "
                     f"{p.elapsed_s:.0f} s",
            )
            with metric_slot.container():
                m = p.scores
                a, b, c, d = st.columns(4)
                a.metric("RMSE", f"{m.rmse:,.1f} m")
                b.metric("weighted RMSE", f"{m.weighted_rmse:,.1f} m")
                c.metric("MAE", f"{m.mae:,.1f} m")
                d.metric("R²", f"{m.r2:.4f}")
            chart_slot.plotly_chart(
                viz.loss_curve(steps, rmses), use_container_width=True, key=f"loss{p.step}"
            )

        with st.spinner("Reconstructing the full grid …"):
            recon = field.predict_grid(grid.lon, grid.lat, device=device)
            scores = evaluate(grid.z, recon, grid.cell_weights())
        st.session_state.update(
            field=field, recon=recon, scores=scores, history=(steps, rmses),
            grid_shape=(grid.ny, grid.nx), trained_cfg=cfg,
        )
        st.success(
            "Done – the *Globe*, *Maps* and *Super-Resolution* tabs show the result."
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
    items = list(scores.to_dict().items())
    for row in (items[:4], items[4:]):
        cols = st.columns(4)
        for col, (name, value) in zip(cols, row):
            col.metric(name, f"{value:,.4f}" if abs(value) < 10 else f"{value:,.1f}")
    steps, rmses = st.session_state.history
    st.plotly_chart(viz.loss_curve(steps, rmses), use_container_width=True)
    budget_panel(grid, st.session_state.field)


def _require_result(grid: ElevationGrid) -> np.ndarray | None:
    if "field" not in st.session_state:
        st.info("Train first.")
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
    exaggeration = c1.slider("Vertical exaggeration", 1, 200, 40, help="1 = true to scale (and flat).")
    max_rows = c2.select_slider("Display grid", [90, 180, 240, 360], value=180)
    camera_lon = c3.slider("View direction (longitude)", -180, 180, 10)

    display = grid.coarsen(max_rows)
    recon_display = ElevationGrid(grid.lon, grid.lat, recon, "recon").coarsen(max_rows)
    zmin, zmax = float(grid.z.min()), float(grid.z.max())

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            viz.globe(display.lon, display.lat, display.z, title="Original (grid data)",
                      exaggeration=exaggeration, zmin=zmin, zmax=zmax, camera_lon=camera_lon),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.globe(recon_display.lon, recon_display.lat, recon_display.z,
                      title="Neural field", exaggeration=exaggeration,
                      zmin=zmin, zmax=zmax, camera_lon=camera_lon),
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
                              title="Original", zmin=zmin, zmax=zmax),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.elevation_map(recon_view.lon, recon_view.lat, recon_view.z,
                              title="Reconstruction", zmin=zmin, zmax=zmax),
            use_container_width=True,
        )

    a, b = st.columns([3, 2])
    with a:
        st.plotly_chart(viz.error_map(err_lon, err_lat, err_view), use_container_width=True)
    with b:
        st.plotly_chart(viz.error_histogram(err), use_container_width=True)
    if truth_view.ny < grid.ny:
        st.caption(
            f"The maps are thinned to {truth_view.ny} × {truth_view.nx} for drawing "
            f"(elevation maps averaged, error map subsampled). All metrics come from the "
            f"full {grid.ny} × {grid.nx} grid."
        )

    st.markdown(
        "**What to look for:** the error almost never sits in the open ocean, but along "
        "coasts, ocean trenches and mountain ridges – wherever the elevation changes "
        "sharply over a short distance. Those high frequencies are exactly what compression costs."
    )
    with st.expander("Detailed numbers"):
        scores = st.session_state.scores.to_dict()
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
                              zmin=zmin, zmax=zmax),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            viz.elevation_map(lon_q, lat_q, fine, title=f"Field queried ({n} × {n})",
                              zmin=zmin, zmax=zmax),
            use_container_width=True,
        )
    st.warning(
        "**To be honest:** this is not real super-resolution. The field interpolates "
        "smoothly between what it has learned – it doesn't invent details that aren't in "
        "the training data. It is useful because you can evaluate it at any coordinate "
        "without re-gridding (e.g. as a data loader that delivers grids of arbitrary "
        "resolution)."
    )


def tab_rate(grid: ElevationGrid) -> None:
    st.markdown(
        "Compression is always a trade-off: **file size versus error**. This "
        "rate-distortion curve trains several network sizes on the same grid and "
        "shows what you get at each point."
    )
    c1, c2, c3 = st.columns(3)
    steps = int(c1.select_slider("Steps per network", [300, 600, 1000, 2000], value=600))
    widths = c2.multiselect("Network widths", [16, 24, 32, 48, 64, 96, 128, 192, 256],
                            default=[16, 32, 64, 128])
    with_context = c3.checkbox("Also with context field (90 × 180)", value=True)
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
                    "weighted RMSE [m]": last.scores.weighted_rmse, "PSNR [dB]": last.scores.psnr_db,
                })
        bar.empty()
        st.session_state.rd_rows = rows

    rows = st.session_state.get("rd_rows")
    if not rows:
        return
    st.plotly_chart(viz.rate_distortion(rows), use_container_width=True)
    st.dataframe(
        [{**r, "File size": human_bytes(r["File size"]), "Ratio": fmt_ratio(r["Ratio"]),
          "weighted RMSE [m]": round(r["weighted RMSE [m]"], 1), "PSNR [dB]": round(r["PSNR [dB]"], 1)}
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
    if source_available():
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
    st.caption("f(lon, lat) → elevation · the weights are the compressed data")

    labels = ["💡 Idea & Budget", "📉 Training", "🌐 Globe 3D", "🗺️ Maps & Error",
              "🔍 Super-Resolution", "📐 Rate & Error", "💾 Export"]
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
        tab_rate(grid)
    with tabs[6]:
        tab_export(grid)
