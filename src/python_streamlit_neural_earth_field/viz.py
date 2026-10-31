"""Visualisation: 3D globe, maps, error plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

EARTH_RADIUS_M = 6_371_000.0

_OCEAN = ["#06203f", "#0b3a66", "#155e8c", "#2a86b0", "#5fb0cc", "#a8d4e4"]
_LAND = ["#1d6b3a", "#4b9b45", "#8fbb54", "#cdc07a", "#b08a5a", "#8a6a53", "#cfcfcf", "#ffffff"]


def hypsometric_colorscale(zmin: float, zmax: float) -> list[list]:
    """Blue below, green/brown/white above sea level -- zero point exactly at 0 m."""
    if not (zmin < 0 < zmax):
        colors = _OCEAN + _LAND if zmin < 0 else _LAND
        pos = np.linspace(0, 1, len(colors))
        return [[float(p), c] for p, c in zip(pos, colors)]

    split = float(-zmin / (zmax - zmin))
    scale: list[list] = []
    for p, c in zip(np.linspace(0.0, split * 0.999, len(_OCEAN)), _OCEAN):
        scale.append([float(p), c])
    for p, c in zip(np.linspace(split, 1.0, len(_LAND)), _LAND):
        scale.append([float(p), c])
    return scale


_SEA_DISTANCE = ["#0b3a66", "#2a86b0", "#a8d4e4", "#eef6f9"]
_LAND_DISTANCE = ["#f6efe0", "#d9b77a", "#a8702f", "#5c3613"]


def colorscale(zmin: float, zmax: float, unit: str = "m") -> list[list]:
    """Hypsometric colours for elevations, blue/brown with white at the coast for distances."""
    if unit == "m" or not (zmin < 0 < zmax):
        return hypsometric_colorscale(zmin, zmax)
    split = float(-zmin / (zmax - zmin))
    scale = [[float(p), c] for p, c in zip(np.linspace(0.0, split * 0.999, 4), _SEA_DISTANCE)]
    return scale + [[float(p), c] for p, c in zip(np.linspace(split, 1.0, 4), _LAND_DISTANCE)]


def _sphere_mesh(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, exaggeration: float):
    """Spherical coordinates → xyz, with a closed seam at ±180°."""
    lon_w = np.concatenate([lon, lon[:1] + 360.0])
    z_w = np.concatenate([z, z[:, :1]], axis=1)

    lon2d, lat2d = np.meshgrid(np.deg2rad(lon_w), np.deg2rad(lat))
    r = 1.0 + exaggeration * z_w / EARTH_RADIUS_M
    cos_lat = np.cos(lat2d)
    return (
        r * cos_lat * np.cos(lon2d),
        r * cos_lat * np.sin(lon2d),
        r * np.sin(lat2d),
        z_w,
    )


def globe(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    *,
    title: str = "",
    exaggeration: float = 40.0,
    zmin: float | None = None,
    zmax: float | None = None,
    height: int = 520,
    camera_lon: float = 10.0,
    camera_lat: float = 25.0,
    unit: str = "m",
) -> go.Figure:
    zmin = float(np.min(z)) if zmin is None else zmin
    zmax = float(np.max(z)) if zmax is None else zmax
    x, y, zc, colour = _sphere_mesh(lon, lat, z, exaggeration)

    fig = go.Figure(
        go.Surface(
            x=x,
            y=y,
            z=zc,
            surfacecolor=colour,
            colorscale=colorscale(zmin, zmax, unit),
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=unit, thickness=12, len=0.7),
            lighting=dict(ambient=0.72, diffuse=0.85, specular=0.08, roughness=0.9),
            lightposition=dict(x=2, y=1, z=1.5),
            hovertemplate=f"%{{surfacecolor:.0f}} {unit}<extra></extra>",
        )
    )

    d = 2.0
    clat, clon = np.deg2rad(camera_lat), np.deg2rad(camera_lon)
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=0, r=0, t=32 if title else 0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(
                eye=dict(
                    x=d * np.cos(clat) * np.cos(clon),
                    y=d * np.cos(clat) * np.sin(clon),
                    z=d * np.sin(clat),
                ),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        uirevision="globe",
    )
    return fig


def elevation_map(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    *,
    title: str = "",
    zmin: float | None = None,
    zmax: float | None = None,
    height: int = 340,
    unit: str = "m",
) -> go.Figure:
    zmin = float(np.min(z)) if zmin is None else zmin
    zmax = float(np.max(z)) if zmax is None else zmax
    fig = go.Figure(
        go.Heatmap(
            x=lon,
            y=lat,
            z=z,
            colorscale=colorscale(zmin, zmax, unit),
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=unit, thickness=12),
            hovertemplate=f"%{{x:.2f}}°, %{{y:.2f}}° → %{{z:.0f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=40, r=10, t=36 if title else 10, b=30),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Latitude"),
        xaxis=dict(title="Longitude"),
    )
    return fig


def error_map(
    lon: np.ndarray,
    lat: np.ndarray,
    err: np.ndarray,
    *,
    title: str = "Error (reconstruction − original)",
    height: int = 340,
    unit: str = "m",
) -> go.Figure:
    limit = float(np.nanpercentile(np.abs(err), 99.5)) or 1.0  # NaN: no data (e.g. at sea)
    fig = go.Figure(
        go.Heatmap(
            x=lon,
            y=lat,
            z=err,
            colorscale="RdBu",
            reversescale=True,
            zmin=-limit,
            zmax=limit,
            colorbar=dict(title=unit, thickness=12),
            hovertemplate=f"%{{x:.2f}}°, %{{y:.2f}}° → %{{z:+.{1 if unit == '°C' else 0}f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=10, t=36, b=30),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Latitude"),
        xaxis=dict(title="Longitude"),
    )
    return fig


def log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-3-5 steps between lo and hi, for log axes with full labels.

    Plotly's own log ticks label the minor steps with a single digit ("2", "3"),
    which reads as 2 m and 3 m instead of 2000 m and 3000 m.
    Returns an empty list when fewer than two such steps fall inside the range:
    then the range spans well under a decade and Plotly's own labels are fine.
    """
    if not (lo > 0 and hi > 0):
        return []
    ticks = [m * 10.0**e for e in range(-2, 8) for m in (1, 2, 3, 5)]
    if sum(lo <= t <= hi for t in ticks) < 2:
        return []
    return [t for t in ticks if lo / 1.5 <= t <= hi * 1.5]


def log_axis(title: str, lo: float, hi: float) -> dict:
    """Layout for a log axis with full tick labels (see ``log_ticks``)."""
    axis = dict(title=title, type="log")
    ticks = log_ticks(lo, hi)
    if ticks:
        axis.update(tickvals=ticks, ticktext=[f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in ticks])
    return axis


def loss_curve(steps: list[int], rmse: list[float], height: int = 300, unit: str = "m") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=steps, y=rmse, mode="lines", line=dict(width=2, color="#2a86b0"),
            hovertemplate=f"step %{{x}}<br>RMSE %{{y:,.1f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=10, t=10, b=40),
        xaxis=dict(title="Training step"),
        yaxis=log_axis(f"RMSE on the evaluation sample [{unit}]", min(rmse, default=0), max(rmse, default=0)),
    )
    return fig


def error_histogram(err: np.ndarray, height: int = 300, unit: str = "m") -> go.Figure:
    sample = err.ravel()
    if sample.size > 400_000:
        sample = np.random.default_rng(0).choice(sample, 400_000, replace=False)
    # Binned here rather than by Plotly, so the count range is known for the log axis.
    counts, edges = np.histogram(sample, bins=120)
    centres = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(
        go.Bar(
            x=centres, y=counts, width=np.diff(edges), marker_color="#2a86b0",
            hovertemplate=f"%{{x:,.0f}} {unit}: %{{y:,}} grid points<extra></extra>",
        )
    )
    positive = counts[counts > 0]
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=10, t=10, b=40),
        xaxis=dict(title=f"Error [{unit}]"),
        yaxis=log_axis("Number of grid points", float(positive.min(initial=1)), float(positive.max(initial=1))),
        bargap=0.02,
    )
    return fig


def rate_distortion(rows: list[dict], height: int = 420, unit: str = "m") -> go.Figure:
    """Compression ratio versus error -- the core curve of compression."""
    fig = go.Figure()
    palette = {"without context": "#2a86b0", "with context": "#c0622a"}
    for variant in dict.fromkeys(r["Variant"] for r in rows):
        pts = sorted((r for r in rows if r["Variant"] == variant), key=lambda r: r["Ratio"])
        fig.add_trace(
            go.Scatter(
                x=[p["Ratio"] for p in pts],
                y=[p[f"weighted RMSE [{unit}]"] for p in pts],
                mode="lines+markers+text",
                name=variant,
                text=[f" {p['Width']}" for p in pts],
                textposition="top center",
                line=dict(width=2, color=palette.get(variant)),
                marker=dict(size=9),
                hovertemplate=(
                    f"Width %{{text}}<br>Ratio %{{x:.0f}}×<br>weighted RMSE %{{y:.0f}} {unit}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=20, b=50),
        xaxis=dict(title="Compression ratio (higher = smaller file)", type="log"),
        yaxis=dict(title=f"weighted RMSE [{unit}] (lower = better)"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.add_annotation(
        x=0.99, y=0.02, xref="paper", yref="paper", showarrow=False,
        text="bottom right would be ideal", font=dict(size=11, color="#888"),
    )
    return fig


def coastline_map(
    lon: np.ndarray,
    lat: np.ndarray,
    background: np.ndarray,
    recon: np.ndarray,
    lines: list[np.ndarray],
    *,
    title: str = "",
    unit: str = "m",
    height: int = 520,
) -> go.Figure:
    """The field as background, the true coastline in black, the recovered 0 line in orange."""
    limit = float(np.percentile(np.abs(background), 98)) or 1.0
    xs: list[float | None] = []
    ys: list[float | None] = []
    for line in lines:  # only the parts inside the window, so the map keeps its extent
        inside = (
            (line[:, 0] >= lon[0]) & (line[:, 0] <= lon[-1])
            & (line[:, 1] >= lat[0]) & (line[:, 1] <= lat[-1])
        )
        xs += [*np.where(inside, line[:, 0], np.nan), None]
        ys += [*np.where(inside, line[:, 1], np.nan), None]
    fig = go.Figure(
        [
            go.Heatmap(
                x=lon, y=lat, z=background, colorscale=colorscale(-limit, limit, unit),
                zmin=-limit, zmax=limit, colorbar=dict(title=unit, thickness=12),
                hovertemplate=f"%{{x:.2f}}°, %{{y:.2f}}° → %{{z:.0f}} {unit}<extra></extra>",
            ),
            go.Scatter(
                x=xs, y=ys, mode="lines", name="Coastline (Natural Earth)",
                line=dict(color="#111111", width=1.2), hoverinfo="skip",
            ),
            go.Contour(
                x=lon, y=lat, z=recon, name="0 line of the network",
                contours=dict(start=0, end=0, size=1, coloring="none"),
                line=dict(color="#e8590c", width=2.2), showscale=False, showlegend=True,
                hoverinfo="skip",
            ),
        ]
    )
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=40, r=10, t=60 if title else 40, b=30),
        # no axis titles: with the fixed aspect ratio Plotly would draw them across the map
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[lat[0], lat[-1]], ticksuffix="° N",
                   constrain="domain"),
        xaxis=dict(range=[lon[0], lon[-1]], ticksuffix="° E", constrain="domain"),
        legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"),
    )
    return fig


def temperature_map(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    *,
    title: str = "",
    zmin: float = -30.0,
    zmax: float = 35.0,
    points: np.ndarray | None = None,
    height: int = 340,
) -> go.Figure:
    """Temperatures on land (NaN at sea); ``points`` (n, 2) marks the training places."""
    fig = go.Figure(
        go.Heatmap(
            x=lon, y=lat, z=z, colorscale="RdYlBu", reversescale=True, zmin=zmin, zmax=zmax,
            colorbar=dict(title="°C", thickness=12),
            hovertemplate="%{x:.2f}°, %{y:.2f}° → %{z:.1f} °C<extra></extra>",
        )
    )
    if points is not None:
        fig.add_trace(go.Scattergl(
            x=points[:, 0], y=points[:, 1], mode="markers", name="training places",
            marker=dict(size=3, color="#1f2933"), hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=40, r=10, t=36 if title else 10, b=30),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Latitude", range=[-90, 85], constrain="domain"),
        xaxis=dict(title="Longitude", range=[-180, 180], constrain="domain"),
        plot_bgcolor="#eef4fb",
    )
    return fig


def encoder_bars(names: list[str], rmse: list[float], baseline: float, height: int = 320) -> go.Figure:
    """Error per encoder, best at the top; the dashed line is the error of always guessing the mean."""
    order = np.argsort(rmse)[::-1]
    fig = go.Figure(go.Bar(
        x=[rmse[i] for i in order], y=[names[i] for i in order], orientation="h",
        marker_color="#2b6cb0", text=[f"{rmse[i]:.2f} °C" for i in order], textposition="outside",
        hovertemplate="%{y}: %{x:.2f} °C<extra></extra>",
    ))
    fig.add_vline(x=baseline, line_dash="dash", line_color="#e8833a",
                  annotation_text=f"always the mean: {baseline:.1f} °C", annotation_position="top")
    fig.update_layout(
        height=height, margin=dict(l=10, r=40, t=30, b=40),
        xaxis=dict(title="Weighted RMSE on the other places [°C]", range=[0, max(max(rmse), baseline) * 1.15]),
    )
    return fig


_ROOF = [[0.0, "#e9e2d6"], [0.35, "#d8b48a"], [0.7, "#b8683f"], [1.0, "#7a2e1e"]]


def building_mesh(enh: np.ndarray, faces: np.ndarray, *, ground: float, title: str = "",
                  height: int = 420, zmax: float | None = None) -> go.Figure:
    """Buildings as a triangle mesh, coloured by height above the lowest ground; true scale in all directions."""
    h = enh[:, 2] - ground
    zmax = float(h.max()) if zmax is None else zmax
    fig = go.Figure(
        go.Mesh3d(
            x=enh[:, 0], y=enh[:, 1], z=enh[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=h, colorscale=_ROOF, cmin=0, cmax=zmax,
            colorbar=dict(title="m", thickness=12, len=0.6),
            flatshading=True,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.05, roughness=0.9),
            lightposition=dict(x=-1, y=-2, z=3),
            hovertemplate="E %{x:.0f} · N %{y:.0f} · %{z:.0f} m<extra></extra>",
        )
    )
    span = enh.max(axis=0) - enh.min(axis=0)
    ratio = span / span[:2].max()
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=0, r=0, t=32 if title else 0, b=0),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=float(ratio[0]), y=float(ratio[1]), z=float(ratio[2])),
            xaxis=dict(title="E", showticklabels=False),
            yaxis=dict(title="N", showticklabels=False),
            zaxis=dict(title="", showticklabels=False),
            camera=dict(eye=dict(x=-0.75, y=-1.05, z=0.55), center=dict(x=0, y=0, z=-0.15)),
        ),
    )
    return fig


def distance_slice(east: np.ndarray, north: np.ndarray, d: np.ndarray, *, limit: float,
                   title: str = "", height: int = 380) -> go.Figure:
    """A horizontal slice of a signed distance field; the black line is the 0-level (the walls)."""
    fig = go.Figure(
        go.Heatmap(
            x=east, y=north, z=np.clip(d, -limit, limit), zmin=-limit, zmax=limit,
            colorscale="RdBu", reversescale=False,
            colorbar=dict(title="m", thickness=12, len=0.8),
            hovertemplate="E %{x:.0f} · N %{y:.0f} · %{z:.1f} m<extra></extra>",
        )
    )
    fig.add_trace(go.Contour(
        x=east, y=north, z=d, contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color="black", width=1.5), showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title=title or None, height=height, margin=dict(l=0, r=0, t=32 if title else 0, b=0),
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor="x"),
    )
    return fig
