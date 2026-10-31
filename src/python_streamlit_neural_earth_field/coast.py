"""Vector data as a field: the signed distance to the coast (Natural Earth).

The coastline is vector data -- a list of points connected in order.  To store
it in a neural field like the elevations, it first becomes a field: for every
grid cell the distance to the nearest coast, positive on land and negative at
sea (the same sign as the elevation).  The coast is then the line where this
field is exactly zero, and it can be recovered from any network that has
learned the field.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import EARTH_RADIUS_KM, ElevationGrid, cache_dir, data_dir, download_etopo

NE_SCALE = "50m"  # 1:50 million -- detailed enough for cells of 9 km and coarser
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_{scale}_{kind}.geojson"
)
NE_KINDS = ("land", "coastline")
DENSIFY_KM = 2.0  # the distance is measured to points on the line at most this far apart


def ne_path(kind: str, scale: str = NE_SCALE) -> Path:
    return data_dir() / f"ne_{scale}_{kind}.geojson"


def natural_earth_available(scale: str = NE_SCALE) -> bool:
    return all(ne_path(kind, scale).exists() for kind in NE_KINDS)


def download_natural_earth(
    progress: Callable[[int, int | None], None] | None = None, scale: str = NE_SCALE
) -> None:
    """Downloads land polygons and coastlines of Natural Earth (public domain, about 3 MB)."""
    for kind in NE_KINDS:
        if not ne_path(kind, scale).exists():
            download_etopo(progress, url=NE_URL.format(scale=scale, kind=kind), target=ne_path(kind, scale))


def _lines(geometry: dict) -> list[np.ndarray]:
    """All rings or line strings of a GeoJSON geometry as (n, 2) arrays of lon/lat."""
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "LineString":
        parts = [coords]
    elif kind in ("MultiLineString", "Polygon"):
        parts = coords
    elif kind == "MultiPolygon":
        parts = [ring for polygon in coords for ring in polygon]
    else:
        raise ValueError(f"unexpected geometry {kind}")
    return [np.asarray(p, dtype=np.float64)[:, :2] for p in parts]


def read_lines(path: Path) -> list[np.ndarray]:
    with open(path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    return [line for feature in features for line in _lines(feature["geometry"])]


@dataclass
class Coastline:
    """The vector data: land polygons (for inside/outside) and coastlines (for the distance)."""

    land: list[np.ndarray]
    coast: list[np.ndarray]

    @property
    def n_points(self) -> int:
        return sum(len(line) for line in self.coast)

    @property
    def raw_bytes(self) -> int:
        """The coastline as bare float32 lon/lat pairs -- what the vector data costs."""
        return self.n_points * 2 * 4


def load_coastline(scale: str = NE_SCALE) -> Coastline:
    return Coastline(land=read_lines(ne_path("land", scale)), coast=read_lines(ne_path("coastline", scale)))


def land_mask(rings: list[np.ndarray], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """True where a cell centre lies inside the polygons (even-odd rule, row by row)."""
    a = np.concatenate([r[:-1] for r in rings])
    b = np.concatenate([r[1:] for r in rings])
    mask = np.zeros((lat.size, lon.size), dtype=bool)
    for i, phi in enumerate(lat):
        crosses = (a[:, 1] <= phi) != (b[:, 1] <= phi)
        if not crosses.any():
            continue
        pa, pb = a[crosses], b[crosses]
        x = pa[:, 0] + (phi - pa[:, 1]) * (pb[:, 0] - pa[:, 0]) / (pb[:, 1] - pa[:, 1])
        x.sort()
        mask[i] = np.searchsorted(x, lon) % 2 == 1
    return mask


def _unit_vectors(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def densify(line: np.ndarray, step_km: float = DENSIFY_KM) -> np.ndarray:
    """Points along the line no more than ``step_km`` apart."""
    p = _unit_vectors(line[:, 0], line[:, 1])
    angles = np.arccos(np.clip(np.sum(p[1:] * p[:-1], axis=1), -1.0, 1.0))
    pieces = [p[:1]]
    for start, end, angle in zip(p[:-1], p[1:], angles):
        n = max(1, int(np.ceil(angle * EARTH_RADIUS_KM / step_km)))
        t = np.arange(1, n + 1)[:, None] / n
        if angle < 1e-9:
            pieces.append(np.repeat(end[None], n, axis=0))
            continue
        # evenly spaced along the great circle (spherical linear interpolation)
        pieces.append((np.sin((1 - t) * angle) * start + np.sin(t * angle) * end) / np.sin(angle))
    return np.concatenate(pieces)


def coast_distance_km(coast: list[np.ndarray], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle distance from every cell centre to the nearest point of the coastline."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.concatenate([densify(line) for line in coast]))
    out = np.empty((lat.size, lon.size), dtype=np.float32)
    for i, phi in enumerate(lat):
        chord, _ = tree.query(_unit_vectors(lon, np.full_like(lon, phi)))
        out[i] = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * EARTH_RADIUS_KM
    return out


def cell_centres(ny: int) -> tuple[np.ndarray, np.ndarray]:
    nx = 2 * ny
    lon = (-180 + (np.arange(nx) + 0.5) * 360 / nx).astype(np.float32)
    lat = (-90 + (np.arange(ny) + 0.5) * 180 / ny).astype(np.float32)
    return lon, lat


def load_coast_distance(target_ny: int = 360, use_cache: bool = True) -> ElevationGrid:
    """Signed distance to the coast in km on a grid with ``target_ny`` rows: land +, sea −."""
    cache = cache_dir() / f"coast_{NE_SCALE}_{target_ny}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        lon, lat, z = d["lon"], d["lat"], d["z"]
    else:
        coastline = load_coastline()
        lon, lat = cell_centres(target_ny)
        distance = coast_distance_km(coastline.coast, lon, lat)
        land = land_mask(coastline.land, lon.astype(np.float64), lat.astype(np.float64))
        z = np.where(land, distance, -distance).astype(np.float32)
        np.savez_compressed(cache, lon=lon, lat=lat, z=z)
    return ElevationGrid(
        lon=lon, lat=lat, z=z, source=f"Natural Earth 1:{NE_SCALE[:-1]} million → {lat.size}x{lon.size}",
        quantity="Distance to the coast", unit="km",
    )


def coast_cells(land: np.ndarray) -> np.ndarray:
    """Cells with a neighbour on the other side of the coast (left/right wraps around at ±180°)."""
    out = np.zeros_like(land, dtype=bool)
    across = land != np.roll(land, -1, axis=1)
    out |= across | np.roll(across, 1, axis=1)
    across = land[1:] != land[:-1]
    out[1:] |= across
    out[:-1] |= across
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cumulative, q), values.size - 1)])


@dataclass
class CoastScores:
    correct_side: float  # share of the Earth's area on the correct side of the coast
    correct_side_at_coast: float  # the same, only for cells on the true coast
    offset_median_km: float  # how far the recovered coast lies from the true one
    offset_p90_km: float

    @classmethod
    def compare(cls, distance: ElevationGrid, predicted_land: np.ndarray) -> "CoastScores":
        """``distance`` is the true signed distance, ``predicted_land`` what a network says is land."""
        w = distance.cell_weights()
        land = distance.z > 0
        correct = (predicted_land == land) * w
        at_coast = coast_cells(land)
        recovered = coast_cells(predicted_land)
        offsets = np.abs(distance.z[recovered]).astype(np.float64)
        weights = w[recovered].astype(np.float64)
        if offsets.size == 0:
            nan = float("nan")
            return cls(float(correct.mean()), float(correct[at_coast].sum() / w[at_coast].sum()), nan, nan)
        return cls(
            correct_side=float(correct.mean()),
            correct_side_at_coast=float(correct[at_coast].sum() / w[at_coast].sum()),
            offset_median_km=_weighted_quantile(offsets, weights, 0.5),
            offset_p90_km=_weighted_quantile(offsets, weights, 0.9),
        )
