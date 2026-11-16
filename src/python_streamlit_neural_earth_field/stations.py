"""Weather stations: a temperature map of Germany from a few hundred points.

Until now every network learned a complete grid.  Here the data are points: the
mean annual temperature 1991-2020 at about 1 100 stations of the Deutscher
Wetterdienst.  Between the stations nobody knows the temperature -- so for the
first time the stations are split: the methods learn from four fifths and are
checked on the fifth they have never seen.

Compared are the classical ways to fill the gaps -- inverse distance weighting
and kriging, also with the station height -- and a small neural field f(x, y)
or f(x, y, height).  The second question: where has Germany warmed most between
1961-1990 and 1991-2020?

Data: DWD Climate Data Center, multi-annual station means (CC BY 4.0,
"Deutscher Wetterdienst"); the height of the map comes from ETOPO 2022.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from torch import nn

from .data import data_dir, download_etopo, source_path

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/multi_annual/"
PERIODS = {"1991-2020": "mean_91-20", "1961-1990": "mean_61-90"}
ATTRIBUTION = "Deutscher Wetterdienst (DWD Climate Data Center, CC BY 4.0); height of the map: ETOPO 2022"
LON0, LAT0 = 10.5, 51.0  # centre of the local map projection
KM_PER_DEG = 111.195

METHODS = ("IDW", "Kriging", "Kriging + height", "Network", "Network + height")


def _files(period: str) -> dict[str, str]:
    folder = PERIODS[period]
    return {name: f"{BASE_URL}{folder}/{name}" for name in (f"Temperatur_{period}.txt",
                                                             f"Temperatur_{period}_Stationsliste.txt")}


def station_dir() -> Path:
    return data_dir() / "dwd"


def missing() -> list[str]:
    return [name for period in PERIODS for name in _files(period) if not (station_dir() / name).exists()]


def download(progress: Callable[[int, int | None], None] | None = None) -> None:
    station_dir().mkdir(parents=True, exist_ok=True)
    for period in PERIODS:
        for name, url in _files(period).items():
            if not (station_dir() / name).exists():
                download_etopo(progress, url=url, target=station_dir() / name)


# --------------------------------------------------------------------------- the stations


def to_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A simple local projection: kilometres east and north of 10.5° E, 51° N (good to 1 % within Germany)."""
    return np.stack([(lon - LON0) * KM_PER_DEG * math.cos(math.radians(LAT0)), (lat - LAT0) * KM_PER_DEG], -1)


@dataclass
class Stations:
    """Stations with a mean annual temperature 1991-2020; ``old`` is NaN where 1961-1990 is missing."""

    ids: np.ndarray
    names: list[str]
    lon: np.ndarray
    lat: np.ndarray
    height: np.ndarray  # m above sea level
    temperature: np.ndarray  # °C, 1991-2020
    old: np.ndarray  # °C, 1961-1990

    @property
    def n(self) -> int:
        return self.ids.size

    @property
    def xy(self) -> np.ndarray:
        return to_km(self.lon, self.lat)

    @property
    def warming(self) -> np.ndarray:
        return self.temperature - self.old


def _read(path: Path) -> list[list[str]]:
    lines = path.read_bytes().decode("latin-1").splitlines()
    return [[c.strip() for c in line.split(";")] for line in lines[1:] if line.strip()]


def load_stations() -> Stations:
    def means(period: str) -> tuple[dict[int, float], dict[int, list[str]]]:
        values = {int(r[0]): float(r[15]) for r in _read(station_dir() / f"Temperatur_{period}.txt")}
        meta = {int(r[0]): r for r in _read(station_dir() / f"Temperatur_{period}_Stationsliste.txt")}
        return values, meta

    new, meta = means("1991-2020")
    old, _ = means("1961-1990")
    ids = np.array(sorted(i for i in new if i in meta))
    rows = [meta[i] for i in ids]
    return Stations(
        ids=ids,
        names=[r[1] for r in rows],
        lat=np.array([float(r[2]) for r in rows]),
        lon=np.array([float(r[3]) for r in rows]),
        height=np.array([float(r[4]) for r in rows]),
        temperature=np.array([new[i] for i in ids]),
        old=np.array([old.get(i, np.nan) for i in ids]),
    )


def split(n: int, test_share: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random stations for the test, the rest for training -- every method sees the same split."""
    order = np.random.default_rng(seed).permutation(n)
    k = int(round(n * test_share))
    return np.sort(order[k:]), np.sort(order[:k])


# --------------------------------------------------------------------------- classical methods


def idw(xy: np.ndarray, v: np.ndarray, query: np.ndarray, power: float = 2.0, k: int = 12) -> np.ndarray:
    """Inverse distance weighting: the mean of the ``k`` nearest stations, near ones counting more."""
    d, j = cKDTree(xy).query(query, k=min(k, len(xy)))
    w = 1 / np.maximum(d, 1e-6) ** power
    return np.sum(w * v[j], axis=1) / np.sum(w, axis=1)


def _spherical(h, nugget, sill, rng):
    h = np.asarray(h, dtype=np.float64)
    inside = nugget + sill * (1.5 * h / rng - 0.5 * (h / rng) ** 3)
    return np.where(h < rng, inside, nugget + sill)


@dataclass
class Variogram:
    """How different two stations are, on average, as a function of their distance (spherical model)."""

    nugget: float
    sill: float
    range_km: float
    lags: np.ndarray  # the empirical variogram the model was fitted to
    gamma: np.ndarray

    def __call__(self, h: np.ndarray) -> np.ndarray:
        return _spherical(h, self.nugget, self.sill, self.range_km)


def fit_variogram(xy: np.ndarray, v: np.ndarray, max_km: float = 300.0, bins: int = 20) -> Variogram:
    i, j = np.triu_indices(len(v), k=1)
    h = np.hypot(*(xy[i] - xy[j]).T)
    g = 0.5 * (v[i] - v[j]) ** 2
    edges = np.linspace(0, max_km, bins + 1)
    which = np.digitize(h, edges) - 1
    keep = [b for b in range(bins) if np.sum(which == b) > 10]
    lags = np.array([h[which == b].mean() for b in keep])
    gamma = np.array([g[which == b].mean() for b in keep])
    p0 = [gamma[0] * 0.5, max(gamma.max() - gamma[0] * 0.5, 1e-3), max_km / 2]
    (nugget, sill, rng), _ = curve_fit(_spherical, lags, gamma, p0=p0,
                                       bounds=([0, 1e-6, 5], [np.inf, np.inf, 5 * max_km]), maxfev=20000)
    return Variogram(float(nugget), float(sill), float(rng), lags, gamma)


def kriging(xy: np.ndarray, v: np.ndarray, query: np.ndarray, vario: Variogram | None = None,
            chunk: int = 20000) -> np.ndarray:
    """Ordinary kriging: weights from the variogram, so that the error is as small as possible on average."""
    vario = vario or fit_variogram(xy, v)
    n = len(v)
    a = np.ones((n + 1, n + 1))
    a[:n, :n] = vario(np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)))
    a[n, n] = 0
    np.fill_diagonal(a[:n, :n], 0)
    inv = np.linalg.inv(a)
    out = np.empty(len(query))
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk]
        b = np.ones((n + 1, len(q)))
        b[:n] = vario(np.hypot(*(xy[:, None] - q[None]).transpose(2, 0, 1)))
        w = inv @ b
        out[s:s + chunk] = w[:n].T @ v
    return out


@dataclass
class HeightTrend:
    """A straight line through temperature against height: the lapse rate."""

    a: float
    b: float  # °C per metre

    @classmethod
    def fit(cls, height: np.ndarray, v: np.ndarray) -> "HeightTrend":
        b, a = np.polyfit(height, v, 1)
        return cls(float(a), float(b))

    def __call__(self, height: np.ndarray) -> np.ndarray:
        return self.a + self.b * height


def kriging_with_height(xy, height, v, query, query_height) -> tuple[np.ndarray, HeightTrend]:
    """Regression kriging: the height explains what it can, kriging fills in the rest."""
    trend = HeightTrend.fit(height, v)
    rest = v - trend(height)
    return trend(query_height) + kriging(xy, rest, query), trend


# --------------------------------------------------------------------------- the neural field


@dataclass
class StationNetConfig:
    n_features: int = 64
    sigma: float = 0.5  # waves per 500 km
    hidden: int = 64
    layers: int = 3
    steps: int = 1000
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0


class StationNet(nn.Module):
    """f(x, y) or f(x, y, height): Fourier features of the place in km, the height as an extra input."""

    def __init__(self, cfg: StationNetConfig, with_height: bool):
        super().__init__()
        gen = torch.Generator().manual_seed(cfg.seed)
        self.register_buffer("B", torch.randn(2, cfg.n_features, generator=gen) * cfg.sigma, persistent=False)
        self.with_height = with_height
        dims = [2 * cfg.n_features + (1 if with_height else 0)] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, xy: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        p = 2 * math.pi * (xy / 500.0) @ self.B
        h = [torch.sin(p), torch.cos(p)] + ([height[:, None] / 1000.0] if self.with_height else [])
        return self.head(self.body(torch.cat(h, -1))).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def network(xy, height, v, query, query_height, cfg: StationNetConfig | None = None,
            with_height: bool = False) -> tuple[np.ndarray, StationNet]:
    """Trains on all training stations at once (full batch) and answers for the query points."""
    cfg = cfg or StationNetConfig()
    torch.manual_seed(cfg.seed)
    net = StationNet(cfg, with_height)
    mean, std = float(v.mean()), float(v.std())
    x = torch.tensor(xy, dtype=torch.float32)
    h = torch.tensor(height, dtype=torch.float32)
    y = torch.tensor((v - mean) / std, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = cfg.lr * 0.5 * (1 + math.cos(math.pi * step / cfg.steps))
        loss = torch.mean((net(x, h) - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(query, dtype=torch.float32), torch.tensor(query_height, dtype=torch.float32))
    return out.numpy() * std + mean, net


def predict(method: str, xy, height, v, query, query_height, cfg: StationNetConfig | None = None) -> np.ndarray:
    if method == "IDW":
        return idw(xy, v, query)
    if method == "Kriging":
        return kriging(xy, v, query)
    if method == "Kriging + height":
        return kriging_with_height(xy, height, v, query, query_height)[0]
    if method == "Network":
        return network(xy, height, v, query, query_height, cfg, with_height=False)[0]
    if method == "Network + height":
        return network(xy, height, v, query, query_height, cfg, with_height=True)[0]
    raise ValueError(method)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# --------------------------------------------------------------------------- the map


@dataclass
class GermanyGrid:
    """Map points every 0.05° with their height from ETOPO; only points within reach of a station count."""

    lon: np.ndarray  # (nx,)
    lat: np.ndarray  # (ny,)
    height: np.ndarray  # (ny, nx) m
    inside: np.ndarray  # (ny, nx) bool

    @property
    def points(self) -> np.ndarray:
        lon, lat = np.meshgrid(self.lon, self.lat)
        return to_km(lon.ravel(), lat.ravel())


def germany_grid(stations: Stations, step: float = 0.05, reach_km: float = 20.0) -> GermanyGrid:
    """Heights from ETOPO 2022 (1 arc-minute), averaged to ``step``; points farther than ``reach_km`` from every station are left out."""
    import netCDF4

    lon = np.arange(5.6, 15.3, step) + step / 2
    lat = np.arange(47.2, 55.1, step) + step / 2
    with netCDF4.Dataset(source_path()) as ds:
        src_lat = np.asarray(ds.variables["lat"][:])
        src_lon = np.asarray(ds.variables["lon"][:])
        r0, r1 = np.searchsorted(src_lat, [lat[0] - step, lat[-1] + step])
        c0, c1 = np.searchsorted(src_lon, [lon[0] - step, lon[-1] + step])
        z = np.asarray(ds.variables["z"][r0:r1, c0:c1], dtype=np.float32)
        zlat, zlon = src_lat[r0:r1], src_lon[c0:c1]
    rows = np.clip(np.searchsorted(zlat, lat), 0, len(zlat) - 1)
    cols = np.clip(np.searchsorted(zlon, lon), 0, len(zlon) - 1)
    k = int(round(step * 60))  # source cells per map cell
    height = np.empty((lat.size, lon.size), dtype=np.float32)
    for a, r in enumerate(rows):
        block = z[max(r - k // 2, 0):r + k - k // 2]
        height[a] = np.array([block[:, max(c - k // 2, 0):c + k - k // 2].mean() for c in cols])
    grid = GermanyGrid(lon, lat, height, np.zeros_like(height, dtype=bool))
    d, _ = cKDTree(stations.xy).query(grid.points)
    grid.inside = (d <= reach_km).reshape(height.shape) & (height > -5)
    return grid
