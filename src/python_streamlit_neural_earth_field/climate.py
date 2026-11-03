"""Monthly temperature anomalies since 1950 as one neural field f(x, y, z, t).

The elevation network stored one map.  Here the field has a fourth input, time:
GISTEMP v4 gives for every month since 1880 and every 2° cell how much warmer or
colder it was than the mean of 1951-1980.  From January 1950 on that is more than
900 maps of 90 × 180 cells.  A single network learns all of them at once; because
neighbouring months and neighbouring cells are alike, the ratio is far higher than
for the elevations -- the situation of Huang & Hoefler (chapter 11).

Time enters twice: as the date, scaled to −1 … 1 and fed through its own Fourier
features, and as the season, a point on a circle so that December lies next to
January.

The data: GISTEMP v4 with 1200 km smoothing, land and ocean combined, as provided
by NOAA PSL (GISTEMP Team, NASA Goddard Institute for Space Studies).
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import cache_dir, data_dir, download_etopo
from .encoding import FourierFeatures, lonlat_to_xyz

GISTEMP_URL = "https://downloads.psl.noaa.gov/Datasets/gistemp/combined/1200km/air.2x2.1200.mon.anom.comb.nc"
GISTEMP_FILE = "gistemp_air.2x2.1200.mon.anom.comb.nc"
GISTEMP_MB = 338
ATTRIBUTION = "GISTEMP Team, NASA Goddard Institute for Space Studies (GISTEMP v4), data provided by NOAA PSL"
START_YEAR = 1950
FIRST_YEAR_IN_FILE = 1880


def gistemp_path() -> Path:
    return data_dir() / GISTEMP_FILE


def download_gistemp(progress: Callable[[int, int | None], None] | None = None) -> Path:
    path = gistemp_path()
    if not path.exists():
        download_etopo(progress, url=GISTEMP_URL, target=path)
    return path


@dataclass
class ClimateGrid:
    """Monthly anomalies in °C on the 2° grid; NaN where there is no value."""

    anomaly: np.ndarray  # (months, 90, 180) float32, first row = north
    lat: np.ndarray  # (90,) cell centres, north → south
    lon: np.ndarray  # (180,) cell centres, −179 … 179
    start_year: int = START_YEAR

    @property
    def n_months(self) -> int:
        return self.anomaly.shape[0]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.anomaly)

    @property
    def raw_bytes(self) -> int:
        """Two bytes per value (hundredths of a degree fit into int16), like the elevations."""
        return self.anomaly.size * 2

    def year_of(self, month: int) -> float:
        return self.start_year + (month + 0.5) / 12

    def annual(self, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean of every complete year: (years, (years, 90, 180)); NaN where a cell has no value at all."""
        values = self.anomaly if values is None else values
        n_years = self.n_months // 12
        years = np.arange(self.start_year, self.start_year + n_years)
        with warnings.catch_warnings():  # cells without any value in a year stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(values[: n_years * 12].reshape(n_years, 12, *values.shape[1:]), axis=1)
        return years, means

    def global_mean(self, maps: np.ndarray) -> np.ndarray:
        """Area-weighted mean of each map over the cells with a value."""
        w = np.cos(np.deg2rad(self.lat))[:, None] * np.ones((1, self.lon.size))
        ok = np.isfinite(maps)
        return np.where(ok, maps, 0).reshape(len(maps), -1) @ w.ravel() / (ok.reshape(len(maps), -1) @ w.ravel())

    def trend(self, values: np.ndarray | None = None, first: int = 1950, last: int = 2025) -> np.ndarray:
        """Linear trend of the annual means in °C per decade, for every cell."""
        years, means = self.annual(values)
        keep = (years >= first) & (years <= last)
        years, means = years[keep], means[keep]
        x = years - years.mean()
        with np.errstate(invalid="ignore"):
            return np.sum(x[:, None, None] * (means - means.mean(axis=0)), axis=0) / np.sum(x**2) * 10


def load_climate(start_year: int = START_YEAR, use_cache: bool = True) -> ClimateGrid:
    cache = cache_dir() / f"gistemp_{start_year}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return ClimateGrid(d["anomaly"], d["lat"], d["lon"], start_year)
    import netCDF4  # imported locally so the app also starts without data

    with netCDF4.Dataset(gistemp_path()) as ds:
        first = (start_year - FIRST_YEAR_IN_FILE) * 12
        anomaly = np.ma.filled(ds["air"][first:].astype(np.float32), np.nan)
        lat = np.asarray(ds["lat"][:], dtype=np.float32)
        lon = np.asarray(ds["lon"][:], dtype=np.float32)
    lon = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon)  # −179 … 179 like the other maps of the app
    grid = ClimateGrid(anomaly[:, :, order], lat, lon[order], start_year)
    if use_cache:
        np.savez(cache, anomaly=grid.anomaly, lat=grid.lat, lon=grid.lon)
    return grid


# --------------------------------------------------------------------------- the field


@dataclass
class ClimateConfig:
    n_space: int = 128  # Fourier frequencies of the place
    sigma_space: float = 4.0  # waves per unit of the unit sphere
    n_time: int = 64  # Fourier frequencies of the date
    sigma_time: float = 20.0  # waves across the whole period (−1 … 1)
    hidden: int = 128
    layers: int = 4
    seed: int = 0


CLIMATE_PRESETS: dict[str, ClimateConfig] = {
    "Small": ClimateConfig(hidden=64),
    "Medium": ClimateConfig(hidden=128),
    "Large": ClimateConfig(hidden=256),
}


class ClimateField(nn.Module):
    """(point on the unit sphere, date, season) → anomaly in °C.

    The place gets the Fourier features of chapter 5 on (x, y, z), the date its own
    Fourier features, and the season enters as (cos, sin) of the month.  The first
    month and the number of months belong to the 'file', like the corner of the
    orthophoto.
    """

    def __init__(self, cfg: ClimateConfig, n_months: int, start_year: int = START_YEAR):
        super().__init__()
        self.cfg = cfg
        self.n_months = n_months
        self.start_year = start_year
        self.space = FourierFeatures(3, cfg.n_space, cfg.sigma_space, cfg.seed)
        self.time = FourierFeatures(1, cfg.n_time, cfg.sigma_time, cfg.seed + 1)
        dims = [self.space.out_dim + self.time.out_dim + 2] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)

    def date(self, month: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Month index (0 = January of the start year) → date in −1 … 1 and season on the circle."""
        tau = (month.float() + 0.5) / self.n_months * 2 - 1
        angle = 2 * math.pi * (month % 12).float() / 12
        return tau[:, None], torch.stack([torch.cos(angle), torch.sin(angle)], -1)

    def forward(self, xyz: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        tau, season = self.date(month)
        h = torch.cat([self.space(xyz), self.time(tau), season], -1)
        return self.head(self.body(h)).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Weights (float16 by default) + seed + first year + number of months."""
        return self.n_parameters() * bytes_per_param + 4 + 4 + 4

    @torch.no_grad()
    def render(self, grid: ClimateGrid, months: range | None = None) -> np.ndarray:
        """The anomalies as the network sees them, for every cell of ``grid`` and every month in ``months``."""
        device = next(self.parameters()).device
        months = range(grid.n_months) if months is None else months
        xyz = _grid_xyz(grid).to(device)
        out = np.empty((len(months), grid.lat.size, grid.lon.size), dtype=np.float32)
        was_training = self.training
        self.eval()
        for k, m in enumerate(months):
            month = torch.full((len(xyz),), m, device=device, dtype=torch.long)
            out[k] = self(xyz, month).cpu().numpy().reshape(grid.lat.size, grid.lon.size)
        if was_training:
            self.train()
        return out


def _grid_xyz(grid: ClimateGrid) -> torch.Tensor:
    lat, lon = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    return lonlat_to_xyz(torch.from_numpy(lon.ravel()), torch.from_numpy(lat.ravel())).float()


def build_climate_field(cfg: ClimateConfig, grid: ClimateGrid) -> ClimateField:
    return ClimateField(cfg, grid.n_months, grid.start_year)


# --------------------------------------------------------------------------- training and scores


@dataclass
class ClimateTrainConfig:
    steps: int = 2000
    batch_size: int = 32768
    lr: float = 2e-3
    lr_final_factor: float = 0.02
    eval_every: int = 50
    seed: int = 0


@dataclass
class ClimateProgress:
    step: int
    total: int
    rmse: float  # area-weighted, °C, on the last batch
    elapsed_s: float


def train_climate(field: ClimateField, grid: ClimateGrid, cfg: ClimateTrainConfig,
                  device: torch.device) -> Iterator[ClimateProgress]:
    """Overfits the field to all months; places are drawn by area, as in chapter 2."""
    torch.manual_seed(cfg.seed)
    field.to(device).train()
    xyz = _grid_xyz(grid).to(device)
    target = torch.from_numpy(np.nan_to_num(grid.anomaly)).to(device).reshape(grid.n_months, -1)
    month, cell = torch.nonzero(torch.from_numpy(grid.valid).reshape(grid.n_months, -1), as_tuple=True)
    w = torch.from_numpy(np.cos(np.deg2rad(grid.lat))).repeat_interleave(grid.lon.size)[cell]
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    optimiser = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    started = time.perf_counter()
    for step in range(cfg.steps):
        t = step / max(cfg.steps - 1, 1)
        for group in optimiser.param_groups:
            group["lr"] = cfg.lr * (cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t)))
        k = torch.multinomial(w, cfg.batch_size, replacement=True, generator=gen)
        m, c = month[k].to(device), cell[k].to(device)
        loss = torch.mean((field(xyz[c], m) - target[m, c]) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            yield ClimateProgress(step + 1, cfg.steps, loss.item() ** 0.5, time.perf_counter() - started)
    field.eval()


@dataclass
class ClimateScores:
    rmse_month: float  # area-weighted error of single months, °C
    rmse_global_year: float  # error of the global annual means, °C
    rmse_trend: float  # error of the trend map 1950-2025, °C per decade
    mean_trend: float  # area-weighted mean trend of the data, °C per decade

    @classmethod
    def compare(cls, grid: ClimateGrid, predicted: np.ndarray) -> "ClimateScores":
        ok = grid.valid
        w = np.cos(np.deg2rad(grid.lat))[None, :, None] * np.ones_like(grid.anomaly)
        err = np.where(ok, predicted - grid.anomaly, 0)
        rmse = float(np.sqrt(np.sum(w * err**2) / np.sum(w * ok)))
        _, truth_years = grid.annual()
        _, pred_years = grid.annual(np.where(ok, predicted, np.nan))
        g = float(np.sqrt(np.mean((grid.global_mean(truth_years) - grid.global_mean(pred_years)) ** 2)))
        tr_t, tr_p = grid.trend(), grid.trend(np.where(ok, predicted, np.nan))
        both = np.isfinite(tr_t) & np.isfinite(tr_p)
        wt = np.cos(np.deg2rad(grid.lat))[:, None] * np.ones((1, grid.lon.size))
        trend_rmse = float(np.sqrt(np.sum(wt[both] * (tr_t[both] - tr_p[both]) ** 2) / np.sum(wt[both])))
        mean_trend = float(np.sum(wt[both] * tr_t[both]) / np.sum(wt[both]))
        return cls(rmse, g, trend_rmse, mean_trend)


def line_per_cell(grid: ClimateGrid) -> np.ndarray:
    """The classic baseline: for every cell its mean plus a straight line through time (2 numbers per cell)."""
    t = np.arange(grid.n_months, dtype=np.float32)[:, None, None]
    ok = grid.valid
    n = np.maximum(ok.sum(axis=0), 1)
    t_mean = np.where(ok, t, 0).sum(axis=0) / n
    d_mean = np.nansum(grid.anomaly, axis=0) / n
    slope = np.nansum(np.where(ok, (t - t_mean) * (grid.anomaly - d_mean), 0), axis=0) / np.maximum(
        np.where(ok, (t - t_mean) ** 2, 0).sum(axis=0), 1e-9)
    return (d_mean + slope * (t - t_mean)).astype(np.float32)
