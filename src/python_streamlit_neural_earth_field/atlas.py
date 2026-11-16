"""Several climate quantities in one network: f(x, y, z) → temperature, precipitation, elevation.

So far every field had one output (or three colours that belong together).  Here
one place gets 25 numbers: the mean temperature of each of the twelve months, the
mean precipitation of each month and the elevation.  The question of chapter 18:
does one network with 25 outputs do better than three separate networks of the
same total size -- do related quantities help each other?  And how do you weigh
degrees against millimetres against metres in one loss?

The data: University of Delaware v5.01 (Willmott & Matsuura), long-term monthly
means 1981-2010 on 0.5° land cells, as provided by NOAA PSL ("usage
restrictions: none"); the elevation is ETOPO 2022 averaged to the same cells.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import cache_dir, data_dir, download_etopo, load_etopo
from .embedding import TEMPERATURE_FILE, TEMPERATURE_URL
from .encoding import FourierFeatures, lonlat_to_xyz

PRECIPITATION_URL = "https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/precip.mon.v501.ltm.1981-2010.nc"
PRECIPITATION_FILE = "udel_precip.mon.v501.ltm.1981-2010.nc"
FILE_MB = 19
ATTRIBUTION = ("Willmott & Matsuura, University of Delaware v5.01, data provided by NOAA PSL; "
               "elevation ETOPO 2022 (NOAA NCEI)")

QUANTITIES = ("Temperature", "Precipitation", "Elevation")
UNITS = {"Temperature": "°C", "Precipitation": "mm", "Elevation": "m"}
N_OUT = {"Temperature": 12, "Precipitation": 12, "Elevation": 1}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def paths() -> dict[str, Path]:
    return {"Temperature": data_dir() / TEMPERATURE_FILE, "Precipitation": data_dir() / PRECIPITATION_FILE}


def missing() -> list[str]:
    return [q for q, p in paths().items() if not p.exists()]


def download(progress: Callable[[int, int | None], None] | None = None) -> None:
    urls = {"Temperature": TEMPERATURE_URL, "Precipitation": PRECIPITATION_URL}
    for q, p in paths().items():
        if not p.exists():
            download_etopo(progress, url=urls[q], target=p)


# --------------------------------------------------------------------------- the data


@dataclass
class Atlas:
    """Every land cell with its 25 numbers."""

    lon: np.ndarray  # (n,)
    lat: np.ndarray  # (n,)
    temperature: np.ndarray  # (n, 12) °C, mean of each month
    precipitation: np.ndarray  # (n, 12) mm per month
    elevation: np.ndarray  # (n, 1) m, mean of the 0.5° cell

    @property
    def n(self) -> int:
        return self.lon.size

    def values(self, quantity: str) -> np.ndarray:
        return {"Temperature": self.temperature, "Precipitation": self.precipitation,
                "Elevation": self.elevation}[quantity]

    @property
    def weights(self) -> np.ndarray:
        """Cell area ∝ cos(latitude)."""
        return np.cos(np.deg2rad(self.lat))

    @property
    def raw_bytes(self) -> int:
        """Two bytes per value (int16 with a suitable step), as for the elevations."""
        return self.n * 25 * 2

    def grid(self, values: np.ndarray, step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One value per place back on a lon/lat grid (NaN at sea) -- for maps."""
        lon = np.arange(-180 + step / 2, 180, step)
        lat = np.arange(-90 + step / 2, 90, step)
        z = np.full((lat.size, lon.size), np.nan, dtype=np.float32)
        z[np.rint((self.lat - lat[0]) / step).astype(int), np.rint((self.lon - lon[0]) / step).astype(int)] = values
        return lon, lat, z


def load_atlas(use_cache: bool = True) -> Atlas:
    cache = cache_dir() / "atlas_udel_etopo.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return Atlas(d["lon"], d["lat"], d["temperature"], d["precipitation"], d["elevation"])

    import xarray as xr

    with xr.open_dataset(paths()["Temperature"], decode_times=False) as ds:
        air = ds["air"].values.astype(np.float32)  # (12, 360, 720) °C
        lat = ds["lat"].values.astype(np.float64)
        lon = ds["lon"].values.astype(np.float64)
    with xr.open_dataset(paths()["Precipitation"], decode_times=False) as ds:
        rain = ds["precip"].values.astype(np.float32) * 10  # cm → mm
    lon = np.where(lon > 180, lon - 360, lon)
    lon2, lat2 = np.meshgrid(lon, lat)
    ok = np.isfinite(air).all(0) & np.isfinite(rain).all(0)

    etopo = load_etopo(360)  # 0.5° cells, south → north, −180 → 180
    row = np.rint((lat2[ok] - etopo.lat[0]) / 0.5).astype(int)
    col = np.rint((lon2[ok] - etopo.lon[0]) / 0.5).astype(int)
    out = Atlas(lon2[ok].astype(np.float32), lat2[ok].astype(np.float32),
                air[:, ok].T.copy(), rain[:, ok].T.copy(), etopo.z[row, col][:, None].astype(np.float32))
    np.savez_compressed(cache, lon=out.lon, lat=out.lat, temperature=out.temperature,
                        precipitation=out.precipitation, elevation=out.elevation)
    return out


# --------------------------------------------------------------------------- scaling


SCALINGS = ("Standardised", "Raw units")


@dataclass
class Scaler:
    """How the 25 numbers are turned into what the network learns, and back.

    *Standardised*: every quantity gets mean 0 and spread 1 (chapter 1), and
    precipitation is first taken as a square root, because a few very wet places
    would otherwise dominate.  *Raw units*: °C, mm and m as they are -- then a
    metre of elevation weighs as much as a degree of temperature.
    """

    mode: str = "Standardised"
    sqrt_precipitation: bool = True

    def fit(self, atlas: Atlas) -> "Scaler":
        w = atlas.weights[:, None]
        self.mean, self.std = {}, {}
        for q in QUANTITIES:
            v = self._forward_transform(q, atlas.values(q))
            m = float(np.sum(w * v) / np.sum(w * np.ones_like(v)))
            s = float(np.sqrt(np.sum(w * (v - m) ** 2) / np.sum(w * np.ones_like(v))))
            self.mean[q], self.std[q] = (m, s) if self.mode == "Standardised" else (0.0, 1.0)
        return self

    def _forward_transform(self, quantity: str, v: np.ndarray) -> np.ndarray:
        if quantity == "Precipitation" and self.sqrt_precipitation and self.mode == "Standardised":
            return np.sqrt(np.maximum(v, 0))
        return v

    def encode(self, quantity: str, v: np.ndarray) -> np.ndarray:
        return (self._forward_transform(quantity, v) - self.mean[quantity]) / self.std[quantity]

    def decode(self, quantity: str, y: np.ndarray) -> np.ndarray:
        v = y * self.std[quantity] + self.mean[quantity]
        if quantity == "Precipitation" and self.sqrt_precipitation and self.mode == "Standardised":
            return np.maximum(v, 0) ** 2
        return v


# --------------------------------------------------------------------------- the networks


@dataclass
class AtlasConfig:
    n_features: int = 128
    sigma: float = 8.0
    hidden: int = 256
    layers: int = 4
    seed: int = 0


ATLAS_PRESETS = {
    "Small": AtlasConfig(hidden=128),
    "Medium": AtlasConfig(hidden=256),
    "Large": AtlasConfig(hidden=512),
}

LAYOUTS = ("One network, 25 outputs", "Three separate networks")


class MLP(nn.Module):
    """Fourier features of the place on the sphere, then ``layers`` hidden layers, then ``n_out`` outputs."""

    def __init__(self, cfg: AtlasConfig, hidden: int, n_out: int):
        super().__init__()
        self.features = FourierFeatures(3, cfg.n_features, cfg.sigma, cfg.seed)
        dims = [self.features.out_dim] + [hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], n_out)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.features(xyz)))


def mlp_parameters(cfg: AtlasConfig, hidden: int, n_out: int) -> int:
    d = 2 * cfg.n_features
    return d * hidden + hidden + (cfg.layers - 1) * (hidden * hidden + hidden) + hidden * n_out + n_out


def width_for(cfg: AtlasConfig, budget: int, n_out: int) -> int:
    """The widest hidden layer whose network does not exceed ``budget`` weights."""
    h = 1
    while mlp_parameters(cfg, h + 1, n_out) <= budget:
        h += 1
    return h


class AtlasField(nn.Module):
    """Either one network for all 25 numbers, or one network per quantity with a third of the weights each.

    ``quantities`` can be smaller than all three -- e.g. temperature alone with
    the whole budget, to see what sharing costs or brings.
    """

    def __init__(self, cfg: AtlasConfig, layout: str = LAYOUTS[0], quantities: tuple[str, ...] = QUANTITIES):
        super().__init__()
        self.cfg, self.layout, self.quantities = cfg, layout, quantities
        n_all = sum(N_OUT[q] for q in quantities)
        budget = mlp_parameters(cfg, cfg.hidden, n_all)
        if layout == LAYOUTS[0] or len(quantities) == 1:
            self.nets = nn.ModuleDict({"all": MLP(cfg, cfg.hidden, n_all)})
        else:
            share = budget // len(quantities)
            self.nets = nn.ModuleDict({q: MLP(cfg, width_for(cfg, share, N_OUT[q]), N_OUT[q]) for q in quantities})

    def forward(self, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        if "all" in self.nets:
            y = self.nets["all"](xyz)
            out, k = {}, 0
            for q in self.quantities:
                out[q] = y[:, k:k + N_OUT[q]]
                k += N_OUT[q]
            return out
        return {q: self.nets[q](xyz) for q in self.quantities}

    def widths(self) -> dict[str, int]:
        return {name: net.head.in_features for name, net in self.nets.items()}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Weights in float16, plus seed and the mean and spread of every quantity."""
        return self.n_parameters() * bytes_per_param + 4 + len(self.quantities) * 8

    @torch.no_grad()
    def predict(self, atlas: Atlas, scaler: Scaler, chunk: int = 65536) -> dict[str, np.ndarray]:
        device = next(self.parameters()).device
        xyz = _xyz(atlas)
        parts: dict[str, list[np.ndarray]] = {q: [] for q in self.quantities}
        was_training = self.training
        self.eval()
        for i in range(0, atlas.n, chunk):
            y = self(xyz[i:i + chunk].to(device))
            for q in self.quantities:
                parts[q].append(y[q].float().cpu().numpy())
        if was_training:
            self.train()
        return {q: scaler.decode(q, np.concatenate(parts[q])) for q in self.quantities}


def _xyz(atlas: Atlas) -> torch.Tensor:
    return lonlat_to_xyz(torch.from_numpy(atlas.lon), torch.from_numpy(atlas.lat)).float()


# --------------------------------------------------------------------------- training and scores


@dataclass
class AtlasTrainConfig:
    steps: int = 2000
    batch_size: int = 16384
    lr: float = 2e-3
    lr_final_factor: float = 0.02
    eval_every: int = 50
    seed: int = 0


@dataclass
class AtlasProgress:
    step: int
    total: int
    loss: float
    elapsed_s: float


def train_atlas(field: AtlasField, atlas: Atlas, scaler: Scaler, cfg: AtlasTrainConfig,
                device: torch.device) -> Iterator[AtlasProgress]:
    """Every quantity counts the same in the loss: the mean of the three mean squared errors.

    Places are drawn by area (chapter 2).  With separate networks the three losses
    do not touch each other, so the sum trains each network on its own.
    """
    torch.manual_seed(cfg.seed)
    field.to(device).train()
    xyz = _xyz(atlas).to(device)
    target = {q: torch.from_numpy(scaler.encode(q, atlas.values(q)).astype(np.float32)).to(device)
              for q in field.quantities}
    w = torch.from_numpy(atlas.weights.astype(np.float64))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    optimiser = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    started = time.perf_counter()
    for step in range(cfg.steps):
        t = step / max(cfg.steps - 1, 1)
        for group in optimiser.param_groups:
            group["lr"] = cfg.lr * (cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t)))
        k = torch.multinomial(w, cfg.batch_size, replacement=True, generator=gen).to(device)
        y = field(xyz[k])
        loss = sum(torch.mean((y[q] - target[q][k]) ** 2) for q in field.quantities) / len(field.quantities)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            yield AtlasProgress(step + 1, cfg.steps, loss.item(), time.perf_counter() - started)
    field.eval()


@dataclass
class AtlasScores:
    """Area-weighted RMSE per quantity, over all land cells (and all months)."""

    rmse: dict[str, float]

    @classmethod
    def compare(cls, atlas: Atlas, predicted: dict[str, np.ndarray]) -> "AtlasScores":
        w = atlas.weights[:, None]
        out = {}
        for q, p in predicted.items():
            err = p - atlas.values(q)
            out[q] = float(np.sqrt(np.sum(w * err**2) / np.sum(w * np.ones_like(err))))
        return cls(out)


def annual(atlas: Atlas, quantity: str, values: np.ndarray | None = None) -> np.ndarray:
    """Annual mean temperature or annual total precipitation per place."""
    v = atlas.values(quantity) if values is None else values
    return v.sum(1) if quantity == "Precipitation" else v.mean(1)
