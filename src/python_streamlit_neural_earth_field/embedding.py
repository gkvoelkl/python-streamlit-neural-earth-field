"""Location encoders: a place becomes a vector of numbers that many tasks can use.

The elevation network *stores* one field.  A location encoder is meant to be
*transferred*: it turns (lon, lat) into, say, 256 numbers, and a small second
network learns a new task from these numbers with only a few training places.
How good an encoder is shows in how well that small network does.

The task here: the mean annual air temperature over land (University of Delaware
v5.01, 1981-2010 climatology on 0.5° cells, Willmott & Matsuura, no restrictions).  The encoders compared:
plain lon/lat, Fourier features, spherical harmonics, the pretrained SatCLIP
encoder (Klemmer et al., MIT licence), the last hidden layer of the app's own
elevation network -- and, as a reminder of what domain knowledge is worth, plain
lon/lat plus the elevation that this network predicts.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import cache_dir, data_dir, download_etopo
from .encoding import FourierFeatures, SphericalHarmonics, lonlat_to_xyz

TEMPERATURE_URL = "https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/air.mon.v501.ltm.1981-2010.nc"
TEMPERATURE_FILE = "udel_air.mon.v501.ltm.1981-2010.nc"
SATCLIP_URL = "https://huggingface.co/microsoft/SatCLIP-ResNet18-L10/resolve/main/satclip-resnet18-l10.ckpt"
SATCLIP_FILE = "satclip-resnet18-l10.ckpt"
SATCLIP_MB = 57
TEMPERATURE_MB = 19

HIDDEN_LAYER = "Elevation network (hidden layer)"
WITH_ELEVATION = "lon/lat + elevation"
ENCODERS = ("lon/lat", "Fourier features", "Spherical harmonics", "SatCLIP", HIDDEN_LAYER, WITH_ELEVATION)
NEEDS_ELEVATION_NETWORK = (HIDDEN_LAYER, WITH_ELEVATION)


def temperature_path() -> Path:
    return data_dir() / TEMPERATURE_FILE


def satclip_path() -> Path:
    return data_dir() / SATCLIP_FILE


def download(path: Path, url: str, progress: Callable[[int, int | None], None] | None = None) -> None:
    if not path.exists():
        download_etopo(progress, url=url, target=path)


# --------------------------------------------------------------------------- the task


@dataclass
class Temperature:
    """Mean annual air temperature in °C for every land cell with data, as a list of places."""

    lon: np.ndarray
    lat: np.ndarray
    celsius: np.ndarray

    @property
    def n(self) -> int:
        return self.celsius.size

    @property
    def lonlat(self) -> np.ndarray:
        return np.stack([self.lon, self.lat], axis=1).astype(np.float32)

    @property
    def weights(self) -> np.ndarray:
        """Cell area ∝ cos(latitude): every km² of land counts the same."""
        return np.cos(np.deg2rad(self.lat))

    def grid(self, values: np.ndarray | None = None, step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The places back on a lon/lat grid (NaN at sea) -- for maps."""
        values = self.celsius if values is None else values
        lon = np.arange(-180 + step / 2, 180, step)
        lat = np.arange(-90 + step / 2, 90, step)
        z = np.full((lat.size, lon.size), np.nan, dtype=np.float32)
        z[np.rint((self.lat - lat[0]) / step).astype(int), np.rint((self.lon - lon[0]) / step).astype(int)] = values
        return lon, lat, z


def load_temperature(use_cache: bool = True) -> Temperature:
    """Annual mean of the twelve monthly long-term means, land cells only (the data has no sea)."""
    cache = cache_dir() / "temperature_udel.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return Temperature(d["lon"], d["lat"], d["celsius"])

    import xarray as xr

    with xr.open_dataset(temperature_path(), decode_times=False) as ds:
        celsius = ds["air"].mean("time").values  # already in °C
        lat = ds["lat"].values.astype(np.float64)
        lon = ds["lon"].values.astype(np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    lon2, lat2 = np.meshgrid(lon, lat)
    ok = np.isfinite(celsius)
    out = Temperature(lon2[ok].astype(np.float32), lat2[ok].astype(np.float32), celsius[ok].astype(np.float32))
    np.savez_compressed(cache, lon=out.lon, lat=out.lat, celsius=out.celsius)
    return out


def sample_places(data: Temperature, n: int, seed: int = 0) -> np.ndarray:
    """Indices of ``n`` training places, drawn by area so that Siberia is not over-represented."""
    rng = np.random.default_rng(seed)
    p = data.weights / data.weights.sum()
    return rng.choice(data.n, size=min(n, data.n), replace=False, p=p)


# --------------------------------------------------------------------------- SatCLIP


class SatCLIPHarmonics(nn.Module):
    """The spherical harmonics exactly as SatCLIP was trained with them (L = 10 → 100 numbers).

    They follow the expressions SatCLIP generated with sympy, including two quirks
    that the pretrained weights depend on: the angles are measured from the south
    pole and from −180°, and the terms with m = 0 carry the factor √((2l+1)·π/4).
    """

    def __init__(self, legendre_polys: int = 10):
        super().__init__()
        self.L = legendre_polys
        self.out_dim = legendre_polys**2

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        phi = torch.deg2rad(lonlat[:, 0] + 180)
        x = torch.cos(torch.deg2rad(lonlat[:, 1] + 90))
        s = torch.sqrt(torch.clamp(1 - x * x, min=0))
        features = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                p = _legendre(l, abs(m), x, s)
                if m == 0:
                    features.append(math.sqrt((2 * l + 1) * math.pi / 4) * p)
                    continue
                k = abs(m)
                norm = math.sqrt(2 * (2 * l + 1) / (4 * math.pi) * math.factorial(l - k) / math.factorial(l + k))
                features.append(norm * p * (torch.cos(k * phi) if m > 0 else torch.sin(k * phi)))
        return torch.stack(features, dim=-1)


def _legendre(l: int, m: int, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Associated Legendre function P_l^m(x) without the Condon-Shortley sign."""
    pmm = torch.ones_like(x)
    for i in range(1, m + 1):
        pmm = pmm * (2 * i - 1) * s
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    for ll in range(m + 2, l + 1):
        pmm, pmmp1 = pmmp1, ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
    return pmmp1


class SatCLIPEncoder(nn.Module):
    """SatCLIP's location encoder: spherical harmonics, then a small SIREN → 256 numbers.

    Pretrained by Klemmer et al. (2025) so that a place and a Sentinel-2 image of it
    end up with similar numbers.  Only this location half of SatCLIP is used here.
    """

    def __init__(self, legendre_polys: int = 10, hidden: int = 512, out: int = 256, w0_first: float = 30.0):
        super().__init__()
        self.harmonics = SatCLIPHarmonics(legendre_polys)
        self.layers = nn.ModuleList([nn.Linear(self.harmonics.out_dim, hidden), nn.Linear(hidden, hidden)])
        self.last = nn.Linear(hidden, out)
        self.w0 = (w0_first, 1.0)
        self.out_dim = out

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        h = self.harmonics(lonlat)
        for layer, w0 in zip(self.layers, self.w0):
            h = torch.sin(w0 * layer(h))
        return self.last(h)


def load_satclip(path: Path | None = None) -> SatCLIPEncoder:
    ckpt = torch.load(path or satclip_path(), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    enc = SatCLIPEncoder(hp["legendre_polys"], hp["capacity"], hp["embed_dim"])
    sd = ckpt["state_dict"]
    prefix = "model.location.nnet."
    names = {"layers.0": "layers.0", "layers.1": "layers.1", "last_layer": "last"}
    enc.load_state_dict({
        f"{names[k.rsplit('.', 1)[0][len(prefix):]]}.{k.rsplit('.', 1)[1]}": v.float()
        for k, v in sd.items() if k.startswith(prefix)
    })
    return enc.eval()


# --------------------------------------------------------------------------- the encoders


@torch.no_grad()
def encode(name: str, lonlat: np.ndarray, elevation_field: nn.Module | None = None, chunk: int = 20_000) -> np.ndarray:
    """(n, 2) degrees → (n, d) numbers for one of the ``ENCODERS``."""
    x = torch.from_numpy(np.ascontiguousarray(lonlat, dtype=np.float32))
    if name == "lon/lat":
        return np.stack([lonlat[:, 0] / 180, lonlat[:, 1] / 90], axis=1).astype(np.float32)
    if name == "Fourier features":
        f = FourierFeatures(3, n_features=128, sigma=2.0, seed=0)
        fn = lambda c: f(lonlat_to_xyz(c[:, 0], c[:, 1]))
    elif name == "Spherical harmonics":
        f = SphericalHarmonics(3, degree=9)  # 100 numbers, like SatCLIP's L = 10
        fn = lambda c: f(lonlat_to_xyz(c[:, 0], c[:, 1]))
    elif name == "SatCLIP":
        f = load_satclip()
        fn = f
    elif name in NEEDS_ELEVATION_NETWORK:
        if elevation_field is None:
            raise ValueError("the elevation network has to be trained first")
        field = copy.deepcopy(elevation_field).to("cpu").float().eval()  # leave the app's network where it is
        if name == HIDDEN_LAYER:
            fn = lambda c: _hidden_layer(field, c)
        else:  # the network's output in km next to the two coordinates
            fn = lambda c: torch.stack([c[:, 0] / 180, c[:, 1] / 90, field(c) / 1000], dim=-1)
    else:
        raise ValueError(f"unknown encoder {name}")
    return torch.cat([fn(x[i:i + chunk]) for i in range(0, len(x), chunk)]).numpy()


def _hidden_layer(field: nn.Module, lonlat: torch.Tensor) -> torch.Tensor:
    """What the elevation network's head sees: the output of its last hidden layer."""
    h = field.encoding(field.transform(lonlat))
    if field.context is not None:
        context = (field.context(lonlat) - field.z_mean) / field.z_std
        h = torch.cat([h, context.unsqueeze(-1)], dim=-1)
    return field.body(h)


# --------------------------------------------------------------------------- the small network on top


@dataclass
class HeadConfig:
    hidden: int = 64
    layers: int = 2
    steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


def fit_and_predict(features: np.ndarray, target: np.ndarray, train: np.ndarray, cfg: HeadConfig = HeadConfig()) -> np.ndarray:
    """Trains a small MLP on the training places only and predicts every place.

    The features are standardised with the training places' mean and spread, so
    that encoders with very different scales start on equal terms.
    """
    torch.manual_seed(cfg.seed)
    mu, sd = features[train].mean(0), features[train].std(0) + 1e-6
    x_all = torch.from_numpy(((features - mu) / sd).astype(np.float32))
    y_mu, y_sd = target[train].mean(), target[train].std() + 1e-6
    x, y = x_all[train], torch.from_numpy(((target[train] - y_mu) / y_sd).astype(np.float32))

    dims = [x.shape[1]] + [cfg.hidden] * cfg.layers
    blocks: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        blocks += [nn.Linear(a, b), nn.ReLU()]
    net = nn.Sequential(*blocks, nn.Linear(dims[-1], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.steps):
        opt.zero_grad()
        loss = torch.mean((net(x).squeeze(-1) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(x_all).squeeze(-1).numpy() * y_sd + y_mu


def weighted_rmse(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (truth - pred) ** 2) / np.sum(weights)))
