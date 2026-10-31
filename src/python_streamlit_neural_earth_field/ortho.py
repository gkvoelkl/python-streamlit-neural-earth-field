"""An aerial photo as a neural field: f(E, N) → (R, G, B).

The globe is gone here.  A tile of one kilometre is flat for all practical
purposes, so the network gets the metric coordinates of a map projection --
ETRS89 / UTM zone 32N (EPSG:25832), in which Bavaria publishes its digital
orthophotos -- instead of a point on the sphere.  Three outputs instead of one.

The data: DOP40 of the Bavarian surveying administration, 40 cm per pixel,
1 km × 1 km per tile, free under CC BY 4.0 (Bayerische Vermessungsverwaltung –
www.geodaten.bayern.de).  The default tile shows the old town of Regensburg
with the cathedral, the Stone Bridge and the Danube.
"""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import cache_dir, data_dir, download_etopo
from .encoding import FourierFeatures

TILE = "32726_5434"  # lower-left corner in km: E 726, N 5434 -- Regensburg old town
TILE_URL = "https://download1.bayernwolke.de/a/dop40/data/{tile}.tif"
TILE_MB = 18
TILE_M = 1000.0  # edge length of a tile in metres
ATTRIBUTION = "Bayerische Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0)"

# pixels per edge → metres per pixel
RESOLUTIONS = {2500: 0.4, 1250: 0.8, 500: 2.0}

# details to zoom in on: (east, north) in metres
PLACES = {
    "Cathedral": (726470, 5434247),
    "Stone Bridge": (726376, 5434577),
    "Old Town Hall": (726184, 5434258),
}


def tile_path(tile: str = TILE) -> Path:
    d = data_dir() / "dop"
    d.mkdir(exist_ok=True)
    return d / f"{tile}.tif"


def download_tile(tile: str = TILE, progress: Callable[[int, int | None], None] | None = None) -> Path:
    path = tile_path(tile)
    if not path.exists():
        download_etopo(progress, url=TILE_URL.format(tile=tile), target=path)
    return path


# --------------------------------------------------------------------------- the image


@dataclass
class Orthophoto:
    """An RGB image on a regular grid of UTM coordinates (pixel centres)."""

    rgb: np.ndarray  # (n, n, 3) uint8, first row = north
    east: np.ndarray  # (n,) pixel centres in metres, west → east
    north: np.ndarray  # (n,) pixel centres in metres, north → south
    tile: str = TILE

    @property
    def n(self) -> int:
        return self.rgb.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.rgb.shape[0] * self.rgb.shape[1]

    @property
    def metres_per_pixel(self) -> float:
        return float(self.east[1] - self.east[0])

    @property
    def raw_bytes(self) -> int:
        """Uncompressed: one byte per colour and pixel."""
        return self.rgb.size

    def crop(self, east: float, north: float, size_m: float) -> tuple[slice, slice]:
        """Rows and columns of a square of ``size_m`` around a point."""
        half = size_m / 2 / self.metres_per_pixel
        i = int(np.argmin(np.abs(self.north - north)))
        j = int(np.argmin(np.abs(self.east - east)))
        i0 = int(np.clip(i - half, 0, self.n - 2 * half))
        j0 = int(np.clip(j - half, 0, self.n - 2 * half))
        return slice(i0, i0 + int(2 * half)), slice(j0, j0 + int(2 * half))


def load_orthophoto(n: int = 1250, tile: str = TILE, use_cache: bool = True) -> Orthophoto:
    """The tile, reduced to ``n`` × ``n`` pixels by averaging (n must divide 2500)."""
    cache = cache_dir() / f"dop_{tile}_{n}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return Orthophoto(d["rgb"], d["east"], d["north"], tile)

    from PIL import Image

    with Image.open(tile_path(tile)) as im:
        rgb = np.asarray(im.convert("RGB"))
        scale_x, scale_y = im.tag_v2[33550][:2]  # GeoTIFF ModelPixelScale
        _, _, _, e0, n0, _ = im.tag_v2[33922]  # GeoTIFF ModelTiepoint: upper-left corner
    full = rgb.shape[0]
    if full % n:
        raise ValueError(f"{n} pixels do not divide the tile of {full} pixels")
    f = full // n
    if f > 1:
        rgb = rgb.reshape(n, f, n, f, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    step_e, step_n = scale_x * f, scale_y * f
    east = e0 + step_e * (np.arange(n) + 0.5)
    north = n0 - step_n * (np.arange(n) + 0.5)
    out = Orthophoto(rgb, east, north, tile)
    np.savez_compressed(cache, rgb=out.rgb, east=out.east, north=out.north)
    return out


# --------------------------------------------------------------------------- the network


@dataclass
class OrthoConfig:
    n_features: int = 256  # Fourier frequencies
    sigma: float = 20.0  # bandwidth, in waves per tile edge
    hidden: int = 256
    layers: int = 4
    fourier_seed: int = 0


ORTHO_PRESETS: dict[str, OrthoConfig] = {
    "Small": OrthoConfig(n_features=128, sigma=20.0, hidden=128, layers=4),
    "Medium": OrthoConfig(n_features=256, sigma=20.0, hidden=256, layers=4),
    "Large": OrthoConfig(n_features=256, sigma=20.0, hidden=512, layers=5),
}


class OrthoField(nn.Module):
    """(E, N) in metres → (R, G, B) in 0 … 255.

    The coordinates are first scaled to −1 … 1 across the tile; the corner of the
    tile is part of the 'file' (two numbers), just like the seed of the Fourier
    matrix.  The colours come out of a sigmoid, so they can never leave the range.
    """

    def __init__(self, cfg: OrthoConfig, east0: float, north0: float, size_m: float = TILE_M):
        super().__init__()
        self.cfg = cfg
        self.origin = (float(east0), float(north0))  # plain floats: MPS has no float64
        self.size_m = size_m
        self.encoding = FourierFeatures(2, cfg.n_features, cfg.sigma / 2, cfg.fourier_seed)
        dims = [self.encoding.out_dim] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 3)

    def scale(self, en: torch.Tensor) -> torch.Tensor:
        """Metres → −1 … 1 within the tile (in float64 on the CPU: UTM northings have 7 digits)."""
        device = next(self.parameters()).device
        xy = (en.to("cpu", torch.float64) - torch.tensor(self.origin, dtype=torch.float64)) / self.size_m * 2 - 1
        return xy.to(device, torch.float32)

    def forward_unit(self, xy: torch.Tensor) -> torch.Tensor:
        """Scaled coordinates → colours in 0 … 1."""
        return torch.sigmoid(self.head(self.body(self.encoding(xy))))

    def forward(self, en: torch.Tensor) -> torch.Tensor:
        return self.forward_unit(self.scale(en)) * 255

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Weights (float16 by default) + corner of the tile (2 × float64) + seed."""
        return self.n_parameters() * bytes_per_param + 16 + 4

    @torch.no_grad()
    def render(self, photo: Orthophoto, rows: slice = slice(None), cols: slice = slice(None),
               upsample: int = 1) -> np.ndarray:
        """The image as the network sees it, optionally ``upsample`` times finer than the grid."""
        device = next(self.parameters()).device
        step = photo.metres_per_pixel / upsample
        east = photo.east[cols]
        north = photo.north[rows]
        east = east[0] - photo.metres_per_pixel / 2 + step * (np.arange(len(east) * upsample) + 0.5)
        north = north[0] + photo.metres_per_pixel / 2 - step * (np.arange(len(north) * upsample) + 0.5)
        e0, n0 = self.origin
        xy_e = torch.from_numpy(((east - e0) / self.size_m * 2 - 1).astype(np.float32)).to(device)
        xy_n = torch.from_numpy(((north - n0) / self.size_m * 2 - 1).astype(np.float32)).to(device)
        out = np.empty((len(north), len(east), 3), dtype=np.float32)
        chunk = max(1, 400_000 // len(east))
        was_training = self.training
        self.eval()
        for i in range(0, len(north), chunk):
            n = xy_n[i:i + chunk]
            xy = torch.stack([xy_e.expand(len(n), -1), n[:, None].expand(-1, len(east))], -1).reshape(-1, 2)
            out[i:i + len(n)] = (self.forward_unit(xy) * 255).cpu().numpy().reshape(len(n), len(east), 3)
        if was_training:
            self.train()
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def build_ortho_field(cfg: OrthoConfig, photo: Orthophoto) -> OrthoField:
    e0 = photo.east[0] - photo.metres_per_pixel / 2
    n0 = photo.north[-1] - photo.metres_per_pixel / 2
    return OrthoField(cfg, e0, n0, TILE_M)


# --------------------------------------------------------------------------- training and scores


@dataclass
class OrthoTrainConfig:
    steps: int = 3000
    batch_size: int = 65536
    lr: float = 2e-3
    lr_final_factor: float = 0.02
    warmup: int = 100
    eval_every: int = 100
    seed: int = 0


@dataclass
class OrthoProgress:
    step: int
    total: int
    psnr_db: float  # on a fixed sample of pixels
    elapsed_s: float


def train_ortho(field: OrthoField, photo: Orthophoto, cfg: OrthoTrainConfig,
                device: torch.device) -> Iterator[OrthoProgress]:
    """Overfits the field to this one image; every pixel counts the same."""
    torch.manual_seed(cfg.seed)
    field.to(device).train()
    n = photo.n
    # scaled pixel centres along both axes, kept on the device
    e_scaled = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2 - 1
    n_scaled = 1 - (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2
    target = torch.from_numpy(photo.rgb).to(device).float() / 255
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    eval_rows = torch.randint(0, n, (100_000,), generator=gen).to(device)
    eval_cols = torch.randint(0, n, (100_000,), generator=gen).to(device)
    optimiser = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    started = time.perf_counter()
    for step in range(cfg.steps):
        if step < cfg.warmup:
            lr = cfg.lr * (step + 1) / cfg.warmup
        else:
            t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
            lr = cfg.lr * (cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t)))
        for group in optimiser.param_groups:
            group["lr"] = lr
        rows = torch.randint(0, n, (cfg.batch_size,), generator=gen).to(device)
        cols = torch.randint(0, n, (cfg.batch_size,), generator=gen).to(device)
        pred = field.forward_unit(torch.stack([e_scaled[cols], n_scaled[rows]], -1))
        loss = torch.mean((pred - target[rows, cols]) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                p = field.forward_unit(torch.stack([e_scaled[eval_cols], n_scaled[eval_rows]], -1))
                mse = torch.mean((torch.round(p * 255) - target[eval_rows, eval_cols] * 255) ** 2).item()
            yield OrthoProgress(step + 1, cfg.steps, psnr(mse), time.perf_counter() - started)
    field.eval()


def psnr(mse: float, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB, as usual for images with 8 bits per colour."""
    return float(10 * np.log10(peak**2 / mse)) if mse > 0 else float("inf")


def image_psnr(truth: np.ndarray, pred: np.ndarray) -> float:
    return psnr(float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2)))


def jpeg_at_size(rgb: np.ndarray, max_bytes: int) -> tuple[np.ndarray, int, int]:
    """The best JPEG that is not larger than ``max_bytes``: (image, bytes, quality).

    Quality 1 is the floor; if even that is too large, it is returned anyway.
    """
    from PIL import Image

    image = Image.fromarray(rgb)

    def encode(q: int) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q)
        return buf.getvalue()

    lo, hi = 1, 95
    best = encode(lo)
    best_q = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode(mid)
        if len(data) <= max_bytes:
            best, best_q, lo = data, mid, mid + 1
        else:
            hi = mid - 1
    decoded = np.asarray(Image.open(io.BytesIO(best)).convert("RGB"))
    return decoded, len(best), best_q
