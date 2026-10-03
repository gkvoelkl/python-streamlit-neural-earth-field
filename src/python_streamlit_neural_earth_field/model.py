"""The neural field itself: f(lon, lat) → elevation in metres.

The weights *are* the compressed data.  So everything needed for the
reconstruction belongs in the model and in the size calculation:
the normalisation constants and -- if used -- the coarse context field.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .encoding import FourierFeatures, Identity, InputTransform, SphericalHarmonics


@dataclass
class FieldConfig:
    input_mode: str = "xyz"  # "xyz" | "lonlat"
    encoding: str = "fourier"  # "fourier" | "spherical" | "none"
    n_features: int = 128  # number of Fourier frequencies
    sigma: float = 4.0  # bandwidth of the Fourier features
    fourier_seed: int = 0
    sh_degree: int = 16  # maximum degree L of the spherical harmonics
    hidden: int = 128
    layers: int = 4  # number of hidden layers; 0 = linear model on the encoding
    activation: str = "gelu"  # "gelu" | "relu" | "siren"
    siren_w0: float = 30.0
    use_context: bool = False  # coarse elevation field as extra input
    context_ny: int = 90  # resolution of that context field

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, FieldConfig] = {
    "Fourier MLP (recommended)": FieldConfig(
        input_mode="xyz", encoding="fourier", n_features=128, sigma=6.0,
        hidden=128, layers=4, activation="gelu",
    ),
    "Fourier MLP + context field": FieldConfig(
        input_mode="xyz", encoding="fourier", n_features=128, sigma=6.0,
        hidden=128, layers=4, activation="gelu", use_context=True, context_ny=90,
    ),
    "SIREN (sine activation)": FieldConfig(
        input_mode="xyz", encoding="none", hidden=160, layers=4, activation="siren",
    ),
    "Spherical harmonics (L = 16)": FieldConfig(
        input_mode="xyz", encoding="spherical", sh_degree=16,
        hidden=128, layers=4, activation="relu",
    ),
    "Spherical harmonic expansion (L = 40, linear)": FieldConfig(
        input_mode="xyz", encoding="spherical", sh_degree=40, layers=0,
    ),
    "Plain MLP (fails on purpose)": FieldConfig(
        input_mode="xyz", encoding="none", hidden=128, layers=4, activation="relu",
    ),
    "Fourier MLP, very small": FieldConfig(
        input_mode="xyz", encoding="fourier", n_features=64, sigma=3.0,
        hidden=48, layers=3, activation="gelu",
    ),
    "Fourier MLP, large": FieldConfig(
        input_mode="xyz", encoding="fourier", n_features=256, sigma=6.0,
        hidden=256, layers=5, activation="gelu",
    ),
}


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def _siren_init(linear: nn.Linear, w0: float, first: bool) -> None:
    """Initialisation from Sitzmann et al. 2020 -- without it a SIREN learns nothing."""
    fan_in = linear.in_features
    bound = 1.0 / fan_in if first else math.sqrt(6.0 / fan_in) / w0
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound)


class CoarseContext(nn.Module):
    """A very coarse elevation grid, sampled bilinearly -- the 'context'.

    It is stored as int16, i.e. exactly as it ends up in the file.
    The network then only has to learn the *residual* to the context, which
    makes the task considerably easier (the idea behind context-aware INRs).
    """

    def __init__(self, coarse_m: np.ndarray):
        super().__init__()
        quantised = np.clip(np.rint(coarse_m), -32768, 32767).astype(np.int16)
        self.register_buffer("coarse", torch.from_numpy(quantised))
        self.ny, self.nx = quantised.shape

    @property
    def stored_bytes(self) -> int:
        return self.ny * self.nx * 2

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """(N, 2) degrees → (N,) context elevation in metres, bilinearly interpolated."""
        table = self.coarse.to(lonlat.dtype)

        fx = (lonlat[..., 0] + 180.0) / 360.0 * self.nx - 0.5
        j0 = torch.floor(fx)
        wx = fx - j0
        j0 = j0.long() % self.nx  # longitude wraps around: no edge, no seam
        j1 = (j0 + 1) % self.nx

        fy = ((lonlat[..., 1] + 90.0) / 180.0 * self.ny - 0.5).clamp(0, self.ny - 1)
        i0 = torch.floor(fy).long().clamp(0, self.ny - 1)
        i1 = (i0 + 1).clamp(max=self.ny - 1)
        wy = fy - i0.to(fy.dtype)

        top = table[i0, j0] * (1 - wx) + table[i0, j1] * wx
        bottom = table[i1, j0] * (1 - wx) + table[i1, j1] * wx
        return top * (1 - wy) + bottom * wy


class NeuralField(nn.Module):
    def __init__(
        self,
        cfg: FieldConfig,
        z_mean: float = 0.0,
        z_std: float = 1.0,
        coarse_m: np.ndarray | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.transform = InputTransform(cfg.input_mode)

        if cfg.encoding == "fourier":
            self.encoding: nn.Module = FourierFeatures(
                self.transform.out_dim, cfg.n_features, cfg.sigma, cfg.fourier_seed
            )
        elif cfg.encoding == "spherical":
            if cfg.input_mode != "xyz":
                raise ValueError("encoding='spherical' requires input_mode='xyz'")
            self.encoding = SphericalHarmonics(self.transform.out_dim, cfg.sh_degree)
        else:
            self.encoding = Identity(self.transform.out_dim)

        if cfg.use_context:
            if coarse_m is None:
                raise ValueError("use_context=True requires a coarse grid (coarse_m)")
            self.context: CoarseContext | None = CoarseContext(coarse_m)
        else:
            self.context = None

        in_dim = self.encoding.out_dim + (1 if self.context is not None else 0)
        dims = [in_dim] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for i in range(cfg.layers):
            linear = nn.Linear(dims[i], dims[i + 1])
            if cfg.activation == "siren":
                _siren_init(linear, cfg.siren_w0, first=(i == 0))
                act: nn.Module = Sine(cfg.siren_w0 if i == 0 else 1.0)
            elif cfg.activation == "relu":
                act = nn.ReLU()
            else:
                act = nn.GELU()
            blocks += [linear, act]

        self.body = nn.Sequential(*blocks)
        # Without hidden layers the head sits directly on the encoding: a plain
        # weighted sum of basis functions (with spherical harmonics, exactly the
        # classic spherical harmonic expansion of geodesy).
        self.head = nn.Linear(dims[-1], 1)
        if cfg.activation == "siren":
            _siren_init(self.head, cfg.siren_w0, first=False)

        # Part of the "file": without these two numbers the weights are worthless.
        self.register_buffer("z_mean", torch.tensor(float(z_mean)))
        self.register_buffer("z_std", torch.tensor(float(max(z_std, 1e-6))))

    def forward_normalised(self, lonlat: torch.Tensor) -> torch.Tensor:
        h = self.encoding(self.transform(lonlat))
        if self.context is None:
            return self.head(self.body(h)).squeeze(-1)

        context = (self.context(lonlat) - self.z_mean) / self.z_std
        h = torch.cat([h, context.unsqueeze(-1)], dim=-1)
        return context + self.head(self.body(h)).squeeze(-1)  # network learns the residual

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """(N, 2) degrees → (N,) elevation in metres."""
        return self.forward_normalised(lonlat) * self.z_std + self.z_mean

    # --- Bookkeeping for the compression ratio ------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stored_bytes(self, bytes_per_param: int = 4) -> int:
        """Payload of a stored file -- everything needed for the reconstruction."""
        total = self.n_parameters() * bytes_per_param
        total += 2 * 4  # z_mean, z_std as float32
        if isinstance(self.encoding, FourierFeatures):
            total += 4  # the seed is enough, B is reconstructed from it
        if self.context is not None:
            total += self.context.stored_bytes  # int16, independent of bytes_per_param
        return total

    @torch.no_grad()
    def predict_grid(self, lon_deg, lat_deg, chunk_rows: int | None = None, device=None):
        """Evaluates the field on a lon/lat grid → (ny, nx) in metres (numpy).

        Row by row, so that grids with many millions of points fit as well.
        """
        device = device or next(self.parameters()).device
        lon = np.ascontiguousarray(lon_deg, dtype=np.float32)
        lat = np.ascontiguousarray(lat_deg, dtype=np.float32)
        ny, nx = len(lat), len(lon)
        # ~100 M floats per encoded chunk, however wide the encoding is
        points = max(1, 100_000_000 // max(self.encoding.out_dim, 256))
        chunk_rows = chunk_rows or max(1, points // max(nx, 1))

        lon_t = torch.from_numpy(lon).to(device)
        was_training = self.training
        self.eval()
        out = np.empty((ny, nx), dtype=np.float32)
        for start in range(0, ny, chunk_rows):
            stop = min(start + chunk_rows, ny)
            lat_t = torch.from_numpy(lat[start:stop]).to(device)
            rows = stop - start
            lonlat = torch.stack(
                [lon_t.expand(rows, nx), lat_t[:, None].expand(rows, nx)], dim=-1
            ).reshape(-1, 2)
            out[start:stop] = self(lonlat).to("cpu", torch.float32).numpy().reshape(rows, nx)
        if was_training:
            self.train()
        return out


def build_field(
    cfg: FieldConfig, z_mean: float, z_std: float, coarse_m: np.ndarray | None = None
) -> NeuralField:
    return NeuralField(cfg, z_mean, z_std, coarse_m)
