"""How coordinates enter the network.

This is the most important decision in the whole project.  An MLP that receives
raw (lon, lat) cannot represent high-frequency detail at all -- it has a strong
"spectral bias".  Two tricks against it:

1. **Spherical coordinates → xyz.**  (lon, lat) has a seam at ±180° and two
   degenerate poles.  The point on the unit sphere has neither.
2. **Fourier features.**  Instead of feeding x directly, you feed
   [sin(2π B x), cos(2π B x)] with a fixed random matrix B.  This boosts high
   frequencies so that the MLP can learn them at all.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def lonlat_to_xyz(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    """(lon, lat) in degrees → point on the unit sphere. Shape (..., 3)."""
    lon = torch.deg2rad(lon_deg)
    lat = torch.deg2rad(lat_deg)
    cos_lat = torch.cos(lat)
    return torch.stack([cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)], dim=-1)


class InputTransform(nn.Module):
    """lonlat → normalised to [-1, 1], or → unit-sphere xyz."""

    def __init__(self, mode: str = "xyz"):
        super().__init__()
        if mode not in {"xyz", "lonlat"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.out_dim = 3 if mode == "xyz" else 2

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        if self.mode == "xyz":
            return lonlat_to_xyz(lonlat[..., 0], lonlat[..., 1])
        return torch.stack([lonlat[..., 0] / 180.0, lonlat[..., 1] / 90.0], dim=-1)


class FourierFeatures(nn.Module):
    """[sin(2π B x), cos(2π B x)] with a fixed, random B ~ N(0, sigma²).

    B is *not* stored (``persistent=False``): it can be reconstructed from the
    seed.  So the compression ratio only counts 4 bytes of seed instead of a
    whole matrix -- a detail that really matters in practice.
    """

    def __init__(self, in_dim: int, n_features: int = 128, sigma: float = 4.0, seed: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        self.sigma = sigma
        self.seed = seed
        self.out_dim = 2 * n_features
        self.register_buffer("B", self._make_B(), persistent=False)

    def _make_B(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.seed)
        return torch.randn(self.in_dim, self.n_features, generator=gen) * self.sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Identity(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
