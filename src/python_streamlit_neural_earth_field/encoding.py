"""How coordinates enter the network.

This is the most important decision in the whole project.  An MLP that receives
raw (lon, lat) cannot represent high-frequency detail at all -- it has a strong
"spectral bias".  Two tricks against it:

1. **Spherical coordinates → xyz.**  (lon, lat) has a seam at ±180° and two
   degenerate poles.  The point on the unit sphere has neither.
2. **Fourier features.**  Instead of feeding x directly, you feed
   [sin(2π B x), cos(2π B x)] with a fixed random matrix B.  This boosts high
   frequencies so that the MLP can learn them at all.

An alternative to 2. is the Fourier basis that actually belongs to the sphere:
**spherical harmonics** Y_lm up to degree L.  Complete and orthogonal instead of
random -- but (L+1)² features for degree L, so fine detail gets expensive.
"""

from __future__ import annotations

import math

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


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics Y_lm(x, y, z) for all degrees l <= L. (L+1)² features.

    Normalised so that every feature has mean square 1 over the sphere (Y_00 = 1),
    which keeps the network input well scaled.  Like the Fourier matrix B, nothing
    here needs to be stored: the degree L in the config is enough.
    """

    def __init__(self, in_dim: int, degree: int = 16):
        super().__init__()
        if in_dim != 3:
            raise ValueError("spherical harmonics need xyz input on the unit sphere")
        self.degree = degree
        self.out_dim = (degree + 1) ** 2

        # Coefficients of the three-term recurrence in l for the normalised
        # associated Legendre functions; zero where (l, m) is not covered by it.
        L = degree
        a = torch.zeros(L + 1, L + 1)
        b = torch.zeros(L + 1, L + 1)
        for l in range(2, L + 1):
            for m in range(l - 1):
                a[l, m] = math.sqrt((4 * l * l - 1) / (l * l - m * m))
                b[l, m] = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
        diag = torch.tensor([1.0] + [math.sqrt((2 * m + 1) / (2 * m)) for m in range(1, L + 1)])
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self.register_buffer("diag", diag, persistent=False)
        self.register_buffer("m", torch.arange(L + 1, dtype=torch.float32), persistent=False)

        ls, ms = torch.tril_indices(L + 1, L + 1)  # all (l, m) with m <= l
        self.register_buffer("l_zonal", torch.arange(L + 1), persistent=False)
        self.register_buffer("l_sect", ls[ms > 0], persistent=False)
        self.register_buffer("m_sect", ms[ms > 0], persistent=False)

    def legendre(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """P̄_l^m(z) for z = sin(lat), c = cos(lat). Shape (N, L+1, L+1) as [l, m]."""
        L = self.degree
        n = z.shape[0]
        # P̄_m^m = diag_m · c · P̄_{m-1}^{m-1}: the sectoral values all at once
        sectoral = torch.cumprod(self.diag.to(z.dtype) * torch.cat(
            [torch.ones_like(c)[:, None], c[:, None].expand(n, L)], dim=1), dim=1)

        rows = []
        prev2 = torch.zeros(n, L + 1, dtype=z.dtype, device=z.device)
        prev = torch.zeros_like(prev2)
        for l in range(L + 1):
            row = self.a[l] * (z[:, None] * prev - self.b[l] * prev2)
            row = row.clone()
            if l >= 1:
                row[:, l - 1] = math.sqrt(2 * l + 1) * z * prev[:, l - 1]
            row[:, l] = sectoral[:, l]
            rows.append(row)
            prev2, prev = prev, row
        return torch.stack(rows, dim=1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        x, y, z = xyz.unbind(-1)
        c = torch.sqrt(x * x + y * y)
        lon = torch.atan2(y, x)

        P = self.legendre(z, c)
        angle = lon[:, None] * self.m
        P_sect = P[:, self.l_sect, self.m_sect] * math.sqrt(2.0)
        features = torch.cat(
            [
                P[:, self.l_zonal, 0],
                P_sect * torch.cos(angle[:, self.m_sect]),
                P_sect * torch.sin(angle[:, self.m_sect]),
            ],
            dim=-1,
        )
        return features.reshape(*shape, self.out_dim)


class Identity(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
