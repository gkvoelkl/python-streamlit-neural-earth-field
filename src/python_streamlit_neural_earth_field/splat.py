"""Gaussian splatting: an aerial photo (2D) and buildings (3D) as many small Gaussians.

The neural field of ``ortho.py`` stores the photo in the weights of a network
that every pixel has to pass through.  Gaussian splatting stores it as a list
of coloured blobs instead: each Gaussian has a centre, two widths, a rotation
and a colour -- eight numbers -- and adds its colour to the pixels around it,
fading with the distance.  The image is the sum of all blobs (as in
GaussianImage, Zhang et al. 2024: no sorting, no transparency).

Training is the same as for the network: the blobs start on a regular grid and
gradient descent moves, stretches, turns and recolours them until the sum
looks like the photo.  Because a Gaussian is small, it only touches a window of
pixels around its centre; that is what makes splatting fast to draw.

In 3D the same idea fills the LoD2 buildings of ``buildings.py``: the Gaussians
add up 'how much building' there is at every voxel, and a voxel counts as inside
where the sum exceeds one half.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .buildings import BuildingGrid, BuildingProgress
from .ortho import Orthophoto, OrthoProgress, psnr

NUMBERS_PER_GAUSSIAN = 8  # centre (2), widths (2), rotation (1), colour (3)


@dataclass
class SplatConfig:
    grid: int = 203  # Gaussians per tile edge at the start; grid² Gaussians in total
    sigma_max: float = 0.6  # largest width, in units of the starting spacing


# the same file sizes as the network presets in ortho.py (float16)
SPLAT_PRESETS: dict[str, SplatConfig] = {
    "Small": SplatConfig(grid=101),  # 10,201 Gaussians ≈ 159 KB
    "Medium": SplatConfig(grid=203),  # 41,209 Gaussians ≈ 644 KB
    "Large": SplatConfig(grid=405),  # 164,025 Gaussians ≈ 2.5 MB
}


class GaussianImage(nn.Module):
    """A square tile as a sum of 2D Gaussians.

    Centres and widths are kept in tile units (−1 … 1, north up), so the same
    Gaussians can be drawn at any pixel size.  The widths go through a sigmoid
    and can never exceed ``sigma_max`` starting spacings: a Gaussian then only
    reaches a small window of pixels, and every pixel only a few Gaussians.
    """

    def __init__(self, cfg: SplatConfig, east0: float, north0: float, size_m: float):
        super().__init__()
        self.cfg = cfg
        self.origin = (float(east0), float(north0))  # part of the 'file', like for the network
        self.size_m = size_m
        g = cfg.grid
        spacing = 2 / g
        centres = (torch.arange(g, dtype=torch.float32) + 0.5) * spacing - 1
        north, east = torch.meshgrid(-centres, centres, indexing="ij")
        self.centre = nn.Parameter(torch.stack([east.flatten(), north.flatten()], -1))
        self.sigma_max = cfg.sigma_max * spacing
        start = min(0.95, 0.5 / cfg.sigma_max)  # half a spacing wide: neighbours overlap a little
        self.raw_width = nn.Parameter(torch.full((g * g, 2), math.log(start / (1 - start))))
        self.angle = nn.Parameter(torch.zeros(g * g))
        self.colour = nn.Parameter(torch.zeros(g * g, 3))

    @property
    def n_gaussians(self) -> int:
        return self.centre.shape[0]

    def n_parameters(self) -> int:
        return self.n_gaussians * NUMBERS_PER_GAUSSIAN

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Eight numbers per Gaussian (float16 by default) + corner of the tile (2 × float64)."""
        return self.n_parameters() * bytes_per_param + 16

    def width(self) -> torch.Tensor:
        return self.sigma_max * torch.sigmoid(self.raw_width)

    @torch.no_grad()
    def init_colours(self, rgb: torch.Tensor) -> None:
        """Start every Gaussian with the colour under its centre, shared with its overlapping neighbours."""
        n = rgb.shape[0]
        col = ((self.centre[:, 0] + 1) / 2 * n).long().clamp(0, n - 1)
        row = ((1 - self.centre[:, 1]) / 2 * n).long().clamp(0, n - 1)
        overlap = 2 * math.pi * (self.width()[:, 0] / (2 / self.cfg.grid)) ** 2  # sum of the neighbours' weights
        self.colour.copy_(rgb[row, col] / overlap[:, None])

    def splat(self, n: int) -> torch.Tensor:
        """The image at ``n`` × ``n`` pixels, colours in 0 … 1 (not clipped)."""
        device = self.centre.device
        half = int(math.ceil(3 * self.sigma_max / 2 * n))  # 3 widths, in pixels
        offsets = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx, dy = dx.flatten(), dy.flatten()
        # centres in pixels: x to the east, y to the south
        cx = (self.centre[:, 0] + 1) / 2 * n
        cy = (1 - self.centre[:, 1]) / 2 * n
        px = cx.detach().floor()[:, None] + dx[None]
        py = cy.detach().floor()[:, None] + dy[None]
        inside = (px >= 0) & (px < n) & (py >= 0) & (py < n)
        ex = px + 0.5 - cx[:, None]
        ey = py + 0.5 - cy[:, None]
        w = self.width() / 2 * n
        cos, sin = torch.cos(self.angle)[:, None], torch.sin(self.angle)[:, None]
        u = (cos * ex + sin * ey) / w[:, 0:1]
        v = (-sin * ex + cos * ey) / w[:, 1:2]
        weight = torch.exp(-0.5 * (u * u + v * v)) * inside
        index = (py.clamp(0, n - 1) * n + px.clamp(0, n - 1)).long().flatten()
        image = torch.zeros(n * n, 3, device=device)
        image = image.index_add(0, index, (weight[..., None] * self.colour[:, None, :]).reshape(-1, 3))
        return image.reshape(n, n, 3)

    @torch.no_grad()
    def render(self, photo: Orthophoto) -> np.ndarray:
        """The image as the Gaussians draw it, at the size of the photo."""
        out = self.splat(photo.n).clamp(0, 1) * 255
        return np.rint(out.cpu().numpy()).astype(np.uint8)


def build_gaussian_image(cfg: SplatConfig, photo: Orthophoto) -> GaussianImage:
    e0 = photo.east[0] - photo.metres_per_pixel / 2
    n0 = photo.north[-1] - photo.metres_per_pixel / 2
    return GaussianImage(cfg, e0, n0, photo.n * photo.metres_per_pixel)


@dataclass
class SplatTrainConfig:
    steps: int = 1000
    lr_centre: float = 8e-5  # tile units (−1 … 1); about 0.05 pixel at 1250 pixels
    lr_shape: float = 5e-3  # widths and rotation
    lr_colour: float = 5e-3
    lr_final_factor: float = 0.02
    eval_every: int = 25


def train_splat(model: GaussianImage, photo: Orthophoto, cfg: SplatTrainConfig,
                device: torch.device) -> Iterator[OrthoProgress]:
    """Fits the Gaussians to the whole image in every step."""
    model.to(device).train()
    target = torch.from_numpy(photo.rgb).to(device).float() / 255
    model.init_colours(target)
    groups = [([model.centre], cfg.lr_centre), ([model.raw_width, model.angle], cfg.lr_shape),
              ([model.colour], cfg.lr_colour)]
    optimiser = torch.optim.Adam([{"params": p, "lr": lr} for p, lr in groups])
    started = time.perf_counter()
    for step in range(cfg.steps):
        t = step / max(cfg.steps - 1, 1)
        factor = cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t))
        for group, (_, lr) in zip(optimiser.param_groups, groups):
            group["lr"] = lr * factor
        loss = torch.mean((model.splat(photo.n) - target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            yield OrthoProgress(step + 1, cfg.steps, psnr(loss.item(), peak=1.0), time.perf_counter() - started)
    model.eval()


# --------------------------------------------------------------------------- 3D: buildings


@dataclass
class VolumeConfig:
    n_gaussians: int = 41_120  # as many numbers as the Medium network of buildings.py has weights
    sigma_max: float = 0.8  # largest width, in units of the mean spacing of the Gaussians inside the buildings
    seed: int = 0


# the same file sizes as the network presets in buildings.py (float16): weights × 2 / 16 bytes
VOLUME_PRESETS: dict[str, VolumeConfig] = {
    "Small": VolumeConfig(n_gaussians=10_320),
    "Medium": VolumeConfig(n_gaussians=41_120),
    "Large": VolumeConfig(n_gaussians=164_224),
}


class GaussianVolume(nn.Module):
    """Buildings as a sum of 3D Gaussians: 'how much building' at every point, inside where the sum exceeds ½.

    Eight numbers per Gaussian again: centre (3), three widths (3), a rotation about the vertical (1) and a
    strength (1).  LoD2 walls are vertical, so turning about the vertical is enough.  Centres are in metres from
    the lower south-west corner of the grid (east, north, up).  The Gaussians start at random voxels inside the
    buildings – like 3D Gaussian splatting starts from a point cloud – and a Gaussian only reaches 2.5 widths.
    """

    def __init__(self, cfg: VolumeConfig, grid: BuildingGrid):
        super().__init__()
        self.cfg = cfg
        self.origin = grid.origin  # part of the 'file', like the corner of the orthophoto
        rng = np.random.default_rng(cfg.seed)
        nz, ny, nx = grid.shape
        res = grid.res
        k, i, j = np.nonzero(grid.inside)
        pick = rng.choice(len(k), cfg.n_gaussians, replace=len(k) < cfg.n_gaussians)
        start = np.stack([j[pick] + 0.5, ny - i[pick] - 0.5, k[pick] + 0.5], -1) * res
        start += rng.uniform(-0.5, 0.5, start.shape) * res
        self.centre = nn.Parameter(torch.from_numpy(start.astype(np.float32)))
        spacing = (len(k) * res**3 / cfg.n_gaussians) ** (1 / 3)  # metres of building per Gaussian, as an edge
        self.sigma_max = cfg.sigma_max * spacing  # stored with the Gaussians (one more number)
        width = min(0.95, 0.6 * spacing / self.sigma_max)
        self.raw_width = nn.Parameter(torch.full((cfg.n_gaussians, 3), math.log(width / (1 - width))))
        self.yaw = nn.Parameter(torch.zeros(cfg.n_gaussians))
        self.strength = nn.Parameter(torch.full((cfg.n_gaussians,), 0.5))

    @property
    def n_gaussians(self) -> int:
        return self.centre.shape[0]

    def n_parameters(self) -> int:
        return self.n_gaussians * NUMBERS_PER_GAUSSIAN

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Eight numbers per Gaussian + corner of the grid (3 × float64) + the largest width (float32)."""
        return self.n_parameters() * bytes_per_param + 24 + 4

    def width(self) -> torch.Tensor:
        return self.sigma_max * torch.sigmoid(self.raw_width)

    def _half(self, res: float) -> int:
        return int(math.ceil(2.5 * self.sigma_max / res))

    def _window(self, part: slice, shape: tuple[int, int, int], res: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat voxel indices and weights for the Gaussians in ``part``, each over its own window."""
        nz, ny, nx = shape
        half = self._half(res)
        o = torch.arange(-half, half + 1, device=self.centre.device, dtype=torch.float32)
        ok_, oi, oj = (t.flatten() for t in torch.meshgrid(o, o, o, indexing="ij"))
        c = self.centre[part]
        # nearest voxel of each centre: column j to the east, row i to the south, layer k upwards
        j = torch.floor(c[:, 0].detach() / res)[:, None] + oj
        i = torch.floor(ny - c[:, 1].detach() / res)[:, None] + oi
        k = torch.floor(c[:, 2].detach() / res)[:, None] + ok_
        inside = (j >= 0) & (j < nx) & (i >= 0) & (i < ny) & (k >= 0) & (k < nz)
        dx = (j + 0.5) * res - c[:, 0:1]
        dy = (ny - i - 0.5) * res - c[:, 1:2]
        dz = (k + 0.5) * res - c[:, 2:3]
        w = self.width()[part]
        cos, sin = torch.cos(self.yaw[part])[:, None], torch.sin(self.yaw[part])[:, None]
        u = (cos * dx + sin * dy) / w[:, 0:1]
        v = (-sin * dx + cos * dy) / w[:, 1:2]
        h = dz / w[:, 2:3]
        weight = torch.exp(-0.5 * (u * u + v * v + h * h)) * inside * self.strength[part, None]
        flat = (k.clamp(0, nz - 1) * ny + i.clamp(0, ny - 1)) * nx + j.clamp(0, nx - 1)
        return flat.long(), weight

    def _parts(self, res: float, budget: int = 8_000_000) -> list[slice]:
        """Groups of Gaussians whose windows together have about ``budget`` voxels."""
        per = max(1, budget // (2 * self._half(res) + 1) ** 3)
        return [slice(a, a + per) for a in range(0, self.n_gaussians, per)]

    @torch.no_grad()
    def occupancy(self, shape: tuple[int, int, int], res: float) -> torch.Tensor:
        """The sum of all Gaussians on a voxel grid of ``shape`` with voxels of ``res`` metres."""
        out = torch.zeros(shape[0] * shape[1] * shape[2], device=self.centre.device)
        for part in self._parts(res):
            flat, weight = self._window(part, shape, res)
            out.index_add_(0, flat.flatten(), weight.flatten())
        return out.reshape(shape)

    def render(self, grid: BuildingGrid) -> np.ndarray:
        """½ − occupancy on the voxels of ``grid``: negative inside, so the scores of buildings.py apply."""
        return (0.5 - self.occupancy(grid.shape, grid.res)).cpu().numpy()


def build_gaussian_volume(cfg: VolumeConfig, grid: BuildingGrid) -> GaussianVolume:
    return GaussianVolume(cfg, grid)


@dataclass
class VolumeTrainConfig:
    steps: int = 300
    lr_centre: float = 0.03  # metres
    lr_shape: float = 5e-3
    lr_strength: float = 5e-3
    lr_final_factor: float = 0.02
    eval_every: int = 10


def train_gaussian_volume(model: GaussianVolume, grid: BuildingGrid, cfg: VolumeTrainConfig,
                          device: torch.device) -> Iterator[BuildingProgress]:
    """Fits the sum of the Gaussians to inside (1) and outside (0) on every voxel.

    The whole volume does not fit into memory with gradients, so each step first adds up all Gaussians
    without gradients, then lets one group of Gaussians after the other add its share of the gradient of
    the squared error – the same gradient, computed in pieces.
    """
    model.to(device).train()
    target = torch.from_numpy(grid.inside).to(device).float()
    groups = [([model.centre], cfg.lr_centre), ([model.raw_width, model.yaw], cfg.lr_shape),
              ([model.strength], cfg.lr_strength)]
    optimiser = torch.optim.Adam([{"params": p, "lr": lr} for p, lr in groups])
    parts = model._parts(grid.res)
    started = time.perf_counter()
    for step in range(cfg.steps):
        t = step / max(cfg.steps - 1, 1)
        factor = cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t))
        for group, (_, lr) in zip(optimiser.param_groups, groups):
            group["lr"] = lr * factor
        occupancy = model.occupancy(grid.shape, grid.res)
        residual = (2 * (occupancy - target) / target.numel()).flatten()
        optimiser.zero_grad(set_to_none=True)
        for part in parts:
            flat, weight = model._window(part, grid.shape, grid.res)
            (residual[flat] * weight).sum().backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            correct = torch.mean(((occupancy > 0.5) == (target > 0.5)).float()).item()
            yield BuildingProgress(step + 1, cfg.steps, correct, time.perf_counter() - started)
    model.eval()
