"""An aerial photo as many small Gaussians: 2D Gaussian splatting.

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
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

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
