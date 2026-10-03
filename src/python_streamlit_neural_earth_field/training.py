"""Deliberately overfitting the network.

Unlike in ordinary machine learning there is no validation split and no
regularisation here: the goal is to reproduce *exactly this one* grid as
accurately as possible.  Overfitting is the compression.

So that grids with many millions of points fit as well, the coordinates are
not precomputed but built per batch from the row/column indices.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .data import ElevationGrid
from .metrics import Scores, evaluate
from .model import NeuralField


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 32768
    lr: float = 3e-3
    lr_final_factor: float = 0.02  # cosine decay down to lr * factor
    warmup: int = 100
    area_weighted_sampling: bool = True
    eval_every: int = 100
    eval_points: int = 200_000  # sample for intermediate evaluations
    seed: int = 0


@dataclass
class Progress:
    step: int
    total: int
    loss: float  # MSE on normalised elevations
    lr: float
    elapsed_s: float
    scores: Scores  # on the evaluation sample


class Sampler:
    """Draws grid cells -- uniformly or area-weighted (∝ cos lat)."""

    def __init__(self, grid: ElevationGrid, device: torch.device, area_weighted: bool, seed: int):
        self.ny, self.nx = grid.ny, grid.nx
        self.device = device
        self.area_weighted = area_weighted
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.from_numpy(grid.lat_weights().astype(np.float64))
        self.row_cdf = torch.cumsum(w / w.sum(), dim=0).to(torch.float32)

    def __call__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.area_weighted:
            u = torch.rand(n, generator=self.generator)
            rows = torch.searchsorted(self.row_cdf, u).clamp_(max=self.ny - 1)
        else:
            rows = torch.randint(0, self.ny, (n,), generator=self.generator)
        cols = torch.randint(0, self.nx, (n,), generator=self.generator)
        return rows.to(self.device), cols.to(self.device)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return cfg.lr * (cfg.lr_final_factor + (1 - cfg.lr_final_factor) * cos)


def train(
    field: NeuralField,
    grid: ElevationGrid,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> Iterator[Progress]:
    """Generator: trains and reports progress every ``eval_every`` steps."""
    device = device or pick_device()
    field.to(device).train()
    torch.manual_seed(cfg.seed)

    lon_t = torch.from_numpy(np.ascontiguousarray(grid.lon, dtype=np.float32)).to(device)
    lat_t = torch.from_numpy(np.ascontiguousarray(grid.lat, dtype=np.float32)).to(device)
    z_norm = ((torch.from_numpy(grid.z).to(device) - field.z_mean) / field.z_std).contiguous()

    def lonlat_of(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return torch.stack([lon_t[cols], lat_t[rows]], dim=-1)

    sampler = Sampler(grid, device, cfg.area_weighted_sampling, cfg.seed)
    optimiser = torch.optim.Adam(field.parameters(), lr=cfg.lr)

    # fixed evaluation sample so that intermediate results are comparable
    eval_n = min(cfg.eval_points, grid.n_points)
    eval_sampler = Sampler(grid, device, False, cfg.seed + 1)
    eval_rows, eval_cols = eval_sampler(eval_n)
    eval_lonlat = lonlat_of(eval_rows, eval_cols)
    eval_truth = torch.from_numpy(grid.z).to(device)[eval_rows, eval_cols].cpu().numpy()
    eval_w = np.cos(np.deg2rad(grid.lat[eval_rows.cpu().numpy()])).clip(1e-6).astype(np.float64)

    started = time.perf_counter()

    for step in range(cfg.steps):
        lr = _lr_at(step, cfg)
        for group in optimiser.param_groups:
            group["lr"] = lr

        rows, cols = sampler(cfg.batch_size)
        pred = field.forward_normalised(lonlat_of(rows, cols))
        loss = torch.mean((pred - z_norm[rows, cols]) ** 2)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            field.eval()
            with torch.no_grad():  # in chunks: wide encodings get large at 200 k points
                eval_pred = torch.cat(
                    [field(part) for part in eval_lonlat.split(cfg.batch_size)]
                ).cpu().numpy()
            field.train()
            yield Progress(
                step=step + 1,
                total=cfg.steps,
                loss=float(loss.detach().to("cpu")),
                lr=lr,
                elapsed_s=time.perf_counter() - started,
                scores=evaluate(eval_truth, eval_pred, eval_w),
            )

    field.eval()
