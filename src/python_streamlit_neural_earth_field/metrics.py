"""Error metrics and compression bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scores:
    rmse: float
    mae: float
    weighted_rmse: float  # weighted by cos(lat), as in WeatherBench
    weighted_mae: float
    max_abs: float
    psnr_db: float
    r2: float

    def to_dict(self, unit: str = "m") -> dict[str, float]:
        return {
            f"RMSE [{unit}]": self.rmse,
            f"MAE [{unit}]": self.mae,
            f"weighted RMSE [{unit}]": self.weighted_rmse,
            f"weighted MAE [{unit}]": self.weighted_mae,
            f"max. error [{unit}]": self.max_abs,
            "PSNR [dB]": self.psnr_db,
            "R²": self.r2,
        }


def evaluate(truth: np.ndarray, pred: np.ndarray, cell_weights: np.ndarray | None = None) -> Scores:
    err = (pred - truth).astype(np.float64)
    se = err**2
    ae = np.abs(err)

    if cell_weights is None:
        w = np.ones_like(se)
    else:
        w = cell_weights.astype(np.float64)
        w = w / w.mean()

    value_range = float(truth.max() - truth.min())
    rmse = float(np.sqrt(se.mean()))
    var = float(truth.astype(np.float64).var())

    return Scores(
        rmse=rmse,
        mae=float(ae.mean()),
        weighted_rmse=float(np.sqrt((se * w).mean())),
        weighted_mae=float((ae * w).mean()),
        max_abs=float(ae.max()),
        psnr_db=float(20 * np.log10(value_range / rmse)) if rmse > 0 else float("inf"),
        r2=float(1.0 - se.mean() / var) if var > 0 else float("nan"),
    )


@dataclass
class Compression:
    raw_bytes: int
    model_bytes: int
    n_points: int
    n_parameters: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.model_bytes

    @property
    def bits_per_point(self) -> float:
        return 8 * self.model_bytes / self.n_points

    @property
    def points_per_parameter(self) -> float:
        return self.n_points / self.n_parameters


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GB"
