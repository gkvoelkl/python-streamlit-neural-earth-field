"""Climate: the GISTEMP grid, the field f(x, y, z, t), the scores and the straight-line baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.climate import (
    CLIMATE_PRESETS,
    ClimateConfig,
    ClimateGrid,
    ClimateScores,
    ClimateTrainConfig,
    build_climate_field,
    line_per_cell,
    train_climate,
)


def _grid(years: int = 4) -> ClimateGrid:
    """A coarse 30° grid: a warming of 0.5 °C per year towards the north pole, plus a seasonal wave."""
    lat = np.arange(75, -90, -30, dtype=np.float32)
    lon = np.arange(-165, 180, 30, dtype=np.float32)
    months = np.arange(12 * years)
    north = (lat[None, :, None] + 90) / 180
    anomaly = (0.5 * months[:, None, None] / 12 * north + 0.3 * np.sin(2 * np.pi * months / 12)[:, None, None]
               + 0 * lon[None, None, :]).astype(np.float32)
    anomaly[:, -1, :3] = np.nan  # a few cells without data
    return ClimateGrid(anomaly, lat, lon, 2000)


def test_grid_bookkeeping():
    grid = _grid()
    assert grid.n_months == 48
    assert grid.raw_bytes == 48 * 6 * 12 * 2
    assert grid.year_of(0) == pytest.approx(2000 + 1 / 24)
    years, means = grid.annual()
    assert list(years) == [2000, 2001, 2002, 2003]
    assert np.isnan(means[:, -1, 0]).all()
    # the trend in °C per decade: 0.5 °C per year at the top row, scaled by how far north it is
    assert grid.trend(first=2000, last=2003)[0, 0] == pytest.approx(5 * (75 + 90) / 180, rel=1e-4)


def test_line_per_cell_recovers_a_straight_line():
    grid = _grid()
    grid.anomaly[:] = np.arange(48, dtype=np.float32)[:, None, None] * 0.1 + 2.0
    line = line_per_cell(grid)
    assert np.allclose(line, grid.anomaly, atol=1e-4)


def test_field_size_and_presets():
    grid = _grid()
    field = build_climate_field(ClimateConfig(n_space=8, n_time=4, hidden=16, layers=2), grid)
    assert field.stored_bytes() == field.n_parameters() * 2 + 12
    tau, season = field.date(torch.tensor([0, 11, 12, 47]))
    assert tau[0, 0] == pytest.approx(-1 + 1 / 48) and tau[-1, 0] == pytest.approx(1 - 1 / 48)
    assert torch.allclose(season[0], season[2])  # January twice: the same point on the circle
    sizes = [build_climate_field(cfg, grid).n_parameters() for cfg in CLIMATE_PRESETS.values()]
    assert sizes == sorted(sizes)


def test_training_beats_the_mean_and_scores_are_consistent():
    grid = _grid()
    field = build_climate_field(ClimateConfig(n_space=16, sigma_space=1.0, n_time=8, sigma_time=2.0, hidden=32, layers=2), grid)
    progress = list(train_climate(field, grid, ClimateTrainConfig(steps=300, batch_size=1024, eval_every=100),
                                  torch.device("cpu")))
    assert progress[-1].rmse < progress[0].rmse
    pred = field.render(grid)
    assert pred.shape == grid.anomaly.shape
    s = ClimateScores.compare(grid, pred)
    zero = ClimateScores.compare(grid, np.zeros_like(grid.anomaly))
    assert s.rmse_month < zero.rmse_month
    assert s.mean_trend == pytest.approx(zero.mean_trend)  # the trend of the data does not depend on the prediction
