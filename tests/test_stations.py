"""Stations: projection, split, IDW, kriging (also with height) and the small network."""

from __future__ import annotations

import numpy as np
import pytest

from python_streamlit_neural_earth_field.stations import (
    HeightTrend,
    StationNetConfig,
    fit_variogram,
    idw,
    kriging,
    kriging_with_height,
    network,
    split,
    to_km,
)


def _field(n: int = 300, seed: int = 0):
    """Stations on a 400 km square: a smooth north-south gradient minus 0.6 °C per 100 m of height."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-200, 200, (n, 2))
    height = rng.uniform(0, 1500, n)
    t = 9 - 0.004 * xy[:, 1] - 0.006 * height + rng.normal(0, 0.05, n)
    return xy, height, t


def test_projection_and_split():
    assert np.allclose(to_km(np.array([10.5]), np.array([51.0])), 0)
    assert to_km(np.array([10.5]), np.array([52.0]))[0, 1] == pytest.approx(111.2, abs=0.1)
    train, test = split(100, test_share=0.2, seed=3)
    assert len(test) == 20 and len(train) == 80
    assert not set(train) & set(test)
    assert np.array_equal(split(100, seed=3)[1], test)  # the same seed gives the same split


def test_idw_and_kriging_reproduce_the_stations():
    xy, height, t = _field()
    assert np.allclose(idw(xy, t, xy[:5]), t[:5], atol=1e-3)  # at a station its own value dominates
    smooth = 9 - 0.004 * xy[:, 1]  # without the height the field is smooth, kriging nearly hits the stations
    vario = fit_variogram(xy, smooth)
    assert vario.range_km > 0 and vario.sill > 0
    assert np.allclose(kriging(xy, smooth, xy[:5], vario), smooth[:5], atol=0.1)
    trend = HeightTrend.fit(height, t)
    assert trend.b == pytest.approx(-0.006, abs=0.001)


def test_height_helps_on_a_mountain_field():
    xy, height, t = _field()
    train, test = split(len(t), seed=0)
    plain = kriging(xy[train], t[train], xy[test])
    with_height, _ = kriging_with_height(xy[train], height[train], t[train], xy[test], height[test])
    assert np.sqrt(np.mean((with_height - t[test]) ** 2)) < np.sqrt(np.mean((plain - t[test]) ** 2))


def test_network_learns_the_field():
    xy, height, t = _field()
    train, test = split(len(t), seed=0)
    pred, net = network(xy[train], height[train], t[train], xy[test], height[test],
                        StationNetConfig(steps=300, hidden=32, n_features=16), with_height=True)
    assert pred.shape == (len(test),)
    assert np.sqrt(np.mean((pred - t[test]) ** 2)) < np.std(t[test])
    assert net.n_parameters() > 0
