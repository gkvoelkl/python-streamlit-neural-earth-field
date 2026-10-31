"""The coastline as a field: inside/outside, distance and the comparison of coasts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from python_streamlit_neural_earth_field.coast import (
    DENSIFY_KM,
    CoastScores,
    cell_centres,
    coast_cells,
    coast_distance_km,
    land_mask,
    load_coast_distance,
    ne_path,
)

# A square "continent" from 0° to 40° east and from 10° to 50° north
SQUARE = np.array([[0, 10], [40, 10], [40, 50], [0, 50], [0, 10]], dtype=np.float64)


def test_land_mask_is_true_exactly_inside_the_polygon():
    lon, lat = cell_centres(18)  # 10° cells
    mask = land_mask([SQUARE], lon.astype(np.float64), lat.astype(np.float64))
    inside = (lon[None, :] > 0) & (lon[None, :] < 40) & (lat[:, None] > 10) & (lat[:, None] < 50)
    assert np.array_equal(mask, inside)


def test_a_hole_in_the_polygon_is_sea():
    lake = np.array([[10, 20], [30, 20], [30, 40], [10, 40], [10, 20]], dtype=np.float64)
    mask = land_mask([SQUARE, lake], np.array([5.0, 20.0]), np.array([30.0]))
    assert mask.tolist() == [[True, False]]


def test_distance_is_measured_along_the_sphere():
    meridian = np.array([[0.0, -10.0], [0.0, 10.0]])
    d = coast_distance_km([meridian], np.array([1.0, 0.0], dtype=np.float32), np.array([0.0]))
    assert d[0, 0] == pytest.approx(2 * np.pi * 6371 / 360, rel=1e-3)  # 1° of longitude at the equator
    assert d[0, 1] <= DENSIFY_KM / 2  # on the line, between two of its points


def test_coast_cells_wrap_around_the_date_line():
    land = np.zeros((1, 6), dtype=bool)
    land[0, 0] = True
    assert coast_cells(land).tolist() == [[True, True, False, False, False, True]]


def _write_geojson(path, kind: str, rings: list[np.ndarray]) -> None:
    geometry = (
        {"type": "Polygon", "coordinates": [r.tolist() for r in rings]}
        if kind == "land" else {"type": "LineString", "coordinates": rings[0].tolist()}
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [{"geometry": geometry}]}, f)


def test_signed_distance_is_positive_on_land_and_scores_are_perfect_for_the_truth(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURAL_EARTH_DATA_DIR", str(tmp_path))
    for kind in ("land", "coastline"):
        _write_geojson(ne_path(kind), kind, [SQUARE])
    grid = load_coast_distance(36)
    assert grid.unit == "km"
    inside = land_mask([SQUARE], grid.lon.astype(np.float64), grid.lat.astype(np.float64))
    assert np.all(grid.z[inside] > 0) and np.all(grid.z[~inside] < 0)
    assert grid.z.max() < 2500 and grid.z.min() < -5000  # half the square's width, far ocean

    scores = CoastScores.compare(grid, grid.z > 0)
    assert scores.correct_side == pytest.approx(1.0)
    assert scores.correct_side_at_coast == pytest.approx(1.0)
    assert scores.offset_p90_km <= grid.km_per_cell  # coast cells lie at most a cell from the line
