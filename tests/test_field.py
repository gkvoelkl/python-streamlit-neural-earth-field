"""The properties that would otherwise break silently in this project."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.data import synthetic_grid
from python_streamlit_neural_earth_field.model import (
    PRESETS,
    CoarseContext,
    FieldConfig,
    NeuralField,
    build_field,
)
from python_streamlit_neural_earth_field.training import TrainConfig, train


@pytest.fixture(scope="module")
def grid():
    return synthetic_grid(90)


def _field(preset: str, grid) -> NeuralField:
    cfg = PRESETS[preset]
    coarse = grid.coarsen(cfg.context_ny).z if cfg.use_context else None
    return build_field(cfg, float(grid.z.mean()), float(grid.z.std()), coarse)


@pytest.mark.parametrize(
    "preset", ["Fourier MLP (recommended)", "Fourier MLP + context field", "SIREN (sine activation)"]
)
def test_export_reload_is_exact_in_float32(preset, grid):
    """The loading snippet from the Export tab must yield the same field."""
    field = _field(preset, grid)
    lon = np.array([86.925, -70.0, 179.9, -179.9, 0.0], dtype=np.float32)
    lat = np.array([27.988, -33.0, 0.0, 0.0, 89.5], dtype=np.float32)
    reference = field.predict_grid(lon, lat)

    buffer = io.BytesIO()
    torch.save({"config": field.cfg.to_dict(), "state_dict": field.state_dict()}, buffer)
    buffer.seek(0)

    blob = torch.load(buffer, weights_only=False)
    cfg = FieldConfig(**blob["config"])
    state = {k: v.float() if v.is_floating_point() else v for k, v in blob["state_dict"].items()}
    coarse = state["context.coarse"].numpy() if cfg.use_context else None
    restored = NeuralField(cfg, 0.0, 1.0, coarse)
    restored.load_state_dict(state)
    restored.eval()

    assert np.allclose(restored.predict_grid(lon, lat), reference, atol=1e-4)


def test_xyz_input_has_no_seam_at_the_date_line():
    field = build_field(FieldConfig(), 0.0, 1000.0)
    west = field(torch.tensor([[179.999, 12.0]])).item()
    east = field(torch.tensor([[-179.999, 12.0]])).item()
    assert abs(west - east) < 1e-2


def test_context_hits_its_own_grid_points(grid):
    """Bilinear interpolation must return the grid value at the grid nodes
    -- up to rounding to whole metres (int16)."""
    coarse = grid.coarsen(45)
    context = CoarseContext(coarse.z)
    n = min(len(coarse.lon), len(coarse.lat))
    lonlat = torch.stack(
        [torch.from_numpy(coarse.lon[:n]), torch.from_numpy(coarse.lat[:n])], dim=-1
    )
    expected = np.array([coarse.z[i, i] for i in range(n)])
    assert np.abs(context(lonlat).numpy() - expected).max() <= 0.5


def test_context_costs_are_counted(grid):
    plain = _field("Fourier MLP (recommended)", grid)
    with_context = _field("Fourier MLP + context field", grid)
    assert with_context.context is not None
    overhead = with_context.context.ny * with_context.context.nx * 2
    assert with_context.stored_bytes(4) - plain.stored_bytes(4) >= overhead


def test_fourier_matrix_is_not_stored(grid):
    """B is reconstructed from the seed -- otherwise the encoding would be larger than the network."""
    field = _field("Fourier MLP (recommended)", grid)
    assert "encoding.B" not in field.state_dict()


def test_training_reduces_the_error(grid):
    field = _field("Fourier MLP (recommended)", grid)
    history = [
        p.scores.rmse
        for p in train(field, grid, TrainConfig(steps=200, batch_size=4096, eval_every=100))
    ]
    assert history[-1] < history[0] / 2


def test_area_weighted_sampling_prefers_the_equator(grid):
    from python_streamlit_neural_earth_field.training import Sampler

    weighted = Sampler(grid, torch.device("cpu"), True, 0)(20000)[0].numpy()
    uniform = Sampler(grid, torch.device("cpu"), False, 0)(20000)[0].numpy()
    equator = grid.ny / 2
    assert np.abs(weighted - equator).mean() < np.abs(uniform - equator).mean()
