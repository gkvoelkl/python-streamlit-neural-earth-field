"""The properties that would otherwise break silently in this project."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.data import synthetic_grid
from python_streamlit_neural_earth_field.encoding import SphericalHarmonics, lonlat_to_xyz
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
    "preset",
    [
        "Fourier MLP (recommended)",
        "Fourier MLP + context field",
        "SIREN (sine activation)",
        "Spherical harmonics (L = 16)",
    ],
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


def test_spherical_harmonics_are_orthonormal():
    """Area-weighted mean of Y_i · Y_j over the sphere must be the identity matrix."""
    sh = SphericalHarmonics(3, 8)
    ny, nx = 200, 400
    lat = torch.linspace(-90 + 90 / ny, 90 - 90 / ny, ny, dtype=torch.float64)
    lon = torch.linspace(-180 + 180 / nx, 180 - 180 / nx, nx, dtype=torch.float64)
    lat2d, lon2d = torch.meshgrid(lat, lon, indexing="ij")
    Y = sh(lonlat_to_xyz(lon2d.reshape(-1), lat2d.reshape(-1)))
    w = torch.cos(torch.deg2rad(lat2d.reshape(-1)))
    gram = (Y * (w / w.sum())[:, None]).T @ Y
    assert Y.shape[1] == 81
    assert (gram - torch.eye(81, dtype=gram.dtype)).abs().max() < 1e-3


def test_spherical_harmonics_have_no_seam_and_no_pole_problem():
    sh = SphericalHarmonics(3, 20)
    west = sh(lonlat_to_xyz(torch.tensor([179.9999]), torch.tensor([33.0])))
    east = sh(lonlat_to_xyz(torch.tensor([-179.9999]), torch.tensor([33.0])))
    assert (west - east).abs().max() < 1e-3
    assert torch.isfinite(sh(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))).all()


def test_spherical_harmonics_store_nothing():
    field = build_field(FieldConfig(encoding="spherical", sh_degree=10), 0.0, 1.0)
    assert not any(k.startswith("encoding.") for k in field.state_dict())


def test_spherical_harmonics_require_xyz_input():
    with pytest.raises(ValueError):
        build_field(FieldConfig(encoding="spherical", input_mode="lonlat"), 0.0, 1.0)


def test_zero_hidden_layers_is_a_linear_model():
    """Without hidden layers only the head remains: elevation = w · features + b."""
    field = build_field(FieldConfig(encoding="spherical", sh_degree=6, layers=0), 0.0, 1.0)
    assert field.n_parameters() == 7**2 + 1
    lonlat = torch.tensor([[10.0, 20.0], [-120.0, -45.0]])
    features = field.encoding(field.transform(lonlat))
    expected = features @ field.head.weight[0] + field.head.bias
    assert torch.allclose(field(lonlat), expected, atol=1e-5)
