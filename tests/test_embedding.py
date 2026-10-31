"""Location encoders: SatCLIP's spherical harmonics, the small network on top, the temperature places."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.embedding import (
    HIDDEN_LAYER,
    WITH_ELEVATION,
    HeadConfig,
    SatCLIPEncoder,
    SatCLIPHarmonics,
    Temperature,
    encode,
    fit_and_predict,
    sample_places,
    weighted_rmse,
)
from python_streamlit_neural_earth_field.model import PRESETS, build_field


def _places(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n)], axis=1).astype(np.float32)


def test_satclip_harmonics_match_the_expressions_satclip_was_trained_with():
    # A few terms copied from SatCLIP's generated spherical_harmonics_ylm.py
    lonlat = torch.from_numpy(_places(50)).double()
    phi = torch.deg2rad(lonlat[:, 0] + 180)
    theta = torch.deg2rad(lonlat[:, 1] + 90)
    c, s = torch.cos(theta), torch.sqrt(1 - torch.cos(theta) ** 2)
    expected = {
        (0, 0): 0.886226925452758 * torch.ones_like(c),
        (1, -1): 0.48860251190292 * s * torch.sin(phi),
        (1, 0): 1.53499006191973 * c,
        (2, -1): 1.09254843059208 * s * torch.sin(phi) * c,
        (2, 0): 2.97249547320451 * c**2 - 0.990831824401503,
        (2, 2): 0.18209140509868 * (3.0 - 3.0 * c**2) * torch.cos(2 * phi),
        (3, -3): 0.590043589926644 * s**3 * torch.sin(3 * phi),
    }
    y = SatCLIPHarmonics(10)(lonlat)
    assert y.shape == (50, 100)
    for (l, m), value in expected.items():
        assert torch.allclose(y[:, l * l + l + m], value, atol=1e-9), (l, m)


def test_satclip_encoder_gives_256_numbers_per_place():
    enc = SatCLIPEncoder()
    assert enc(torch.from_numpy(_places(7))).shape == (7, 256)


def test_simple_encoders_have_the_expected_width():
    places = _places(10)
    assert encode("lon/lat", places).shape == (10, 2)
    assert encode("Fourier features", places).shape == (10, 256)
    assert encode("Spherical harmonics", places).shape == (10, 100)


def test_encoders_from_the_elevation_network():
    field = build_field(PRESETS["Fourier MLP (recommended)"], 0.0, 1000.0)
    places = _places(10)
    assert encode(HIDDEN_LAYER, places, field).shape == (10, 128)
    with_elevation = encode(WITH_ELEVATION, places, field)
    with torch.no_grad():
        expected = field(torch.from_numpy(places)).numpy() / 1000
    assert np.allclose(with_elevation[:, 2], expected, atol=1e-5)
    with pytest.raises(ValueError):
        encode(HIDDEN_LAYER, places)


def test_small_network_learns_a_smooth_function_from_few_places():
    places = _places(3000)
    target = 30 * np.cos(np.deg2rad(places[:, 1])) - 10  # warm at the equator, cold at the poles
    train = np.arange(300)
    pred = fit_and_predict(encode("lon/lat", places), target, train, HeadConfig(steps=500))
    weights = np.ones(len(places))
    assert weighted_rmse(target[300:], pred[300:], weights[300:]) < 2.0


def test_sample_places_is_area_weighted_and_without_repeats():
    lat = np.repeat(np.array([0.25, 80.25], dtype=np.float32), 5000)
    data = Temperature(np.zeros_like(lat), lat, np.zeros_like(lat))
    idx = sample_places(data, 1000, seed=0)
    assert len(np.unique(idx)) == 1000
    share_equator = np.mean(data.lat[idx] < 1)
    expected = 1 / (1 + math.cos(math.radians(80.25)))
    assert abs(share_equator - expected) < 0.05


def test_temperature_grid_puts_every_place_into_its_cell():
    data = Temperature(np.array([-179.75, 10.25], np.float32), np.array([-89.75, 47.25], np.float32),
                       np.array([-20.0, 9.0], np.float32))
    lon, lat, z = data.grid()
    assert z.shape == (360, 720)
    assert z[0, 0] == -20.0
    assert z[np.searchsorted(lat, 47.25), np.searchsorted(lon, 10.25)] == 9.0
    assert np.isnan(z).sum() == z.size - 2
