"""Climate atlas: scaling, one network against three, the size budget, training and scores."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.atlas import (
    ATLAS_PRESETS,
    LAYOUTS,
    Atlas,
    AtlasConfig,
    AtlasField,
    AtlasScores,
    AtlasTrainConfig,
    Scaler,
    annual,
    mlp_parameters,
    train_atlas,
    width_for,
)


def _atlas(n: int = 400) -> Atlas:
    """Random land cells: warm at the equator, wet where it is warm, a few mountains."""
    rng = np.random.default_rng(0)
    lat = rng.uniform(-60, 80, n).astype(np.float32)
    lon = rng.uniform(-180, 180, n).astype(np.float32)
    season = np.sin(2 * np.pi * np.arange(12) / 12)[None, :]
    temperature = (30 * np.cos(np.deg2rad(lat))[:, None] - 5 + 8 * season * np.sign(lat)[:, None]).astype(np.float32)
    precipitation = (np.maximum(temperature, 0) ** 1.5 + rng.gamma(1, 5, (n, 12))).astype(np.float32)
    elevation = (2000 * np.exp(-((lon - 80) / 20) ** 2) + 100)[:, None].astype(np.float32)
    return Atlas(lon, lat, temperature, precipitation, elevation)


def test_scaler_round_trip():
    data = _atlas()
    for mode in ("Standardised", "Raw units"):
        scaler = Scaler(mode).fit(data)
        for q in ("Temperature", "Precipitation", "Elevation"):
            v = data.values(q)
            assert np.allclose(scaler.decode(q, scaler.encode(q, v)), v, atol=1e-3)
    scaler = Scaler("Standardised").fit(data)
    t = scaler.encode("Temperature", data.temperature)
    w = data.weights[:, None] * np.ones_like(t)
    assert np.sum(w * t) / np.sum(w) == pytest.approx(0, abs=1e-5)  # mean 0, by area
    assert Scaler("Raw units").fit(data).encode("Elevation", data.elevation)[0, 0] == data.elevation[0, 0]


def test_separate_networks_get_the_same_budget():
    cfg = AtlasConfig(n_features=16, hidden=64, layers=3)
    shared = AtlasField(cfg, LAYOUTS[0])
    separate = AtlasField(cfg, LAYOUTS[1])
    assert shared.n_parameters() == mlp_parameters(cfg, 64, 25)
    assert separate.n_parameters() <= shared.n_parameters()
    assert separate.n_parameters() > 0.95 * shared.n_parameters()
    assert set(separate.widths()) == {"Temperature", "Precipitation", "Elevation"}
    h = width_for(cfg, 10_000, 12)
    assert mlp_parameters(cfg, h, 12) <= 10_000 < mlp_parameters(cfg, h + 1, 12)
    y = separate(torch.randn(5, 3))
    assert {q: v.shape for q, v in y.items()} == {"Temperature": (5, 12), "Precipitation": (5, 12), "Elevation": (5, 1)}
    sizes = [AtlasField(c).n_parameters() for c in ATLAS_PRESETS.values()]
    assert sizes == sorted(sizes)
    alone = AtlasField(cfg, LAYOUTS[0], ("Temperature",))
    assert alone.n_parameters() == mlp_parameters(cfg, 64, 12)


def test_training_reduces_the_error():
    data = _atlas()
    scaler = Scaler().fit(data)
    field = AtlasField(AtlasConfig(n_features=16, sigma=1.0, hidden=32, layers=2), LAYOUTS[0])
    before = AtlasScores.compare(data, field.predict(data, scaler)).rmse
    losses = [p.loss for p in train_atlas(field, data, scaler, AtlasTrainConfig(steps=200, batch_size=256, eval_every=50),
                                          torch.device("cpu"))]
    after = AtlasScores.compare(data, field.predict(data, scaler)).rmse
    assert losses[-1] < losses[0]
    assert after["Temperature"] < before["Temperature"]
    assert (field.predict(data, scaler)["Precipitation"] >= 0).all()  # the square root keeps rain positive
    assert annual(data, "Precipitation").shape == (data.n,)
