"""Gaussian splatting: the Gaussians, their file size, drawing and training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.ortho import Orthophoto, image_psnr
from python_streamlit_neural_earth_field.splat import (
    SPLAT_PRESETS,
    SplatConfig,
    SplatTrainConfig,
    build_gaussian_image,
    train_splat,
)


def _photo(n: int = 40) -> Orthophoto:
    """A colour pattern with a sharp edge on a 1 km tile with the corner at E 726 000, N 5 434 000."""
    step = 1000 / n
    east = 726_000 + step * (np.arange(n) + 0.5)
    north = 5_435_000 - step * (np.arange(n) + 0.5)
    y, x = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    rgb = np.stack([200 * x + 20, 200 * y + 20, np.where(x > 0.5, 220, 40)], axis=-1)
    return Orthophoto(rgb.round().astype(np.uint8), east, north, "test")


def test_presets_match_the_file_sizes_of_the_networks():
    sizes = {name: build_gaussian_image(cfg, _photo()).stored_bytes() for name, cfg in SPLAT_PRESETS.items()}
    assert sizes["Small"] == pytest.approx(162 * 1024, rel=0.03)
    assert sizes["Medium"] == pytest.approx(644 * 1024, rel=0.03)
    assert sizes["Large"] == pytest.approx(2.5 * 1024**2, rel=0.03)


def test_gaussians_start_on_a_grid_with_the_corner_of_the_tile():
    model = build_gaussian_image(SplatConfig(grid=10), _photo())
    assert model.n_gaussians == 100
    assert model.n_parameters() == 800
    assert model.stored_bytes() == 800 * 2 + 16
    assert model.origin == pytest.approx((726_000.0, 5_434_000.0))
    assert model.size_m == pytest.approx(1000.0)
    centre = model.centre.detach()
    assert centre.min() == pytest.approx(-0.9) and centre.max() == pytest.approx(0.9)
    assert torch.all(model.width() <= model.sigma_max)


def test_a_single_gaussian_is_drawn_where_it_belongs():
    model = build_gaussian_image(SplatConfig(grid=1, sigma_max=0.2), _photo())
    with torch.no_grad():
        model.centre.copy_(torch.tensor([[0.5, 0.5]]))  # north-east quarter
        model.colour.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
    image = model.splat(40).detach()
    row, col = np.unravel_index(int(image[..., 0].argmax()), (40, 40))
    assert (row, col) in {(9, 29), (9, 30), (10, 29), (10, 30)}  # 3/4 to the east, 1/4 from the top
    assert image[..., 0].max() == pytest.approx(1.0, abs=0.05)
    assert float(image[..., 1:].abs().max()) == 0.0
    assert float(image[0, 0, 0]) == 0.0  # far away: outside the window of the Gaussian


def test_training_improves_the_image_and_renders_uint8():
    photo = _photo()
    model = build_gaussian_image(SplatConfig(grid=12), photo)
    before = image_psnr(photo.rgb, np.zeros_like(photo.rgb))
    progress = list(train_splat(model, photo, SplatTrainConfig(steps=150, eval_every=50), torch.device("cpu")))
    assert progress[-1].step == 150
    assert progress[-1].psnr_db > progress[0].psnr_db
    image = model.render(photo)
    assert image.dtype == np.uint8 and image.shape == photo.rgb.shape
    assert image_psnr(photo.rgb, image) > max(before, 22.0)
    assert math.isfinite(progress[-1].psnr_db)
