"""Gaussian splatting: the Gaussians, their file size, drawing and training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.buildings import BuildingGrid
from python_streamlit_neural_earth_field.ortho import Orthophoto, image_psnr
from python_streamlit_neural_earth_field.splat import (
    SPLAT_PRESETS,
    VOLUME_PRESETS,
    SplatConfig,
    SplatTrainConfig,
    VolumeConfig,
    VolumeTrainConfig,
    build_gaussian_image,
    build_gaussian_volume,
    train_gaussian_volume,
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


# --------------------------------------------------------------------------- 3D


def _grid(res: float = 1.0) -> BuildingGrid:
    """A 24 m × 24 m × 12 m block with one house of 8 m × 6 m × 7 m, corner at E 726 000, N 5 434 000, H 330."""
    n_xy, n_z = int(24 / res), int(12 / res)
    east = 726_000 + res * (np.arange(n_xy) + 0.5)
    north = 5_434_024 - res * (np.arange(n_xy) + 0.5)
    height = 330 + res * (np.arange(n_z) + 0.5)
    z, y, x = np.meshgrid(height - 330, north - 5_434_000, east - 726_000, indexing="ij")
    inside = (x > 6) & (x < 14) & (y > 10) & (y < 16) & (z < 7)
    return BuildingGrid(inside, np.where(inside, -1.0, 1.0).astype(np.float32), east, north, height, 1, 8, 100)


def test_volume_presets_match_the_file_sizes_of_the_networks():
    from python_streamlit_neural_earth_field.buildings import BUILDING_PRESETS, build_building_field

    grid = _grid()
    for name, cfg in VOLUME_PRESETS.items():
        network = build_building_field(BUILDING_PRESETS[name], grid).stored_bytes()
        assert build_gaussian_volume(cfg, grid).stored_bytes() == pytest.approx(network, rel=0.001)


def test_volume_starts_inside_the_buildings_and_draws_them():
    grid = _grid()
    model = build_gaussian_volume(VolumeConfig(n_gaussians=60), grid)
    c = model.centre.detach().numpy()
    assert np.all((c[:, 0] > 5) & (c[:, 0] < 15) & (c[:, 1] > 9) & (c[:, 1] < 17) & (c[:, 2] < 8))
    field = model.render(grid)
    assert field.shape == grid.shape
    assert field[grid.inside].mean() < field[~grid.inside].mean()  # more building where the house is


def test_gradient_in_pieces_is_the_full_gradient():
    grid = _grid()
    model = build_gaussian_volume(VolumeConfig(n_gaussians=40), grid)
    target = torch.from_numpy(grid.inside).float()
    # all at once
    occ = torch.zeros(grid.inside.size).index_add(0, *[t.flatten() for t in model._window(slice(None), grid.shape, 1.0)])
    torch.mean((occ - target.flatten()) ** 2).backward()
    full = model.centre.grad.clone()
    # in pieces, as in training
    model.zero_grad()
    residual = (2 * (model.occupancy(grid.shape, 1.0) - target) / target.numel()).flatten()
    for part in (slice(0, 15), slice(15, 40)):
        flat, weight = model._window(part, grid.shape, 1.0)
        (residual[flat] * weight).sum().backward()
    assert torch.allclose(model.centre.grad, full, atol=1e-7)


def test_volume_training_puts_more_voxels_on_the_right_side():
    grid = _grid()
    model = build_gaussian_volume(VolumeConfig(n_gaussians=40), grid)
    progress = list(train_gaussian_volume(model, grid, VolumeTrainConfig(steps=60, eval_every=20),
                                          torch.device("cpu")))
    assert progress[-1].step == 60
    assert progress[-1].correct_side > progress[0].correct_side
    pred = model.render(grid) < 0
    iou = (pred & grid.inside).sum() / (pred | grid.inside).sum()
    assert iou > 0.8
