"""Orthophotos: reading a GeoTIFF tile, the field f(E, N) → RGB, PSNR and the JPEG comparison."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.ortho import (
    OrthoConfig,
    OrthoTrainConfig,
    Orthophoto,
    build_ortho_field,
    image_psnr,
    jpeg_at_size,
    load_orthophoto,
    psnr,
    tile_path,
    train_ortho,
)


def _photo(n: int = 40) -> Orthophoto:
    """A smooth colour pattern on a 1 km tile with the corner at E 726 000, N 5 434 000."""
    step = 1000 / n
    east = 726_000 + step * (np.arange(n) + 0.5)
    north = 5_435_000 - step * (np.arange(n) + 0.5)
    y, x = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    rgb = np.stack([200 * x + 20, 200 * y + 20, 120 + 80 * np.sin(6 * x)], axis=-1)
    return Orthophoto(rgb.round().astype(np.uint8), east, north, "test")


def test_load_reads_the_georeference_and_averages_pixels(tmp_path, monkeypatch):
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    monkeypatch.setenv("NEURAL_EARTH_DATA_DIR", str(tmp_path))
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:50] = 200  # northern half bright
    tags = ImageFileDirectory_v2()
    tags[33550] = (10.0, 10.0, 0.0)  # 10 m per pixel
    tags[33922] = (0.0, 0.0, 0.0, 726_000.0, 5_435_000.0, 0.0)  # upper-left corner
    Image.fromarray(rgb).save(tile_path("test"), tiffinfo=tags)

    photo = load_orthophoto(50, tile="test", use_cache=False)
    assert photo.rgb.shape == (50, 50, 3)
    assert photo.metres_per_pixel == pytest.approx(20.0)
    assert photo.east[0] == pytest.approx(726_010.0)
    assert photo.north[0] == pytest.approx(5_434_990.0)  # first row = north
    assert photo.rgb[0, 0, 0] == 200 and photo.rgb[-1, 0, 0] == 0
    with pytest.raises(ValueError):
        load_orthophoto(30, tile="test", use_cache=False)


def test_field_returns_colours_in_range_at_metre_coordinates():
    photo = _photo()
    field = build_ortho_field(OrthoConfig(n_features=16, hidden=32, layers=2), photo)
    assert field.origin == pytest.approx((726_000.0, 5_434_000.0))
    en = torch.tensor([[726_000.0, 5_434_000.0], [726_500.0, 5_434_500.0], [727_000.0, 5_435_000.0]],
                      dtype=torch.float64)
    xy = field.scale(en)
    assert torch.allclose(xy, torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]), atol=1e-6)
    rgb = field(en)
    assert rgb.shape == (3, 3)
    assert (rgb >= 0).all() and (rgb <= 255).all()


def test_render_matches_the_grid_and_can_be_finer():
    photo = _photo()
    field = build_ortho_field(OrthoConfig(n_features=16, hidden=32, layers=2), photo)
    image = field.render(photo)
    assert image.shape == photo.rgb.shape and image.dtype == np.uint8
    rows, cols = photo.crop(726_500, 5_434_500, 250)
    assert field.render(photo, rows, cols, upsample=3).shape == (30, 30, 3)
    # rendering at the training pixels gives the same colours as the forward pass
    en = torch.tensor([[photo.east[3], photo.north[7]]], dtype=torch.float64)
    assert np.allclose(image[7, 3], np.rint(field(en).detach().numpy()[0]), atol=1)


def test_stored_bytes_count_float16_weights_corner_and_seed():
    field = build_ortho_field(OrthoConfig(n_features=16, hidden=32, layers=2), _photo())
    assert field.stored_bytes() == 2 * field.n_parameters() + 16 + 4


def test_training_improves_the_psnr():
    photo = _photo()
    field = build_ortho_field(OrthoConfig(n_features=32, sigma=4.0, hidden=64, layers=2), photo)
    before = image_psnr(photo.rgb, field.render(photo))
    history = list(train_ortho(field, photo, OrthoTrainConfig(steps=300, batch_size=1024, eval_every=100),
                               torch.device("cpu")))
    after = image_psnr(photo.rgb, field.render(photo))
    assert history[-1].step == 300
    assert after > before + 10
    assert after > 30


def test_psnr_of_known_errors():
    assert psnr(255.0**2) == pytest.approx(0.0)
    assert psnr(0.0) == float("inf")
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert image_psnr(a, a + 1) == pytest.approx(20 * np.log10(255))


def test_jpeg_stays_within_the_budget_and_gets_better_with_more_bytes():
    rng = np.random.default_rng(0)
    rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    small, small_bytes, q_small = jpeg_at_size(rgb, 3000)
    large, large_bytes, q_large = jpeg_at_size(rgb, 12000)
    assert small_bytes <= 3000 and large_bytes <= 12000
    assert q_large > q_small
    assert image_psnr(rgb, large) > image_psnr(rgb, small)
    assert small.shape == rgb.shape
