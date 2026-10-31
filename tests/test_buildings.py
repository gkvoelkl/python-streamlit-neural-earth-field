"""Buildings: reading CityGML, voxels and signed distance, the field f(E, N, H) and its scores."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from python_streamlit_neural_earth_field.buildings import (
    BuildingConfig,
    BuildingGrid,
    BuildingScores,
    BuildingTrainConfig,
    build_building_field,
    mesh_enh,
    read_buildings,
    signed_distance,
    surface,
    train_building_field,
    voxelize,
)

CENTRE = (726_500.0, 5_434_500.0)


def _ring(points: list[tuple[float, float, float]]) -> str:
    pts = points + [points[0]]
    return " ".join(f"{x} {y} {z}" for x, y, z in pts)


def _polygon(points: list[tuple[float, float, float]]) -> str:
    return (f"<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>{_ring(points)}</gml:posList>"
            "</gml:LinearRing></gml:exterior></gml:Polygon>")


def _house(name: str, x0: float, y0: float, w: float, d: float, ground: float, eaves: float, ridge: float) -> str:
    """A house with a gable roof, ridge running east-west."""
    x1, y1, ym = x0 + w, y0 + d, y0 + d / 2
    surfaces = {
        "GroundSurface": [[(x0, y0, ground), (x0, y1, ground), (x1, y1, ground), (x1, y0, ground)]],
        "RoofSurface": [
            [(x0, y0, eaves), (x1, y0, eaves), (x1, ym, ridge), (x0, ym, ridge)],
            [(x0, ym, ridge), (x1, ym, ridge), (x1, y1, eaves), (x0, y1, eaves)],
        ],
        "WallSurface": [[(x0, y0, ground), (x1, y0, ground), (x1, y0, eaves), (x0, y0, eaves)]],
    }
    bounded = "".join(
        f"<bldg:boundedBy><bldg:{kind}><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        f"{_polygon(p)}</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:{kind}></bldg:boundedBy>"
        for kind, polys in surfaces.items() for p in polys
    )
    return f'<bldg:Building gml:id="{name}"><gml:name>{name}</gml:name>{bounded}</bldg:Building>'


def _citygml(tmp_path, houses: list[str]):
    path = tmp_path / "test.gml"
    members = "".join(f"<cityObjectMember>{h}</cityObjectMember>" for h in houses)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<CityModel xmlns="http://www.opengis.net/citygml/1.0" xmlns:gml="http://www.opengis.net/gml" '
        f'xmlns:bldg="http://www.opengis.net/citygml/building/1.0">{members}</CityModel>',
        encoding="utf-8",
    )
    return path


def _grid(tmp_path, res: float = 1.0) -> BuildingGrid:
    house = _house("House", 726_490.0, 5_434_490.0, 20.0, 10.0, 330.0, 336.0, 340.0)
    far = _house("Far away", 727_500.0, 5_435_500.0, 10.0, 10.0, 330.0, 333.0, 335.0)
    buildings = read_buildings(_citygml(tmp_path, [house, far]), centre=CENTRE, size_m=40.0)
    inside, east, north, height = voxelize(buildings, res, centre=CENTRE, size_m=40.0, height_m=16.0)
    return BuildingGrid(inside, signed_distance(inside, res), east, north, height,
                        len(buildings), sum(b.n_points for b in buildings), sum(b.gml_bytes for b in buildings))


def test_read_keeps_only_buildings_in_the_square(tmp_path):
    house = _house("House", 726_490.0, 5_434_490.0, 20.0, 10.0, 330.0, 336.0, 340.0)
    far = _house("Far away", 727_500.0, 5_435_500.0, 10.0, 10.0, 330.0, 333.0, 335.0)
    buildings = read_buildings(_citygml(tmp_path, [house, far]), centre=CENTRE, size_m=40.0)
    assert [b.name for b in buildings] == ["House"]
    b = buildings[0]
    assert b.ground == pytest.approx(330.0)
    assert len(b.roofs) == 2
    assert b.n_points == 4 * 5  # four surfaces, closed rings of five points
    assert b.gml_bytes > 0


def test_voxels_follow_footprint_and_roof(tmp_path):
    g = _grid(tmp_path)
    assert g.shape == (16, 40, 40)
    assert g.origin == pytest.approx((726_480.0, 5_434_480.0, 328.0))

    def at(e: float, n: float, h: float) -> bool:
        k = int(np.argmin(abs(g.height - h)))
        i = int(np.argmin(abs(g.north - n)))
        j = int(np.argmin(abs(g.east - e)))
        return bool(g.inside[k, i, j])

    assert at(726_500.5, 5_434_495.5, 339.5)  # under the ridge
    assert not at(726_500.5, 5_434_490.5, 339.5)  # above the eaves at the wall
    assert at(726_500.5, 5_434_490.5, 335.5)
    assert not at(726_500.5, 5_434_495.5, 329.5)  # below the ground of the house
    assert not at(726_485.5, 5_434_495.5, 333.5)  # beside the house
    # volume: 20 × 10 × 6 m up to the eaves plus the roof prism 20 × 10 × 4 / 2
    assert g.inside.sum() == pytest.approx(20 * 10 * 6 + 20 * 10 * 4 / 2, rel=0.08)


def test_signed_distance_is_negative_inside_and_grows_outward(tmp_path):
    g = _grid(tmp_path)
    assert (g.sdf[g.inside] < 0).all() and (g.sdf[~g.inside] > 0).all()
    i = int(np.argmin(abs(g.north - 5_434_495.5)))
    k = int(np.argmin(abs(g.height - 333.5)))
    row = g.sdf[k, i]
    j = int(np.argmin(abs(g.east - 726_484.5)))  # 5.5 m west of the wall
    assert row[j] == pytest.approx(5.5, abs=0.01)
    assert np.abs(signed_distance(np.zeros((3, 3, 3), dtype=bool), 1.0)).min() == np.inf


def test_field_scales_all_axes_alike_and_stays_within_the_limit(tmp_path):
    g = _grid(tmp_path)
    field = build_building_field(BuildingConfig(n_features=16, hidden=32, layers=2, limit_m=3.0), g)
    xyz = field.scale(np.array([g.origin, (g.origin[0] + 40.0, g.origin[1] + 40.0, g.origin[2] + 16.0)]))
    assert field.size_m == 300.0  # the default square of the app
    assert xyz[0] == pytest.approx([-1, -1, -1])
    assert xyz[1, 2] - xyz[0, 2] == pytest.approx(16 / 300 * 2)
    d = field(torch.tensor([[*g.origin], [726_500.0, 5_434_495.0, 335.0]], dtype=torch.float64))
    assert d.shape == (2,) and (d.abs() <= 3.0).all()
    assert field.stored_bytes() == field.n_parameters() * 2 + 28


def test_training_finds_the_house_and_the_scores_see_it(tmp_path):
    g = _grid(tmp_path)
    field = build_building_field(BuildingConfig(n_features=32, sigma=4.0, hidden=64, layers=2, limit_m=4.0), g)
    field.size_m = 40.0  # let the small test grid fill the scaled cube
    last = None
    for last in train_building_field(field, g, BuildingTrainConfig(steps=800, batch_size=4096, eval_every=100),
                                     torch.device("cpu")):
        pass
    assert last.step == 800 and last.correct_side > 0.95
    predicted = field.render(g)
    scores = BuildingScores.compare(g, predicted, 4.0)
    assert scores.iou > 0.7
    assert scores.offset_median_m < 1.0
    perfect = BuildingScores.compare(g, np.clip(g.sdf, -4, 4), 4.0)
    assert perfect.iou == 1.0 and perfect.rmse_m == pytest.approx(0.0)


def test_surface_comes_back_in_utm_coordinates(tmp_path):
    g = _grid(tmp_path)
    verts, faces = surface(g.sdf, g.res)
    enh = mesh_enh(g, verts)
    assert len(faces) > 0
    assert enh[:, 0].min() == pytest.approx(726_490.0, abs=0.6)
    assert enh[:, 0].max() == pytest.approx(726_510.0, abs=0.6)
    assert enh[:, 1].min() == pytest.approx(5_434_490.0, abs=0.6)
    assert enh[:, 2].max() == pytest.approx(340.0, abs=1.1)
    assert len(surface(np.ones((4, 4, 4)), 1.0)[0]) == 0
