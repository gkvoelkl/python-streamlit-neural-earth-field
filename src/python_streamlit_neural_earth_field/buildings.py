"""Buildings as a neural field in three dimensions: f(E, N, H) → signed distance.

DeepSDF at the scale of a town.  The data are the LoD2 building models of the
Bavarian surveying administration: every building is a closed solid of ground,
wall and roof surfaces, in ETRS89 / UTM zone 32N with heights above sea level
(DHHN2016), free under CC BY 4.0 (Bayerische Vermessungsverwaltung –
www.geodaten.bayern.de).  One tile covers 2 km × 2 km; the default one contains
the old town of Regensburg with the cathedral, the same place as the
orthophoto in ``ortho.py``.

LoD2 buildings have vertical walls and no overhangs: above every point of the
footprint a building reaches from its ground up to its roof.  That makes the
step from surfaces to a 3D grid of inside/outside simple -- rasterise each roof
polygon, take its plane as the top and the ground of the building as the
bottom.  The signed distance (inside negative, as on land in ``coast.py``) then
comes from the Euclidean distance transform of that grid.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import cache_dir, data_dir, download_etopo
from .encoding import FourierFeatures

TILE = "726_5434"  # lower-left corner in km of a 2 km tile: E 726, N 5434 -- Regensburg
TILE_URL = "https://download1.bayernwolke.de/a/lod2/citygml/{tile}.gml"
TILE_MB = 108
ATTRIBUTION = "Bayerische Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0)"

CENTRE = (726_517.0, 5_434_245.0)  # the cathedral St. Peter
SIZE_M = 300.0  # edge length of the square around the centre
HEIGHT_M = 120.0  # the cathedral's towers reach 107 m above its ground

GML = "{http://www.opengis.net/gml}"
BLDG = "{http://www.opengis.net/citygml/building/1.0}"


def tile_path(tile: str = TILE) -> Path:
    d = data_dir() / "lod2"
    d.mkdir(exist_ok=True)
    return d / f"{tile}.gml"


def download_tile(tile: str = TILE, progress: Callable[[int, int | None], None] | None = None) -> Path:
    path = tile_path(tile)
    if not path.exists():
        download_etopo(progress, url=TILE_URL.format(tile=tile), target=path)
    return path


# --------------------------------------------------------------------------- reading CityGML


@dataclass
class Building:
    name: str
    ground: float  # lowest point of the ground surface, metres above sea level
    roofs: list[list[np.ndarray]]  # per roof polygon its rings (outer first, then holes), each (n, 3): E, N, H
    n_points: int  # all coordinates of the solid, for the size of the vector data
    gml_bytes: int  # length of the building's CityGML text


def _rings(polygon: ET.Element) -> list[np.ndarray]:
    out = []
    for pos in polygon.iter(f"{GML}posList"):
        out.append(np.array(pos.text.split(), dtype=np.float64).reshape(-1, 3))
    return out


def read_buildings(path: Path, centre: tuple[float, float] = CENTRE, size_m: float = SIZE_M) -> list[Building]:
    """All buildings whose footprint touches the square of ``size_m`` around ``centre``."""
    half = size_m / 2
    x0, x1, y0, y1 = centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half
    out: list[Building] = []
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != f"{BLDG}Building":
            continue
        text = " ".join(pos.text for pos in elem.iter(f"{GML}posList"))
        coords = np.array(text.split(), dtype=np.float64).reshape(-1, 3)
        if coords[:, 0].max() >= x0 and coords[:, 0].min() <= x1 and coords[:, 1].max() >= y0 and coords[:, 1].min() <= y1:
            ground = [r[:, 2].min() for g in elem.iter(f"{BLDG}GroundSurface") for r in _rings(g)]
            roofs = [_rings(poly) for roof in elem.iter(f"{BLDG}RoofSurface") for poly in roof.iter(f"{GML}Polygon")]
            name = elem.findtext(f"{GML}name") or ""
            out.append(Building(name, float(min(ground) if ground else coords[:, 2].min()), roofs,
                                len(coords), len(ET.tostring(elem))))
        elem.clear()
    return out


# --------------------------------------------------------------------------- from surfaces to a grid


def _inside(rings: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd test of points against all rings of a polygon (holes count automatically)."""
    inside = np.zeros(x.shape, dtype=bool)
    for ring in rings:
        xa, ya = ring[:-1, 0], ring[:-1, 1]
        xb, yb = ring[1:, 0], ring[1:, 1]
        for i in range(len(xa)):
            crosses = (ya[i] > y) != (yb[i] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = xa[i] + (y - ya[i]) * (xb[i] - xa[i]) / (yb[i] - ya[i])
            inside ^= crosses & (x < xc)
    return inside


def _plane(ring: np.ndarray) -> tuple[float, float, float] | None:
    """z = a·x + b·y + c through the points of a roof ring; None for (nearly) vertical parts."""
    p = ring - ring.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    normal = vt[-1]
    if abs(normal[2]) < 0.05:
        return None
    a, b = -normal[0] / normal[2], -normal[1] / normal[2]
    m = ring.mean(axis=0)
    return a, b, m[2] - a * m[0] - b * m[1]


@dataclass
class BuildingGrid:
    """Inside/outside and signed distance on a regular 3D grid of voxel centres."""

    inside: np.ndarray  # (nz, ny, nx) bool, first row = north, first layer = bottom
    sdf: np.ndarray  # (nz, ny, nx) float32, metres, inside negative
    east: np.ndarray  # (nx,) voxel centres
    north: np.ndarray  # (ny,) north → south
    height: np.ndarray  # (nz,) metres above sea level, bottom → top
    n_buildings: int
    n_points: int  # coordinates of all buildings in the vector data
    gml_bytes: int

    @property
    def res(self) -> float:
        return float(self.east[1] - self.east[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.inside.shape

    @property
    def n_voxels(self) -> int:
        return self.inside.size

    @property
    def origin(self) -> tuple[float, float, float]:
        """Lower south-west corner of the grid."""
        h = self.res / 2
        return float(self.east[0] - h), float(self.north[-1] - h), float(self.height[0] - h)


def voxelize(buildings: list[Building], res: float = 1.0, centre: tuple[float, float] = CENTRE,
             size_m: float = SIZE_M, height_m: float = HEIGHT_M) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inside/outside on voxel centres: each roof polygon fills the column from the building's ground to its plane."""
    n = int(round(size_m / res))
    nz = int(round(height_m / res))
    x0, y1 = centre[0] - size_m / 2, centre[1] + size_m / 2
    east = x0 + res * (np.arange(n) + 0.5)
    north = y1 - res * (np.arange(n) + 0.5)
    z0 = math.floor(min(b.ground for b in buildings)) - 2.0
    height = z0 + res * (np.arange(nz) + 0.5)
    bottom = np.full((n, n), np.inf)
    top = np.full((n, n), -np.inf)
    for b in buildings:
        for rings in b.roofs:
            plane = _plane(rings[0])
            if plane is None:
                continue
            outer = rings[0]
            cols = np.nonzero((east >= outer[:, 0].min() - res) & (east <= outer[:, 0].max() + res))[0]
            rows = np.nonzero((north >= outer[:, 1].min() - res) & (north <= outer[:, 1].max() + res))[0]
            if not len(cols) or not len(rows):
                continue
            yy, xx = np.meshgrid(north[rows], east[cols], indexing="ij")
            m = _inside(rings, xx, yy)
            if not m.any():
                continue
            a, bb, c = plane
            z = a * xx + bb * yy + c
            r, k = np.nonzero(m)
            r, k = rows[r], cols[k]
            top[r, k] = np.maximum(top[r, k], z[m])
            bottom[r, k] = np.minimum(bottom[r, k], b.ground)
    inside = (height[:, None, None] >= bottom[None]) & (height[:, None, None] <= top[None])
    return inside, east, north, height


def signed_distance(inside: np.ndarray, res: float) -> np.ndarray:
    """Distance to the nearest building surface in metres; inside negative.

    The distance transform measures from voxel centre to voxel centre; half a voxel
    puts the surface between the last inside and the first outside voxel.
    """
    from scipy.ndimage import distance_transform_edt

    if not inside.any():
        return np.full(inside.shape, np.inf, dtype=np.float32)
    out = distance_transform_edt(~inside).astype(np.float32) * res
    inn = distance_transform_edt(inside).astype(np.float32) * res
    return np.where(inside, -(inn - res / 2), out - res / 2).astype(np.float32)


def load_building_grid(res: float = 1.0, tile: str = TILE, use_cache: bool = True) -> BuildingGrid:
    tag = f"{res:g}".replace(".", "p")
    cache = cache_dir() / f"lod2_{tile}_{int(SIZE_M)}m_{tag}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        inside = d["inside"]
        return BuildingGrid(inside, d["sdf"], d["east"], d["north"], d["height"],
                            int(d["n_buildings"]), int(d["n_points"]), int(d["gml_bytes"]))
    buildings = read_buildings(tile_path(tile))
    inside, east, north, height = voxelize(buildings, res)
    sdf = signed_distance(inside, res)
    grid = BuildingGrid(inside, sdf, east, north, height, len(buildings),
                        sum(b.n_points for b in buildings), sum(b.gml_bytes for b in buildings))
    np.savez_compressed(cache, inside=inside, sdf=sdf, east=east, north=north, height=height,
                        n_buildings=grid.n_buildings, n_points=grid.n_points, gml_bytes=grid.gml_bytes)
    return grid


# --------------------------------------------------------------------------- the network


@dataclass
class BuildingConfig:
    n_features: int = 256  # Fourier frequencies
    sigma: float = 20.0  # bandwidth, in waves per 300 m
    hidden: int = 256
    layers: int = 4
    limit_m: float = 5.0  # distances beyond ±limit are cut off (chapter 12)
    fourier_seed: int = 0


BUILDING_PRESETS: dict[str, BuildingConfig] = {
    "Small": BuildingConfig(n_features=128, hidden=128, layers=4),
    "Medium": BuildingConfig(n_features=256, hidden=256, layers=4),
    "Large": BuildingConfig(n_features=256, hidden=512, layers=5),
}


class BuildingField(nn.Module):
    """(E, N, H) in metres → signed distance in metres, cut off at ±limit.

    All three coordinates are scaled with the same factor, so a metre is a metre
    in every direction: the square becomes −1 … 1, the height starts at −1.  The
    corner of the grid (three numbers) belongs to the file, like the seed.
    """

    def __init__(self, cfg: BuildingConfig, origin: tuple[float, float, float], size_m: float = SIZE_M):
        super().__init__()
        self.cfg = cfg
        self.origin = tuple(float(v) for v in origin)
        self.size_m = size_m
        self.encoding = FourierFeatures(3, cfg.n_features, cfg.sigma / 2, cfg.fourier_seed)
        dims = [self.encoding.out_dim] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)

    def forward_unit(self, xyz: torch.Tensor) -> torch.Tensor:
        """Scaled coordinates → distance divided by the limit.

        The output layer is linear; the targets are cut off at ±1, and so is the
        answer when the field is queried.  (A tanh at the end gave the same results.)
        """
        return self.head(self.body(self.encoding(xyz))).squeeze(-1)

    def scale(self, enh: np.ndarray) -> np.ndarray:
        """Metres → scaled coordinates (in float64 first: UTM northings have 7 digits)."""
        return (np.asarray(enh, dtype=np.float64) - np.array(self.origin)) / self.size_m * 2 - 1

    def forward(self, enh: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        xyz = torch.from_numpy(self.scale(enh.cpu().numpy()).astype(np.float32)).to(device)
        return self.forward_unit(xyz).clamp(-1, 1) * self.cfg.limit_m

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stored_bytes(self, bytes_per_param: int = 2) -> int:
        """Weights (float16 by default) + corner of the grid (3 × float64) + seed."""
        return self.n_parameters() * bytes_per_param + 24 + 4

    @torch.no_grad()
    def render(self, grid: BuildingGrid) -> np.ndarray:
        """The distance as the network sees it, on the voxel centres of ``grid`` (metres)."""
        device = next(self.parameters()).device
        x0, y0, z0 = self.origin
        s = lambda v, o: torch.from_numpy(((v - o) / self.size_m * 2 - 1).astype(np.float32)).to(device)
        xe, yn, zh = s(grid.east, x0), s(grid.north, y0), s(grid.height, z0)
        out = np.empty(grid.shape, dtype=np.float32)
        was_training = self.training
        self.eval()
        for k in range(len(zh)):
            yy, xx = torch.meshgrid(yn, xe, indexing="ij")
            xyz = torch.stack([xx, yy, torch.full_like(xx, float(zh[k]))], -1).reshape(-1, 3)
            out[k] = (self.forward_unit(xyz).clamp(-1, 1) * self.cfg.limit_m).cpu().numpy().reshape(len(yn), len(xe))
        if was_training:
            self.train()
        return out


def build_building_field(cfg: BuildingConfig, grid: BuildingGrid) -> BuildingField:
    return BuildingField(cfg, grid.origin, SIZE_M)


# --------------------------------------------------------------------------- training


@dataclass
class BuildingTrainConfig:
    steps: int = 3000
    batch_size: int = 65536
    lr: float = 2e-3
    lr_final_factor: float = 0.02
    warmup: int = 100
    eval_every: int = 100
    seed: int = 0


@dataclass
class BuildingProgress:
    step: int
    total: int
    correct_side: float  # share of a fixed sample of voxels on the right side of the surface
    elapsed_s: float


def train_building_field(field: BuildingField, grid: BuildingGrid, cfg: BuildingTrainConfig,
                         device: torch.device) -> Iterator[BuildingProgress]:
    """Overfits the field to the clipped signed distance of this one grid; every voxel counts the same."""
    torch.manual_seed(cfg.seed)
    field.to(device).train()
    limit = field.cfg.limit_m
    x0, y0, z0 = field.origin
    s = lambda v, o: torch.from_numpy(((v - o) / field.size_m * 2 - 1).astype(np.float32)).to(device)
    xe, yn, zh = s(grid.east, x0), s(grid.north, y0), s(grid.height, z0)
    target = torch.from_numpy(np.clip(grid.sdf, -limit, limit) / limit).to(device)
    nz, ny, nx = grid.shape
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.randint(0, nz, (n,), generator=gen).to(device)
        i = torch.randint(0, ny, (n,), generator=gen).to(device)
        j = torch.randint(0, nx, (n,), generator=gen).to(device)
        return torch.stack([xe[j], yn[i], zh[k]], -1), target[k, i, j]

    eval_xyz, eval_t = sample(100_000)
    optimiser = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    started = time.perf_counter()
    for step in range(cfg.steps):
        if step < cfg.warmup:
            lr = cfg.lr * (step + 1) / cfg.warmup
        else:
            t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
            lr = cfg.lr * (cfg.lr_final_factor + (1 - cfg.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * t)))
        for group in optimiser.param_groups:
            group["lr"] = lr
        xyz, t = sample(cfg.batch_size)
        loss = torch.mean((field.forward_unit(xyz) - t) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                p = field.forward_unit(eval_xyz)
                correct = torch.mean(((p < 0) == (eval_t < 0)).float()).item()
            yield BuildingProgress(step + 1, cfg.steps, correct, time.perf_counter() - started)
    field.eval()


# --------------------------------------------------------------------------- scores


@dataclass
class BuildingScores:
    iou: float  # volume in both / volume in either
    correct_side: float  # share of all voxels on the right side
    offset_median_m: float  # distance of the network's surface from the true surface
    offset_p90_m: float
    rmse_m: float  # of the clipped distance

    @classmethod
    def compare(cls, grid: BuildingGrid, predicted: np.ndarray, limit_m: float) -> "BuildingScores":
        from scipy.ndimage import map_coordinates

        truth = grid.inside
        pred = predicted < 0
        iou = float((truth & pred).sum() / max((truth | pred).sum(), 1))
        correct = float((truth == pred).mean())
        verts, _ = surface(predicted, grid.res)
        if len(verts):
            idx = (verts / grid.res - 0.5).T  # voxel index coordinates (z, y, x)
            d = np.abs(map_coordinates(grid.sdf, idx, order=1, mode="nearest"))
            med, p90 = float(np.median(d)), float(np.quantile(d, 0.9))
        else:
            med = p90 = float("nan")
        rmse = float(np.sqrt(np.mean((np.clip(grid.sdf, -limit_m, limit_m) - predicted) ** 2)))
        return cls(iou, correct, med, p90, rmse)


def surface(field: np.ndarray, res: float, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The 0-surface as triangles (Marching Cubes): vertices (n, 3) in metres from the grid corner as (z, y, x)."""
    from skimage.measure import marching_cubes

    if field.min() >= 0 or field.max() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(res, res, res), step_size=step)
    return verts + res / 2, faces


def mesh_enh(grid: BuildingGrid, verts: np.ndarray) -> np.ndarray:
    """Marching-cubes vertices (z, y, x in metres from the first voxel corner) → E, N, H."""
    h = grid.res / 2
    east = grid.east[0] - h + verts[:, 2]
    north = grid.north[0] + h - verts[:, 1]  # rows run north → south
    height = grid.height[0] - h + verts[:, 0]
    return np.stack([east, north, height], -1)
