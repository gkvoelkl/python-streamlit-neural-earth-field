"""Loading and preparing the Earth's elevation grid (ETOPO 2022).

The raw data is a regular grid of elevations in metres.  This grid is
exactly what the neural field is meant to replace -- which is why
``ElevationGrid`` also computes how many bytes it occupies uncompressed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

ETOPO_FILENAME = "ETOPO_2022_v1_60s_N90W180_surface.nc"
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
ETOPO_BYTES = 478_290_125  # size of the file above (456 MB); the server sends no Content-Length
BYTES_PER_SAMPLE = 2  # whole metres fit int16; ETOPO itself stores float32, so this is conservative


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(exist_ok=True)
    return d


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_path() -> Path:
    return data_dir() / ETOPO_FILENAME


def source_available() -> bool:
    p = source_path()
    return p.exists() and p.stat().st_size > 1_000_000


def download_etopo(
    progress: Callable[[int, int | None], None] | None = None,
    url: str = ETOPO_URL,
    target: Path | None = None,
    chunk_size: int = 1 << 20,
    client: httpx.Client | None = None,
) -> Path:
    """Downloads the ETOPO file; ``progress(bytes_done, bytes_total)`` is called per chunk.

    The data first goes to ``<target>.part`` and is only renamed once complete, so an
    interrupted download never looks like a valid file. Raises ``httpx.HTTPError`` or
    ``OSError`` if it fails.
    """
    target = target or source_path()
    partial = target.with_name(target.name + ".part")
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0))
    try:
        with client.stream("GET", url) as response, open(partial, "wb") as out:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            done = 0
            for chunk in response.iter_bytes(chunk_size):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if total is not None and done != total:
            raise OSError(f"download incomplete: {done} of {total} bytes")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
        if own_client:
            client.close()
    return target


@dataclass
class ElevationGrid:
    """A regular lon/lat grid with elevations in metres."""

    lon: np.ndarray  # (nx,)  cell centres in degrees, -180..180
    lat: np.ndarray  # (ny,)  cell centres in degrees, -90..90
    z: np.ndarray  # (ny, nx) elevation in metres, float32
    source: str = "unknown"

    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    @property
    def n_points(self) -> int:
        return int(self.z.size)

    @property
    def raw_bytes(self) -> int:
        """Size of the grid as raw int16 data -- the reference for the compression ratio."""
        return self.n_points * BYTES_PER_SAMPLE

    @property
    def degrees_per_cell(self) -> float:
        return 180.0 / self.ny

    @property
    def km_per_cell(self) -> float:
        return 2 * np.pi * 6371.0 / 360.0 * self.degrees_per_cell

    def lat_weights(self) -> np.ndarray:
        """cos(lat) -- area weight of a grid cell, as in the papers' 'weighted RMSE'."""
        w = np.cos(np.deg2rad(self.lat)).astype(np.float32)
        return np.clip(w, 1e-6, None)

    def cell_weights(self) -> np.ndarray:
        """Area weight per cell, normalised to mean 1. Shape (ny, nx)."""
        w = np.repeat(self.lat_weights()[:, None], self.nx, axis=1)
        return w / w.mean()

    def coarsen(self, max_ny: int) -> "ElevationGrid":
        """Coarse grid for 3D display (the browser doesn't like millions of points)."""
        if self.ny <= max_ny:
            return self
        fy = max(1, self.ny // max_ny)
        fx = max(1, self.nx // (2 * max_ny))
        ny, nx = self.ny // fy, self.nx // fx
        z = self.z[: ny * fy, : nx * fx].reshape(ny, fy, nx, fx).mean(axis=(1, 3))
        lat = self.lat[: ny * fy].reshape(ny, fy).mean(axis=1)
        lon = self.lon[: nx * fx].reshape(nx, fx).mean(axis=1)
        return ElevationGrid(lon=lon, lat=lat, z=z.astype(np.float32), source=self.source)


def thin(lon: np.ndarray, lat: np.ndarray, values: np.ndarray, max_ny: int):
    """Take every n-th point -- for error images, where averaging cancels out the signs."""
    ny = values.shape[0]
    if ny <= max_ny:
        return lon, lat, values
    fy = max(1, ny // max_ny)
    fx = max(1, values.shape[1] // (2 * max_ny))
    return lon[::fx], lat[::fy], values[::fy, ::fx]


def _block_mean_2d(var, fy: int, fx: int, ny: int, nx: int, block_rows: int = 48) -> np.ndarray:
    """Averages fy x fx blocks without loading the whole 233-million-point array."""
    out = np.empty((ny, nx), dtype=np.float32)
    for start in range(0, ny, block_rows):
        stop = min(start + block_rows, ny)
        chunk = np.asarray(var[start * fy : stop * fy, : nx * fx], dtype=np.float32)
        out[start:stop] = chunk.reshape(stop - start, fy, nx, fx).mean(axis=(1, 3))
    return out


def load_etopo(target_ny: int = 360, use_cache: bool = True) -> ElevationGrid:
    """Loads ETOPO 2022 (1 arc-minute) and averages it down to ``target_ny`` latitude rows."""
    if not source_available():
        raise FileNotFoundError(
            f"{source_path()} is missing.\nDownload:\n  curl -L -o {source_path()} {ETOPO_URL}"
        )

    cache = cache_dir() / f"etopo_{target_ny}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return ElevationGrid(
            lon=d["lon"], lat=d["lat"], z=d["z"], source=f"ETOPO 2022 (cache {cache.name})"
        )

    import netCDF4  # imported locally so the app also starts without data

    with netCDF4.Dataset(source_path()) as ds:
        var = ds.variables["z"]
        var.set_auto_mask(False)
        lat_src = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon_src = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        height, width = var.shape

        fy = max(1, height // target_ny)
        fx = max(1, width // (2 * target_ny))
        ny, nx = height // fy, width // fx

        z = _block_mean_2d(var, fy, fx, ny, nx)

    lat = lat_src[: ny * fy].reshape(ny, fy).mean(axis=1).astype(np.float32)
    lon = lon_src[: nx * fx].reshape(nx, fx).mean(axis=1).astype(np.float32)

    grid = ElevationGrid(lon=lon, lat=lat, z=z, source=f"ETOPO 2022 1' → {ny}x{nx}")
    np.savez_compressed(cache, lon=lon, lat=lat, z=z)
    return grid
