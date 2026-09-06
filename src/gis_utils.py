"""Spatial I/O helpers: grid alignment checks for paired rasters."""
from __future__ import annotations

from pathlib import Path

import rasterio


def assert_rasters_aligned(path_a: str | Path, path_b: str | Path) -> None:
    """Raise if two GeoTIFFs do not share CRS, transform, and shape.

    Args:
        path_a: First raster (e.g. Sentinel-2 composite).
        path_b: Second raster (e.g. WorldCover).

    Raises:
        RuntimeError: When grids differ.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        same = (
            a.crs == b.crs
            and a.transform == b.transform
            and a.width == b.width
            and a.height == b.height
        )
        if not same:
            raise RuntimeError(
                "Rasters are not on the same grid:\n"
                f"  {path_a.name}: CRS={a.crs} size={a.width}x{a.height} "
                f"transform={a.transform}\n"
                f"  {path_b.name}: CRS={b.crs} size={b.width}x{b.height} "
                f"transform={b.transform}"
            )
