"""Bước 1.1 — paired Sentinel-2 composite and ESA WorldCover v200 export.

Downloads (or Drive-exports) two GeoTIFFs on the same UTM 10 m grid:
Sentinel-2 L2A cloud-free median and WorldCover Map (ESA codes, not remapped).
"""
from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, List

import ee
import requests

from .config import (
    GEE_CLOUD_PCT_MAX,
    GEE_CRS,
    GEE_DEFAULT_BANDS,
    GEE_MAX_PIXELS,
    GEE_SCALE_M,
    S2_COLLECTION,
    SCL_EXCLUDE,
    WORLDCOVER_BAND,
    WORLDCOVER_COLLECTION,
    ensure_dirs,
)
from .gis_utils import assert_rasters_aligned

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_S = 300
_STRIP_STEM_SUFFIXES = ("_s2", "_worldcover", "_composite")


def initialize_ee(project: str | None = None) -> None:
    """Initialize the Earth Engine client, authenticating if needed.

    Args:
        project: Optional GCP project id passed to ``ee.Initialize``.
    """
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.debug("Earth Engine initialized")
    except Exception:
        logger.info("Authenticating Earth Engine interactively...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def _load_geojson(geojson_path: Path) -> dict[str, Any]:
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    with open(geojson_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def geometry_from_geojson(gj: dict[str, Any]) -> ee.Geometry:
    """Build an EE geometry; FeatureCollection features are unioned.

    Args:
        gj: Parsed GeoJSON object.

    Returns:
        Earth Engine geometry covering the full AOI.
    """
    gtype = gj.get("type")
    if gtype == "FeatureCollection":
        return ee.FeatureCollection(gj).geometry()
    if gtype == "Feature":
        return ee.Geometry(gj["geometry"])
    return ee.Geometry(gj)


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask L2A pixels using SCL (shadow, cloud, cirrus, nodata, saturated)."""
    scl = image.select("SCL")
    mask = scl.mask()
    for code in SCL_EXCLUDE:
        mask = mask.And(scl.neq(code))
    return image.updateMask(mask)


def _s2_median_composite(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    bands: List[str],
    cloud_pct_max: int = GEE_CLOUD_PCT_MAX,
) -> ee.Image:
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
        .map(_mask_s2_clouds)
    )
    n_images = int(collection.size().getInfo() or 0)
    logger.info("Sentinel-2 scenes after filters: %d", n_images)
    if n_images == 0:
        raise RuntimeError(
            f"No Sentinel-2 images for {start_date}..{end_date} "
            f"(CLOUDY_PIXEL_PERCENTAGE < {cloud_pct_max}). "
            "Widen the date window (dry season, 3–6 months)."
        )
    return collection.median().select(bands).clip(geom).toUint16()


def _worldcover_map(geom: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection(WORLDCOVER_COLLECTION)
        .filterBounds(geom)
        .mosaic()
        .select(WORLDCOVER_BAND)
        .clip(geom)
    )


def _aligned_pair(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    bands: List[str],
    crs: str,
    scale: int,
) -> tuple[ee.Image, ee.Image]:
    """Reproject S2 and WorldCover onto one UTM 10 m projection (nearest for labels)."""
    proj = ee.Projection(crs).atScale(scale)
    s2 = _s2_median_composite(geom, start_date, end_date, bands).reproject(proj)
    # reproject() defaults to nearest-neighbor (resample() only allows bilinear/bicubic).
    worldcover = _worldcover_map(geom).reproject(crs=s2.projection())
    return s2, worldcover


def _write_bytes_as_geotiff(payload: bytes, out_path: Path) -> None:
    """Save a GeoTIFF body, unzipping GEE ZIPPED_GEO_TIFF if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                raise RuntimeError("Download ZIP contained no GeoTIFF")
            if len(tifs) > 1:
                logger.warning("ZIP has %d TIFFs; using first: %s", len(tifs), tifs[0])
            out_path.write_bytes(zf.read(tifs[0]))
        return
    out_path.write_bytes(payload)


def _download_ee_image(
    image: ee.Image,
    out_path: Path,
    geom: ee.Geometry,
    crs: str,
    scale: int,
    max_attempts: int = 3,
) -> None:
    payload = {
        "scale": scale,
        "crs": crs,
        "region": geom,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            url = image.getDownloadURL(payload)
            logger.info("Download URL acquired for %s (attempt %d)", out_path.name, attempt)
            with requests.get(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                response.raise_for_status()
                _write_bytes_as_geotiff(response.content, out_path)
            logger.info("Saved %s", out_path)
            return
        except Exception as exc:  # pragma: no cover - network/ee issues
            last_err = exc
            logger.warning("Attempt %d failed for %s: %s", attempt, out_path.name, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Failed to download {out_path.name}. "
        "If the AOI is large, use --export-drive instead of getDownloadURL."
    ) from last_err


def _output_paths(out: str | Path) -> tuple[Path, Path]:
    """Map a prefix or legacy .tif path to ``*_s2.tif`` and ``*_worldcover.tif``."""
    out = Path(out)
    if out.suffix.lower() in {".tif", ".tiff"}:
        stem = out.stem
        for suffix in _STRIP_STEM_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        base = out.parent / stem
    else:
        base = out
    return Path(f"{base}_s2.tif"), Path(f"{base}_worldcover.tif")


def download_aoi_pair(
    geojson_path: str | Path,
    out_prefix: str | Path,
    start_date: str,
    end_date: str,
    bands: List[str] | None = None,
    scale: int = GEE_SCALE_M,
    crs: str = GEE_CRS,
    max_attempts: int = 3,
    project: str | None = None,
) -> tuple[Path, Path]:
    """Download aligned Sentinel-2 and WorldCover GeoTIFFs for one commune AOI.

    Args:
        geojson_path: Polygon / Feature / FeatureCollection path.
        out_prefix: File prefix or legacy ``.tif`` path.
        start_date: Inclusive start ``YYYY-MM-DD``.
        end_date: Exclusive-or-filter end ``YYYY-MM-DD`` (EE filterDate).
        bands: Sentinel-2 bands (default B2, B3, B4, B8).
        scale: Pixel size in metres.
        crs: Destination CRS (UTM Zone 48N by default).
        max_attempts: Retries per file.
        project: Optional GCP project id.

    Returns:
        Paths to the S2 composite and WorldCover rasters.
    """
    ensure_dirs()
    bands = bands or GEE_DEFAULT_BANDS
    geojson_path = Path(geojson_path)
    s2_path, wc_path = _output_paths(out_prefix)

    initialize_ee(project=project)
    geom = geometry_from_geojson(_load_geojson(geojson_path))
    s2, worldcover = _aligned_pair(geom, start_date, end_date, bands, crs, scale)

    _download_ee_image(s2, s2_path, geom, crs, scale, max_attempts=max_attempts)
    _download_ee_image(worldcover, wc_path, geom, crs, scale, max_attempts=max_attempts)
    assert_rasters_aligned(s2_path, wc_path)
    return s2_path, wc_path


def download_sentinel_composite(
    geojson_path: str | Path,
    out_path: str | Path,
    start_date: str,
    end_date: str,
    bands: List[str] | None = None,
    scale: int = GEE_SCALE_M,
    max_attempts: int = 3,
    project: str | None = None,
    crs: str = GEE_CRS,
) -> None:
    """Download only the Sentinel-2 composite (legacy helper)."""
    ensure_dirs()
    bands = bands or GEE_DEFAULT_BANDS
    out_path = Path(out_path)
    initialize_ee(project=project)
    geom = geometry_from_geojson(_load_geojson(Path(geojson_path)))
    s2, _ = _aligned_pair(geom, start_date, end_date, bands, crs, scale)
    _download_ee_image(s2, out_path, geom, crs, scale, max_attempts=max_attempts)


def _start_drive_export(
    image: ee.Image,
    file_prefix: str,
    geom: ee.Geometry,
    folder: str | None,
    crs: str,
    scale: int,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix[:100],
        folder=folder,
        fileNamePrefix=file_prefix,
        region=geom,
        scale=scale,
        crs=crs,
        maxPixels=GEE_MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    logger.info("Started Drive export %s (id=%s)", file_prefix, task.id)
    return task


def _wait_for_task(task: ee.batch.Task, poll_interval: int) -> dict[str, Any]:
    while True:
        status = task.status()
        state = status.get("state")
        logger.info("Task %s status: %s", task.id, state)
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            return status
        time.sleep(poll_interval)


def export_aoi_pair_to_drive(
    geojson_path: str | Path,
    drive_prefix: str,
    start_date: str,
    end_date: str,
    bands: List[str] | None = None,
    scale: int = GEE_SCALE_M,
    crs: str = GEE_CRS,
    folder: str | None = None,
    project: str | None = None,
    wait: bool = True,
    poll_interval: int = 10,
) -> dict[str, Any]:
    """Export aligned S2 and WorldCover rasters to Google Drive.

    Args:
        geojson_path: AOI GeoJSON path.
        drive_prefix: Drive filename prefix (writes ``{prefix}_s2`` and ``{prefix}_worldcover``).
        start_date: Filter start date.
        end_date: Filter end date.
        bands: Sentinel-2 bands.
        scale: Export scale in metres.
        crs: Export CRS.
        folder: Optional Drive folder.
        project: Optional GCP project id.
        wait: Block until both tasks finish.
        poll_interval: Seconds between status polls.

    Returns:
        Task ids, or full status dicts when ``wait`` is True.
    """
    ensure_dirs()
    bands = bands or GEE_DEFAULT_BANDS
    initialize_ee(project=project)
    geom = geometry_from_geojson(_load_geojson(Path(geojson_path)))
    s2, worldcover = _aligned_pair(geom, start_date, end_date, bands, crs, scale)

    for suffix in _STRIP_STEM_SUFFIXES:
        if drive_prefix.endswith(suffix):
            drive_prefix = drive_prefix[: -len(suffix)]
            break

    t_s2 = _start_drive_export(s2, f"{drive_prefix}_s2", geom, folder, crs, scale)
    t_wc = _start_drive_export(
        worldcover, f"{drive_prefix}_worldcover", geom, folder, crs, scale
    )
    if not wait:
        return {"s2_task_id": t_s2.id, "worldcover_task_id": t_wc.id}
    return {
        "s2": _wait_for_task(t_s2, poll_interval),
        "worldcover": _wait_for_task(t_wc, poll_interval),
    }


def export_sentinel_composite_to_drive(
    geojson_path: str | Path,
    drive_filename: str,
    start_date: str,
    end_date: str,
    bands: List[str] | None = None,
    scale: int = GEE_SCALE_M,
    folder: str | None = None,
    project: str | None = None,
    wait: bool = True,
    poll_interval: int = 10,
    crs: str = GEE_CRS,
) -> dict[str, Any]:
    """Export only the Sentinel-2 composite to Drive (legacy helper)."""
    ensure_dirs()
    bands = bands or GEE_DEFAULT_BANDS
    initialize_ee(project=project)
    geom = geometry_from_geojson(_load_geojson(Path(geojson_path)))
    s2, _ = _aligned_pair(geom, start_date, end_date, bands, crs, scale)
    task = _start_drive_export(s2, drive_filename, geom, folder, crs, scale)
    if not wait:
        return {"task_id": task.id}
    return _wait_for_task(task, poll_interval)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Download aligned Sentinel-2 + ESA WorldCover v200 for an AOI"
    )
    parser.add_argument("geojson", help="Path to AOI GeoJSON")
    parser.add_argument(
        "out",
        help="Output prefix or .tif path (writes {prefix}_s2.tif and {prefix}_worldcover.tif)",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--bands", nargs="+", help="Bands to select (e.g. B2 B3 B4 B8)")
    parser.add_argument("--crs", default=GEE_CRS, help=f"Destination CRS (default {GEE_CRS})")
    parser.add_argument(
        "--scale", type=int, default=GEE_SCALE_M, help="Pixel size in metres"
    )
    parser.add_argument(
        "--export-drive",
        action="store_true",
        help="Export both rasters to Google Drive instead of direct download",
    )
    parser.add_argument("--drive-folder", default=None, help="Drive folder name")
    parser.add_argument("--project", default=None, help="GCP project id for EE Initialize")
    parser.add_argument(
        "--no-wait", action="store_true", help="Do not wait for Drive export tasks"
    )
    args = parser.parse_args()

    if args.export_drive:
        drive_name = Path(args.out).stem if Path(args.out).suffix else Path(args.out).name
        status = export_aoi_pair_to_drive(
            args.geojson,
            drive_name,
            args.start,
            args.end,
            bands=args.bands,
            scale=args.scale,
            crs=args.crs,
            folder=args.drive_folder,
            project=args.project,
            wait=not args.no_wait,
        )
        print("Drive export task status:", status)
    else:
        s2_path, wc_path = download_aoi_pair(
            args.geojson,
            args.out,
            args.start,
            args.end,
            bands=args.bands,
            scale=args.scale,
            crs=args.crs,
            project=args.project,
        )
        print("Saved:", s2_path)
        print("Saved:", wc_path)
