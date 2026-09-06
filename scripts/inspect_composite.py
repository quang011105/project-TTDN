#!/usr/bin/env python3
"""Inspect a GeoTIFF composite and produce a small RGB preview.

Usage:
  python scripts/inspect_composite.py data/raw/giao_thuy_202509_202510_composite.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    p1 = np.nanpercentile(ch, 2)
    p99 = np.nanpercentile(ch, 98)
    ch = (ch - p1) / (p99 - p1) if p99 > p1 else ch
    ch = np.clip(ch, 0.0, 1.0)
    return (ch * 255).astype(np.uint8)


def make_preview(src_path: Path, out_png: Path, downscale: int = 8) -> None:
    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        bands = src.count

        print(f"File: {src_path}")
        print("CRS:", src.crs)
        print("Width, Height, Bands:", w, h, bands)
        print("Bounds:", src.bounds)

        # Read a downsampled overview for stats
        oh = max(1, h // downscale)
        ow = max(1, w // downscale)

        for i in range(1, bands + 1):
            arr = src.read(i, out_shape=(oh, ow)).astype(float)
            arr = np.where(arr == src.nodata if src.nodata is not None else False, np.nan, arr)
            print(f"Band {i}: min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}, mean={np.nanmean(arr):.3f}")

        # Build preview: prefer Sentinel visual bands R=B4,G=B3,B=B2; fallback to band1 grayscale
        out_png.parent.mkdir(parents=True, exist_ok=True)
        def save_rgb_preview(r_band, g_band, b_band, path):
            r8 = normalize_channel(r_band)
            g8 = normalize_channel(g_band)
            b8 = normalize_channel(b_band)
            rgb = np.dstack([r8, g8, b8])
            Image.fromarray(rgb).save(path)

        def save_gray_preview(ch, path):
            ch8 = normalize_channel(ch)
            Image.fromarray(ch8).convert("L").save(path)

        # Composite band order is B2,B3,B4[,B8] → RGB = 3,2,1
        if bands >= 3:
            try:
                r = src.read(3, out_shape=(oh, ow)).astype(float)
                g = src.read(2, out_shape=(oh, ow)).astype(float)
                b = src.read(1, out_shape=(oh, ow)).astype(float)
                # mask nodata
                if src.nodata is not None:
                    r[r == src.nodata] = np.nan
                    g[g == src.nodata] = np.nan
                    b[b == src.nodata] = np.nan
                save_rgb_preview(r, g, b, out_png)
                print(f"Saved RGB preview: {out_png}")
            except Exception:
                # fallback to grayscale
                ch = src.read(1, out_shape=(oh, ow)).astype(float)
                if src.nodata is not None:
                    ch[ch == src.nodata] = np.nan
                gray_path = out_png.with_name(out_png.stem + "_gray.png")
                save_gray_preview(ch, gray_path)
                print(f"Saved grayscale preview: {gray_path}")
        else:
            ch = src.read(1, out_shape=(oh, ow)).astype(float)
            if src.nodata is not None:
                ch[ch == src.nodata] = np.nan
            gray_path = out_png.with_name(out_png.stem + "_gray.png")
            save_gray_preview(ch, gray_path)
            print(f"Saved grayscale preview: {gray_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tif", help="Path to GeoTIFF composite")
    parser.add_argument("--preview", help="Output PNG preview path", default="data/raw/preview_giao_thuy.png")
    args = parser.parse_args()

    make_preview(Path(args.tif), Path(args.preview))


if __name__ == "__main__":
    main()
