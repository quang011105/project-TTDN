#!/usr/bin/env python3
"""Generate an RGB preview (B4, B3, B2) from the Sentinel-2 composite.

Run on Windows (with .venv activated):
  python scripts/preview_rgb.py
"""
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image


def norm(ch: np.ndarray) -> np.ndarray:
    p1 = np.nanpercentile(ch, 2)
    p99 = np.nanpercentile(ch, 98)
    ch = (ch - p1) / (p99 - p1) if p99 > p1 else ch
    ch = np.clip(ch, 0.0, 1.0)
    return (ch * 255).astype(np.uint8)


def main():
    fp = Path("data/raw/giao_thuy_202509_202510_composite.tif")
    out = Path("data/raw/preview_giao_thuy_rgb.png")
    if not fp.exists():
        print("Input file not found:", fp)
        return
    with rasterio.open(fp) as src:
        oh = max(1, src.height // 8)
        ow = max(1, src.width // 8)
        # Composite band order is B2,B3,B4,B8 → RGB = bands 3,2,1 (R=B4, G=B3, B=B2).
        if src.count >= 3:
            try:
                r = src.read(3, out_shape=(oh, ow)).astype(float)
                g = src.read(2, out_shape=(oh, ow)).astype(float)
                b = src.read(1, out_shape=(oh, ow)).astype(float)
                if src.nodata is not None:
                    r[r == src.nodata] = np.nan
                    g[g == src.nodata] = np.nan
                    b[b == src.nodata] = np.nan
                rgb = np.dstack([norm(r), norm(g), norm(b)])
                Image.fromarray(rgb).save(out)
                print("Saved RGB preview to", out)
            except Exception as e:
                print("RGB preview failed, creating grayscale:", e)
                ch = src.read(1, out_shape=(oh, ow)).astype(float)
                if src.nodata is not None:
                    ch[ch == src.nodata] = np.nan
                gray = norm(ch)
                Image.fromarray(gray).convert("L").save(out.with_name("preview_giao_thuy_gray.png"))
                print("Saved grayscale preview to", out.with_name("preview_giao_thuy_gray.png"))
        else:
            ch = src.read(1, out_shape=(oh, ow)).astype(float)
            if src.nodata is not None:
                ch[ch == src.nodata] = np.nan
            gray = norm(ch)
            Image.fromarray(gray).convert("L").save(out.with_name("preview_giao_thuy_gray.png"))
            print("Saved grayscale preview to", out.with_name("preview_giao_thuy_gray.png"))


if __name__ == "__main__":
    main()
