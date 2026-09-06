#!/usr/bin/env python3
"""Create a larger RGB preview and print diagnostics to debug blank/striped output."""
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
    out = Path("data/raw/preview_giao_thuy_rgb_debug.png")
    if not fp.exists():
        print("Input not found:", fp)
        return
    with rasterio.open(fp) as src:
        # use smaller downscale to get clearer image
        downscale = 4
        oh = max(1, src.height // downscale)
        ow = max(1, src.width // downscale)
        print(f"Source size: {src.width}x{src.height}, preview: {ow}x{oh}")
        # Composite band order is B2,B3,B4[,B8] → RGB = 3,2,1
        def read(b):
            try:
                return src.read(b, out_shape=(oh, ow)).astype(float)
            except Exception:
                return None

        r = read(3)
        g = read(2)
        b = read(1)

        if r is None or g is None or b is None:
            print("Missing RGB bands 3/2/1, falling back to band 1 for all channels")
            band1 = read(1)
            r = g = b = band1

        # report NaN counts and save previews
        for name, ch in [("R", r), ("G", g), ("B", b)]:
            if ch is None:
                print(f"{name}: missing")
                continue
            n_nan = np.isnan(ch).sum()
            print(f"{name}: shape={ch.shape}, nan_count={int(n_nan)}, nan_frac={n_nan/ch.size:.6f}")

        # Save RGB if possible, otherwise grayscale
        if r is not None and g is not None and b is not None:
            r8 = norm(r)
            g8 = norm(g)
            b8 = norm(b)
            rgb = np.dstack([r8, g8, b8])
            Image.fromarray(rgb).save(out)
            print("Saved debug RGB preview:", out)
        else:
            # fallback grayscale from band1
            ch = r if r is not None else (g if g is not None else b)
            if ch is None:
                ch = read(1)
            ch8 = norm(ch)
            gray_out = out.with_name(out.stem + "_gray.png")
            Image.fromarray(ch8).convert("L").save(gray_out)
            print("Saved debug grayscale preview:", gray_out)


if __name__ == '__main__':
    main()
