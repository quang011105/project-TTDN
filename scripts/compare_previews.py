from pathlib import Path
from PIL import Image
import numpy as np
import os


def stats(img: Image.Image):
    arr = np.array(img).astype(np.float64)
    if arr.ndim == 2:
        chans = [arr]
    else:
        chans = [arr[..., i] for i in range(arr.shape[2])]
    return {
        'size': img.size,
        'mode': img.mode,
        'channels': [
            {'min': float(np.nanmin(c)), 'max': float(np.nanmax(c)), 'mean': float(np.nanmean(c))}
            for c in chans
        ]
    }


def compare(path1: Path, path2: Path):
    i1 = Image.open(path1)
    i2 = Image.open(path2)
    s1 = stats(i1)
    s2 = stats(i2)
    print(f"File1: {path1} size={os.path.getsize(path1)} bytes", s1)
    print(f"File2: {path2} size={os.path.getsize(path2)} bytes", s2)

    # Resize to common shape for diff
    w = min(i1.width, i2.width)
    h = min(i1.height, i2.height)
    i1r = i1.resize((w, h)).convert('RGB')
    i2r = i2.resize((w, h)).convert('RGB')
    a1 = np.array(i1r).astype(np.float64)
    a2 = np.array(i2r).astype(np.float64)
    diff = np.abs(a1 - a2)
    print('Pixel diff stats (per-channel):')
    for ch in range(3):
        d = diff[..., ch]
        print(f' ch{ch}: mean={d.mean():.3f}, max={d.max():.3f}')


if __name__ == '__main__':
    p1 = Path('data/raw/preview_giao_thuy.png')
    p2 = Path('data/raw/preview_giao_thuy_rgb.png')
    if not p1.exists() or not p2.exists():
        print('One or both preview files missing:', p1.exists(), p2.exists())
    else:
        compare(p1, p2)
