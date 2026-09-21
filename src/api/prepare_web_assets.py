"""
Script tạo các lớp phủ bản đồ (Web Map Overlays) cho Leaflet WebGIS.
Tạo ảnh PNG RGBA chuẩn tọa độ WGS84 (EPSG:4326) để khớp 100% với bản đồ Leaflet:
1. Bản đồ phân loại AI LULC (nền trong suốt, 6 màu chuẩn GIS tương phản cao).
2. Lưu GeoTIFF WGS84 (outputs/maps/giao_thuy_lulc_wgs84.tif) phục vụ tra cứu pixel chuẩn xác.
3. Ảnh vệ tinh tự nhiên Sentinel-2 True Color (RGB).
4. Ảnh vệ tinh hồng ngoại giả Sentinel-2 False Color (CIR).
Xuất thông số bounds WGS84 phục vụ Leaflet L.imageOverlay.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from src.config import CLASS_COLORS, CLASS_NAMES


def prepare_web_assets(
    lulc_tif: Path = Path("outputs/maps/giao_thuy_lulc_prediction.tif"),
    s2_tif: Path = Path("data/raw/giao_thuy_expanded_s2.tif"),
    output_dir: Path = Path("web/assets"),
    wgs84_tif_out: Path = Path("outputs/maps/giao_thuy_lulc_wgs84.tif"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    wgs84_tif_out.parent.mkdir(parents=True, exist_ok=True)

    dst_crs = "EPSG:4326"

    # 1. Đọc raster LULC và Reproject sang WGS84
    with rasterio.open(lulc_tif) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        lulc_wgs84 = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=lulc_wgs84,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        # Tính chính xác tọa độ góc cho Leaflet [ [south, west], [north, east] ]
        north = float(dst_transform[5])
        west = float(dst_transform[2])
        south = float(dst_transform[5] + dst_transform[4] * dst_height)
        east = float(dst_transform[2] + dst_transform[0] * dst_width)

        leaflet_bounds = [
            [round(south, 7), round(west, 7)],
            [round(north, 7), round(east, 7)]
        ]
        center = [
            round((south + north) / 2.0, 7),
            round((west + east) / 2.0, 7)
        ]

        # Xuất GeoTIFF WGS84 để phục vụ API tra cứu pixel chuẩn 100%
        wgs84_meta = src.meta.copy()
        wgs84_meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": 1,
            "dtype": "uint8"
        })
        with rasterio.open(wgs84_tif_out, "w", **wgs84_meta) as dst:
            dst.write(lulc_wgs84, 1)
        print(f"[*] Đã xuất GeoTIFF WGS84: {wgs84_tif_out}")

    # 2. Tạo ảnh LULC RGBA trong suốt từ dữ liệu reprojected
    rgba_lulc = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
    for c_id, rgb in CLASS_COLORS.items():
        if c_id == 0:
            continue  # Background trong suốt hoàn toàn (Alpha = 0)
        mask = (lulc_wgs84 == c_id)
        rgba_lulc[mask, 0] = rgb[0]
        rgba_lulc[mask, 1] = rgb[1]
        rgba_lulc[mask, 2] = rgb[2]
        rgba_lulc[mask, 3] = 230  # Alpha 90% cho màu sắc rõ nét, nổi bật, phân biệt rõ

    lulc_png_path = output_dir / "lulc_overlay.png"
    Image.fromarray(rgba_lulc).save(lulc_png_path, format="PNG", optimize=True)
    print(f"[*] Đã xuất LULC Web Overlay (WGS84): {lulc_png_path}")

    # 3. Reproject Sentinel-2 sang cùng WGS84 Grid để khớp hoàn toàn
    with rasterio.open(s2_tif) as src:
        def reproject_band(b_idx):
            dest = np.zeros((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, b_idx),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            return dest

        b2_wgs = reproject_band(1)
        b3_wgs = reproject_band(2)
        b4_wgs = reproject_band(3)
        b8_wgs = reproject_band(4)

    def stretch_to_8bit(band, p_low=2, p_high=98):
        valid = band[band > 0]
        v_min = np.percentile(valid, p_low) if len(valid) > 0 else 0
        v_max = np.percentile(valid, p_high) if len(valid) > 0 else 3000
        stretched = np.clip((band - v_min) / (v_max - v_min + 1e-6) * 255.0, 0, 255)
        return stretched.astype(np.uint8)

    r_true = stretch_to_8bit(b4_wgs)
    g_true = stretch_to_8bit(b3_wgs)
    b_true = stretch_to_8bit(b2_wgs)

    # Nodata mask
    nodata_mask = (b2_wgs == 0) & (b3_wgs == 0) & (b4_wgs == 0)
    alpha_s2 = np.where(nodata_mask, 0, 255).astype(np.uint8)

    # Sentinel-2 RGB True Color
    rgb_s2 = np.stack([r_true, g_true, b_true, alpha_s2], axis=-1)
    rgb_png_path = output_dir / "sentinel2_rgb.png"
    Image.fromarray(rgb_s2).save(rgb_png_path, format="PNG", optimize=True)
    print(f"[*] Đã xuất Sentinel-2 RGB (WGS84): {rgb_png_path}")

    # Sentinel-2 False Color CIR: NIR (B8), Red (B4), Green (B3)
    r_cir = stretch_to_8bit(b8_wgs)
    g_cir = stretch_to_8bit(b4_wgs)
    b_cir = stretch_to_8bit(b3_wgs)
    cir_s2 = np.stack([r_cir, g_cir, b_cir, alpha_s2], axis=-1)
    cir_png_path = output_dir / "sentinel2_cir.png"
    Image.fromarray(cir_s2).save(cir_png_path, format="PNG", optimize=True)
    print(f"[*] Đã xuất Sentinel-2 CIR (WGS84): {cir_png_path}")

    # 4. Xuất metadata JSON cho Leaflet Frontend
    metadata = {
        "leaflet_bounds": leaflet_bounds,
        "center": center,
        "default_zoom": 12,
        "min_zoom": 10,
        "max_zoom": 17,
        "width": dst_width,
        "height": dst_height,
        "layers": {
            "lulc": "/assets/lulc_overlay.png",
            "s2_rgb": "/assets/sentinel2_rgb.png",
            "s2_cir": "/assets/sentinel2_cir.png",
        }
    }

    meta_path = output_dir / "map_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã cập nhật Map Metadata WGS84: {meta_path}")
    return metadata


if __name__ == "__main__":
    prepare_web_assets()
