"""Bước 1.2 — Ánh xạ nhãn ESA WorldCover v200 sang 7 lớp Việt Nam (0–6).

Module này chuyển đổi raster nhãn ESA WorldCover (11 mã toàn cầu) thành mask
nguyên uint8 chuẩn 7 lớp phục vụ mô hình segmentation ở Giai đoạn 2.
Sử dụng Lookup Table (LUT) 256 phần tử của NumPy để đảm bảo xử lý vectorized O(1)
trên từng pixel mà không dùng vòng lặp Python.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from PIL import Image

from .config import (
    CLASS_COLORS,
    CLASS_NAMES,
    ESA_TO_LOCAL_CLASS,
    GEE_SCALE_M,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

VALID_CLASSES = set(range(7))  # {0, 1, 2, 3, 4, 5, 6}


def build_lookup_table(
    mapping_dict: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Xây dựng bảng tra cứu (Lookup Table) 256 phần tử kiểu uint8.

    Mỗi chỉ số của mảng đại diện cho mã gốc ESA (0–255), giá trị tại chỉ số đó
    là lớp đích (0–6). Các mã không có trong từ điển mặc định về 0 (Background).

    Args:
        mapping_dict: Từ điển ánh xạ {mã_gốc: mã_đích}. Mặc định dùng ESA_TO_LOCAL_CLASS.

    Returns:
        Mảng 1D numpy shape (256,), dtype uint8.
    """
    mapping = mapping_dict if mapping_dict is not None else ESA_TO_LOCAL_CLASS
    lut = np.zeros(256, dtype=np.uint8)
    for src_code, dst_code in mapping.items():
        if not (0 <= src_code <= 255):
            raise ValueError(f"Mã ESA nguồn {src_code} không hợp lệ (phải từ 0 đến 255).")
        if dst_code not in VALID_CLASSES:
            raise ValueError(f"Lớp đích {dst_code} không hợp lệ (phải từ 0 đến 6).")
        lut[src_code] = np.uint8(dst_code)
    return lut


def remap_array(
    wc_array: np.ndarray,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Ánh xạ mảng WorldCover sang ma trận 7 lớp (0–6) bằng LUT vectorized.

    Args:
        wc_array: Mảng NumPy chứa mã ESA WorldCover gốc (kiểu nguyên / uint8).
        lut: Lookup Table 256 phần tử. Nếu None sẽ dùng bảng chuẩn.

    Returns:
        Mảng NumPy kiểu uint8 cùng shape với wc_array, giá trị chỉ thuộc {0..6}.

    Raises:
        ValueError: Nếu mảng đầu vào chứa giá trị ngoài khoảng [0, 255] hoặc kết
            quả chứa lớp không hợp lệ.
    """
    if lut is None:
        lut = build_lookup_table()

    # Đảm bảo mảng đầu vào có thể index vào LUT 256 phần tử
    if not np.issubdtype(wc_array.dtype, np.integer):
        wc_array = wc_array.astype(np.int64)

    min_val, max_val = int(np.min(wc_array)), int(np.max(wc_array))
    if min_val < 0 or max_val > 255:
        raise ValueError(
            f"Giá trị pixel đầu vào ngoài khoảng [0, 255]: min={min_val}, max={max_val}."
        )

    remapped = lut[wc_array.astype(np.uint8)]
    unique_classes = set(np.unique(remapped))
    if not unique_classes.issubset(VALID_CLASSES):
        raise ValueError(
            f"Phát hiện lớp không hợp lệ sau khi remap: {unique_classes - VALID_CLASSES}"
        )
    return remapped


def compute_class_histogram(arr: np.ndarray) -> dict[int, int]:
    """Tính số lượng pixel cho mỗi lớp 0–6.

    Args:
        arr: Mảng NumPy mask 7 lớp.

    Returns:
        Dict {class_id: pixel_count} cho tất cả các lớp 0..6.
    """
    unique, counts = np.unique(arr, return_counts=True)
    stats: dict[int, int] = {c: 0 for c in range(7)}
    for u, cnt in zip(unique, counts):
        stats[int(u)] = int(cnt)
    return stats


def format_histogram_report(
    hist_before: dict[int, int],
    hist_after: dict[int, int],
    scale_m: float = GEE_SCALE_M,
) -> str:
    """Tạo báo cáo dạng bảng chi tiết về phân bố pixel và diện tích trước/sau remap.

    Args:
        hist_before: Thống kê số pixel các mã ESA gốc.
        hist_after: Thống kê số pixel 7 lớp sau remap.
        scale_m: Độ phân giải không gian của pixel (mặc định 10 mét).

    Returns:
        Chuỗi văn bản báo cáo định dạng bảng.
    """
    pixel_area_ha = (scale_m * scale_m) / 10_000.0  # 100 m² = 0.01 ha
    total_pixels = sum(hist_after.values())

    lines = [
        "=" * 72,
        " BÁO CÁO THỐNG KÊ ÁNH XẠ NHÃN (ESA WORLDCOVER -> 7 LỚP VIỆT NAM)",
        "=" * 72,
        f"Tổng số pixel: {total_pixels:,} | Độ phân giải: {scale_m:.1f}m | Diện tích: {total_pixels * pixel_area_ha:,.2f} ha",
        "-" * 72,
        f"{'Mã':<4} | {'Tên Lớp':<30} | {'Số Pixel':>12} | {'Tỷ Lệ (%)':>10} | {'Diện Tích (ha)':>12}",
        "-" * 72,
    ]

    for class_id in range(7):
        count = hist_after.get(class_id, 0)
        pct = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0
        area_ha = count * pixel_area_ha
        name = CLASS_NAMES.get(class_id, "Không xác định")
        lines.append(f"{class_id:<4} | {name:<30} | {count:>12,} | {pct:>9.2f}% | {area_ha:>12.2f}")

    lines.append("-" * 72)
    lines.append("Mã ESA WorldCover gốc ghi nhận:")
    for code, count in sorted(hist_before.items()):
        if count > 0:
            pct = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0
            lines.append(f"  - Mã {code:<3}: {count:>10,} pixel ({pct:>6.2f}%)")
    lines.append("=" * 72)
    return "\n".join(lines)


def save_colored_preview(
    mask: np.ndarray,
    out_png: str | Path,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> Path:
    """Lưu ảnh preview PNG có màu theo bảng màu chuẩn để kiểm tra trực quan.

    Args:
        mask: Ma trận 2D chứa các giá trị lớp 0–6.
        out_png: Đường dẫn file ảnh PNG đầu ra.
        palette: Bảng màu {class_id: (R, G, B)}. Mặc định dùng CLASS_COLORS.

    Returns:
        Đường dẫn file PNG đã lưu.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = palette if palette is not None else CLASS_COLORS

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        rgb[mask == class_id] = color

    img = Image.fromarray(rgb, mode="RGB")
    img.save(out_path)
    logger.info("Saved colored preview to %s", out_path)
    return out_path


def remap_worldcover_raster(
    input_path: str | Path,
    output_path: str | Path,
    mapping_dict: Mapping[int, int] | None = None,
    write_colormap: bool = True,
    preview_path: str | Path | None = None,
) -> tuple[Path, dict[int, int]]:
    """Đọc GeoTIFF WorldCover gốc, ánh xạ về 7 lớp và lưu thành GeoTIFF uint8.

    Args:
        input_path: Đường dẫn GeoTIFF WorldCover v200 gốc (10 m, UTM).
        output_path: Đường dẫn file GeoTIFF mask 7 lớp đầu ra.
        mapping_dict: Bảng ánh xạ tùy chọn. Mặc định dùng ESA_TO_LOCAL_CLASS.
        write_colormap: Nếu True, nhúng bảng màu chuẩn vào metadata GeoTIFF.
        preview_path: Đường dẫn tùy chọn để xuất ảnh preview PNG.

    Returns:
        Tuple (đường dẫn file mask đã ghi, dict thống kê số pixel từng lớp).
    """
    ensure_dirs()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file raster đầu vào: {input_path}")

    lut = build_lookup_table(mapping_dict)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        raw_data = src.read(1)

        # Thống kê trước khi remap
        u_raw, c_raw = np.unique(raw_data, return_counts=True)
        hist_before = {int(u): int(c) for u, c in zip(u_raw, c_raw)}

        # Ánh xạ vectorized
        mask7 = remap_array(raw_data, lut)

        # Cập nhật profile cho file mask
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,  # Background có giá trị 0
            compress="deflate",
        )
        if write_colormap:
            profile["photometric"] = "palette"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask7, 1)

            if write_colormap:
                # Nhúng colormap vào band 1 để QGIS tự động render đúng màu
                cmap = {
                    cid: (r, g, b, 255)
                    for cid, (r, g, b) in CLASS_COLORS.items()
                }
                # Điền đầy đủ 256 giá trị cho colormap
                full_cmap: dict[int, tuple[int, int, int, int]] = {
                    i: cmap.get(i, (0, 0, 0, 0)) for i in range(256)
                }
                dst.write_colormap(1, full_cmap)

    hist_after = compute_class_histogram(mask7)
    report = format_histogram_report(hist_before, hist_after)
    logger.info("\n%s", report)

    if preview_path is not None:
        save_colored_preview(mask7, preview_path)

    return output_path, hist_after


def main() -> None:
    """CLI thực thi ánh xạ nhãn."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Ánh xạ nhãn ESA WorldCover v200 sang mask 7 lớp Việt Nam (0–6)"
    )
    parser.add_argument("input", help="Đường dẫn file GeoTIFF WorldCover gốc (raw)")
    parser.add_argument("output", help="Đường dẫn file GeoTIFF mask 7 lớp đích (processed)")
    parser.add_argument(
        "--preview",
        default=None,
        help="Đường dẫn tùy chọn để xuất ảnh màu PNG preview (ví dụ: preview_mask7.png)",
    )
    args = parser.parse_args()

    out_file, hist = remap_worldcover_raster(
        args.input,
        args.output,
        preview_path=args.preview,
    )
    print(f"\n[OK] Đã lưu mask 7 lớp tại: {out_file}")


if __name__ == "__main__":
    main()
