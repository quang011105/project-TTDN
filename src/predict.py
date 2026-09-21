"""
Module dự đoán phân vùng sử dụng đất (LULC Inference) & Hậu xử lý — Phase 2 (Bước 2.7 & 2.8).

Chức năng:
    1. Bước 2.7: Hậu xử lý ma trận dự đoán:
       - apply_morphological_filter: Làm mịn, triệt tiêu nhiễu muối tiêu (majority / opening & closing).
       - apply_mmu_filter: Lọc ngưỡng đơn vị lập bản đồ tối thiểu (Minimum Mapping Unit - MMU)
         loại bỏ các cụm nhỏ (< 5 pixels / 500 m² @ 10m) và gán lại theo láng giềng gần nhất.
       - postprocess_mask: Pipeline kết hợp linh hoạt cho phép bật/tắt qua cờ tham số.
    2. Bước 2.8: Inference toàn cảnh & xuất bản đồ:
       - predict_sliding_window: Quét cửa sổ trượt (256x256, overlap 64) kèm trọng số 2D Hann
         soft blending loại bỏ triệt để đường ranh giới chắp vá (blocking artifacts).
       - TTA (Test-Time Augmentation): Kết hợp lật ngang / dọc tăng độ ổn định viễn thám.
       - export_lulc_geotiff: Xuất GeoTIFF chuẩn GIS nhúng bảng màu GDAL Colormap và CRS EPSG:32648.
       - export_lulc_png: Xuất bản đồ phân vùng màu trực quan kèm bảng chú giải (Legend).
       - compute_area_statistics: Thống kê diện tích từng lớp (pixel, ha, km², %) ra JSON và CSV.
       - export_comparison_maps: Xuất ảnh so sánh Sentinel-2 CIR vs Ground Truth vs Dự đoán và bản đồ sai khác.

Sử dụng qua dòng lệnh (CLI):
    python -m src.predict --config configs/train_config.yaml
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import scipy.ndimage as ndi
import torch
import yaml

from src.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_ENCODER,
    DEFAULT_ENCODER_WEIGHTS,
    IN_CHANNELS,
    MAP_DIR,
    NUM_CLASSES,
    PATCH_SIZE,
    ensure_dirs,
)
from src.dataset import compute_spectral_indices
from src.model import build_unet

# Cấu hình UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LULC_Predictor")


# ===========================================================================
# 1. BƯỚC 2.7: CÁC THUẬT TOÁN HẬU XỬ LÝ (POST-PROCESSING)
# ===========================================================================

def apply_morphological_filter(
    mask: np.ndarray,
    kernel_size: int = 3,
    method: str = "majority",
    preserve_background: bool = True,
) -> np.ndarray:
    """
    Làm mịn ma trận nhãn phân loại bằng phép toán hình thái học (Morphological Filtering).

    Args:
        mask: Ma trận 2D nhãn nguyên (H, W) với các giá trị 0..6.
        kernel_size: Kích thước cửa sổ lân cận (mặc định: 3x3).
        method: Phương pháp lọc ('majority' — lọc đa số vector hóa siêu nhanh,
                hoặc 'opening_closing' — binary opening & closing từng lớp).
        preserve_background: Giữ nguyên các pixel background gốc nếu cần.

    Returns:
        np.ndarray: Ma trận nhãn sau khi lọc có cùng kích thước và dtype.
    """
    if kernel_size <= 1:
        return mask.copy()

    orig_dtype = mask.dtype
    filtered = mask.copy()

    if method == "majority":
        # Vectorized Majority Filter qua cửa sổ uniform: tính tần suất từng lớp
        class_counts = []
        for c in range(NUM_CLASSES):
            bin_c = (mask == c).astype(np.float32)
            # Dùng uniform_filter để tính mật độ điểm lớp c trong cửa sổ kernel_size x kernel_size
            density = ndi.uniform_filter(bin_c, size=kernel_size, mode="reflect")
            class_counts.append(density)

        stacked_counts = np.stack(class_counts, axis=0)  # (NUM_CLASSES, H, W)
        filtered = np.argmax(stacked_counts, axis=0).astype(orig_dtype)

    elif method == "opening_closing":
        # Áp dụng Binary Opening (triệt tiêu gai nhiễu) sau đó Binary Closing (lấp lỗ rỗng)
        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        cleaned_layers = []

        for c in range(NUM_CLASSES):
            bin_c = (mask == c)
            if not np.any(bin_c):
                cleaned_layers.append(np.zeros_like(bin_c, dtype=np.float32))
                continue
            opened = ndi.binary_opening(bin_c, structure=struct)
            closed = ndi.binary_closing(opened, structure=struct)
            cleaned_layers.append(closed.astype(np.float32))

        stacked = np.stack(cleaned_layers, axis=0)
        max_idx = np.argmax(stacked, axis=0).astype(orig_dtype)
        # Với những điểm không thuộc lớp nào sau opening, giữ nguyên nhãn gốc
        has_class = np.max(stacked, axis=0) > 0
        filtered = np.where(has_class, max_idx, mask)

    else:
        raise ValueError(f"Phương pháp lọc '{method}' không được hỗ trợ. Chọn 'majority' hoặc 'opening_closing'.")

    if preserve_background:
        # Đảm bảo các vùng hoàn toàn là background (0) không bị tràn foreground vào
        filtered = np.where(mask == 0, 0, filtered)

    return filtered.astype(orig_dtype)


def apply_mmu_filter(
    mask: np.ndarray,
    mmu_pixels: int = 5,
    background_val: int = 0,
) -> np.ndarray:
    """
    Bộ lọc Đơn vị Lập bản đồ Tối thiểu (Minimum Mapping Unit - MMU Filter).

    Tìm các vùng liên thông của từng lớp đất 1–6 bằng 8-connectivity. Nếu diện tích
    một vùng nhỏ hơn `mmu_pixels` (< 500 m² @ 10m), vùng đó được coi là nhiễu cô lập
    và được gán lại theo nhãn của vùng láng giềng hợp lệ gần nhất thông qua phép biến đổi
    khoảng cách Euclid (Distance Transform EDT).

    Args:
        mask: Ma trận nhãn 2D (H, W).
        mmu_pixels: Ngưỡng số pixel tối thiểu của một khoanh đất (mặc định: 5 pixels = 500 m²).
        background_val: Giá trị pixel nền/nodata cần bảo toàn.

    Returns:
        np.ndarray: Ma trận nhãn sau khi lọc MMU.
    """
    if mmu_pixels <= 1:
        return mask.copy()

    cleaned_mask = mask.copy()
    to_replace = np.zeros_like(mask, dtype=bool)
    struct = np.ones((3, 3), dtype=int)  # 8-connectivity

    # Duyệt qua từng lớp sử dụng đất tiền cảnh (1 đến 6)
    for c in range(1, NUM_CLASSES):
        class_mask = (mask == c)
        if not np.any(class_mask):
            continue

        labeled, num_features = ndi.label(class_mask, structure=struct)
        if num_features == 0:
            continue

        counts = np.bincount(labeled.ravel())
        # Tìm các nhãn có kích thước < mmu_pixels (bỏ nhãn 0 của labeled)
        small_labels = np.where((counts < mmu_pixels) & (counts > 0))[0]
        small_labels = small_labels[small_labels > 0]

        if len(small_labels) > 0:
            to_replace |= np.isin(labeled, small_labels)

    # Nếu không có cụm nào bị vi phạm MMU
    if not np.any(to_replace):
        return cleaned_mask

    # Các pixel hợp lệ dùng làm điểm tựa gán nhãn
    valid_mask = ~to_replace
    if not np.any(valid_mask):
        return cleaned_mask

    # Sử dụng Distance Transform để tìm tọa độ pixel hợp lệ gần nhất trong O(N)
    indices = ndi.distance_transform_edt(
        ~valid_mask,
        return_distances=False,
        return_indices=True,
    )

    # Gán nhãn của pixel hợp lệ gần nhất cho các pixel bị loại bỏ
    cleaned_mask[to_replace] = mask[indices[0][to_replace], indices[1][to_replace]]

    return cleaned_mask


def postprocess_mask(
    mask: np.ndarray,
    enable_morph: bool = True,
    kernel_size: int = 3,
    morph_method: str = "majority",
    enable_mmu: bool = True,
    mmu_pixels: int = 5,
    preserve_background: bool = True,
) -> np.ndarray:
    """
    Pipeline Hậu xử lý hoàn chỉnh (Bước 2.7).

    Kết hợp lọc hình thái học (Morphological Filter) và lọc quy mô tối thiểu (MMU Filter).
    Mỗi bước có thể bật/tắt độc lập thông qua cấu hình.

    Args:
        mask: Ma trận 2D nhãn nguyên (H, W).
        enable_morph: Bật/tắt lọc hình thái học.
        kernel_size: Kích thước kernel hình thái học.
        morph_method: Phương pháp lọc ('majority' hoặc 'opening_closing').
        enable_mmu: Bật/tắt lọc MMU.
        mmu_pixels: Ngưỡng số pixel tối thiểu.
        preserve_background: Bảo tồn ranh giới nền Nodata.

    Returns:
        np.ndarray: Ma trận nhãn sạch sau hậu xử lý.
    """
    res = mask.copy()

    # Bước 1: Lọc hình thái học để làm mịn và giảm gai nhiễu
    if enable_morph:
        res = apply_morphological_filter(
            res,
            kernel_size=kernel_size,
            method=morph_method,
            preserve_background=preserve_background,
        )

    # Bước 2: Lọc MMU để sáp nhập các khoanh đất nhỏ li ti vào láng giềng
    if enable_mmu:
        res = apply_mmu_filter(
            res,
            mmu_pixels=mmu_pixels,
            background_val=0,
        )

    if preserve_background:
        res = np.where(mask == 0, 0, res)

    return res


# ===========================================================================
# 2. BƯỚC 2.8: INFERENCE TOÀN CẢNH VỚI CỬA SỔ TRƯỢT & SOFT BLENDING
# ===========================================================================

def create_2d_window(patch_size: int = 256, min_weight: float = 0.05) -> np.ndarray:
    """
    Tạo ma trận trọng số 2D Hann mềm (Soft Blending Window) kích thước (patch_size, patch_size).

    Trọng số cao ở tâm và giảm dần đều về 4 cạnh, giúp triệt tiêu hoàn toàn hiệu ứng
    đường ghép chắp vá (seam / blocking artifacts) ở các vùng cửa sổ trượt đè lên nhau.
    """
    h1 = np.hanning(patch_size)
    w2d = np.outer(h1, h1).astype(np.float32)
    # Giữ trọng số tối thiểu để các góc mép không bị triệt tiêu hoàn toàn
    w2d = np.clip(w2d, min_weight, 1.0)
    return w2d


def predict_sliding_window(
    model: torch.nn.Module,
    tif_path: Path | str,
    patch_size: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    normalize_max: float = 10000.0,
    add_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy suy luận trượt (Sliding Window Inference) trên toàn cảnh ảnh vệ tinh Sentinel-2.

    Args:
        model: Mô hình PyTorch đã nạp trọng số tối ưu.
        tif_path: Đường dẫn file GeoTIFF Sentinel-2 gốc (4 kênh, EPSG:32648).
        patch_size: Kích thước cạnh patch vuông (256).
        overlap: Độ rộng vùng chồng lấn giữa các patch kề nhau (64).
        batch_size: Kích thước lô đưa vào mạng nơ-ron (8).
        device: Thiết bị chạy ('cuda' hoặc 'cpu').
        use_tta: Bật Test-Time Augmentation (lật ngang, lật dọc).
        normalize_max: Hệ số chuẩn hóa phản xạ bề mặt Sentinel-2 (10000.0).
        add_indices: Ghép thêm 2 kênh NDVI & NDWI (tensor 6 kênh).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pred_mask: Ma trận nhãn 2D (H, W) kiểu uint8 (0..6).
            - prob_map: Ma trận xác suất 3D (NUM_CLASSES, H, W) kiểu float32.
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh vệ tinh: {tif_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    stride = patch_size - overlap
    assert stride > 0, "Stride phải lớn hơn 0 (overlap phải nhỏ hơn patch_size)."

    window_weights = create_2d_window(patch_size=patch_size)  # (patch_size, patch_size)

    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width
        bands_count = src.count
        # Đọc toàn bộ ảnh 4 kênh (B2, B3, B4, B8)
        img_full = src.read()  # (C, H, W)

    logger.info(
        f"Bắt đầu Sliding Window Inference trên ảnh: {width}x{height} px, {bands_count} kênh | Device: {device}"
    )

    # Nodata mask: những pixel có toàn bộ 4 kênh bằng 0
    is_nodata = (img_full.sum(axis=0) == 0)

    # Chuẩn hoá dải giá trị [0, 1] và tính chỉ số quang phổ nếu cần
    img_full_norm = np.clip(img_full.astype(np.float32) / normalize_max, 0.0, 1.0)
    if add_indices:
        img_full_norm = compute_spectral_indices(img_full_norm)
        logger.info("Đã tính toán và ghép thêm 2 kênh NDVI & NDWI (tensor 6 kênh đầu vào).")

    # Tạo mảng tích lũy xác suất và trọng số
    prob_accum = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    weight_accum = np.zeros((height, width), dtype=np.float32)

    # Lập danh sách tọa độ (y0, y1, x0, x1) bao phủ toàn bộ ảnh
    y_steps = list(range(0, height - patch_size + 1, stride))
    if len(y_steps) == 0 or y_steps[-1] + patch_size < height:
        y_steps.append(max(0, height - patch_size))

    x_steps = list(range(0, width - patch_size + 1, stride))
    if len(x_steps) == 0 or x_steps[-1] + patch_size < width:
        x_steps.append(max(0, width - patch_size))

    # Loại bỏ tọa độ trùng nếu có
    y_steps = sorted(list(set(y_steps)))
    x_steps = sorted(list(set(x_steps)))

    patch_coords: List[Tuple[int, int, int, int]] = []
    for y0 in y_steps:
        y1 = min(y0 + patch_size, height)
        for x0 in x_steps:
            x1 = min(x0 + patch_size, width)
            patch_coords.append((y0, y1, x0, x1))

    total_patches = len(patch_coords)
    logger.info(f"Tổng số patch cần duyệt: {total_patches} (kích thước {patch_size}x{patch_size}, overlap {overlap}px)")

    # Gom batch suy luận
    for i in range(0, total_patches, batch_size):
        batch_slice = patch_coords[i : i + batch_size]
        batch_tensors = []
        batch_meta = []

        for y0, y1, x0, x1 in batch_slice:
            patch_raw = img_full_norm[:, y0:y1, x0:x1]
            ph, pw = patch_raw.shape[1], patch_raw.shape[2]

            # Xử lý trường hợp ảnh nhỏ hơn patch_size (pad 0 nếu cần)
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch_padded = np.pad(patch_raw, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            else:
                patch_padded = patch_raw
                pad_h = 0
                pad_w = 0

            batch_tensors.append(patch_padded)
            batch_meta.append((y0, y1, x0, x1, ph, pw))

        x_batch = torch.from_numpy(np.stack(batch_tensors, axis=0)).float().to(device)

        with torch.no_grad():
            if use_tta:
                # TTA: Dự đoán gốc + lật ngang + lật dọc
                out0 = torch.softmax(model(x_batch), dim=1)

                x_hflip = torch.flip(x_batch, dims=[-1])
                out_hflip = torch.flip(torch.softmax(model(x_hflip), dim=1), dims=[-1])

                x_vflip = torch.flip(x_batch, dims=[-2])
                out_vflip = torch.flip(torch.softmax(model(x_vflip), dim=1), dims=[-2])

                prob_batch = (out0 + out_hflip + out_vflip) / 3.0
            else:
                prob_batch = torch.softmax(model(x_batch), dim=1)

        prob_batch_np = prob_batch.cpu().numpy()  # (B, NUM_CLASSES, patch_size, patch_size)

        for b_idx, (y0, y1, x0, x1, ph, pw) in enumerate(batch_meta):
            p_crop = prob_batch_np[b_idx, :, :ph, :pw]
            w_crop = window_weights[:ph, :pw]

            prob_accum[:, y0:y1, x0:x1] += p_crop * w_crop[np.newaxis, :, :]
            weight_accum[y0:y1, x0:x1] += w_crop

    # Chuẩn hóa xác suất tổng hợp theo tổng trọng số
    weight_safe = np.maximum(weight_accum, 1e-6)
    prob_accum /= weight_safe[np.newaxis, :, :]

    # Trích xuất nhãn dự đoán (argmax)
    pred_mask = np.argmax(prob_accum, axis=0).astype(np.uint8)

    # Đưa các vùng nodata ngoài ranh giới khảo sát về 0 (Background)
    pred_mask[is_nodata] = 0

    logger.info("Hoàn thành suy luận Sliding Window toàn cảnh thành công.")
    return pred_mask, prob_accum


# ===========================================================================
# 3. THỐNG KÊ DIỆN TÍCH & XUẤT FILE GIS / ẢNH
# ===========================================================================

def compute_area_statistics(
    pred_mask: np.ndarray,
    pixel_size_m: float = 10.0,
    output_json: Optional[Path | str] = None,
    output_csv: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Tính toán diện tích từng lớp sử dụng đất (pixel, ha, km², %).

    Quy đổi viễn thám:
        - Độ phân giải: 10m x 10m = 100 m²/pixel
        - 1 pixel = 0.01 ha = 0.0001 km²
    """
    pixel_area_m2 = pixel_size_m * pixel_size_m
    ha_factor = pixel_area_m2 / 10_000.0
    km2_factor = pixel_area_m2 / 1_000_000.0

    # Tổng số pixel đất khảo sát có dữ liệu (lớp 1–6)
    valid_pixels = np.sum(pred_mask > 0)
    total_area_ha = valid_pixels * ha_factor
    total_area_km2 = valid_pixels * km2_factor

    stats: Dict[str, Any] = {
        "pixel_size_m": pixel_size_m,
        "total_foreground_pixels": int(valid_pixels),
        "total_area_ha": round(float(total_area_ha), 2),
        "total_area_km2": round(float(total_area_km2), 2),
        "classes": {},
    }

    csv_rows = []
    header = ["Mã lớp", "Tên lớp", "Số pixel", "Diện tích (ha)", "Diện tích (km²)", "Tỷ lệ (%)"]

    for c in range(1, NUM_CLASSES):
        count = int(np.sum(pred_mask == c))
        area_ha = count * ha_factor
        area_km2 = count * km2_factor
        pct = (count / valid_pixels * 100.0) if valid_pixels > 0 else 0.0

        class_name = CLASS_NAMES.get(c, f"Lớp {c}")
        stats["classes"][str(c)] = {
            "class_id": c,
            "class_name": class_name,
            "pixel_count": count,
            "area_ha": round(float(area_ha), 2),
            "area_km2": round(float(area_km2), 2),
            "percentage": round(float(pct), 2),
        }

        csv_rows.append([
            c,
            class_name,
            count,
            f"{area_ha:.2f}",
            f"{area_km2:.2f}",
            f"{pct:.2f}%",
        ])

    # Ghi file JSON nếu có yêu cầu
    if output_json is not None:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu bảng thống kê diện tích JSON: {out_json_path}")

    # Ghi file CSV nếu có yêu cầu
    if output_csv is not None:
        out_csv_path = Path(output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_rows)
            writer.writerow([])
            writer.writerow([
                "TỔNG CỘNG",
                "Toàn huyện Giao Thủy (Đất có dữ liệu)",
                valid_pixels,
                f"{total_area_ha:.2f}",
                f"{total_area_km2:.2f}",
                "100.00%",
            ])
        logger.info(f"Đã lưu bảng thống kê diện tích CSV: {out_csv_path}")

    return stats


def export_lulc_geotiff(
    pred_mask: np.ndarray,
    src_tif_path: Path | str,
    out_tif_path: Path | str,
) -> Path:
    """
    Xuất ma trận nhãn LULC ra file GeoTIFF chuẩn GIS với Colormap chuẩn GDAL và CRS EPSG:32648.
    """
    out_path = Path(out_tif_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_tif_path) as src:
        crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width

    # Thiết lập profile GeoTIFF chuẩn đơn kênh uint8 có Tiled Deflate và Palette
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "photometric": "palette",
    }

    # Xây dựng bảng màu RGBA chuẩn GDAL
    colormap = {
        0: (0, 0, 0, 0),         # Background: Trong suốt
        1: (255, 215, 0, 255),   # Lúa: Vàng
        2: (220, 20, 60, 255),   # Khu dân cư: Đỏ đô
        3: (0, 191, 255, 255),   # Thủy sản: Xanh nước biển
        4: (0, 100, 0, 255),     # Rừng ngập mặn: Xanh lá đậm
        5: (34, 139, 34, 255),   # Cây lâu năm: Xanh rừng
        6: (210, 180, 140, 255), # Đồng muối / Đất trống: Nâu cát
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_mask.astype(np.uint8), 1)
        dst.write_colormap(1, colormap)

    logger.info(f"Đã xuất bản đồ LULC GeoTIFF chuẩn GDAL: {out_path}")
    return out_path


def export_lulc_png(
    pred_mask: np.ndarray,
    out_png_path: Path | str,
    title: str = "BẢN ĐỒ HIỆN TRẠNG SỬ DỤNG ĐẤT (LULC) - HUYỆN GIAO THỦY",
) -> Path:
    """
    Xuất bản đồ trực quan màu chất lượng cao (PNG 300 DPI) kèm thanh chú giải (Legend).
    """
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Danh sách mã màu HEX tương ứng với CLASS_COLORS
    colors = [
        f"#{CLASS_COLORS[c][0]:02x}{CLASS_COLORS[c][1]:02x}{CLASS_COLORS[c][2]:02x}"
        for c in range(NUM_CLASSES)
    ]
    cmap = ListedColormap(colors)
    bounds = list(range(NUM_CLASSES + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    im = ax.imshow(pred_mask, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Tọa độ pixel X", fontsize=10)
    ax.set_ylabel("Tọa độ pixel Y", fontsize=10)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    # Tạo các patch cho Legend
    legend_elements = [
        Patch(facecolor=colors[c], edgecolor="black", linewidth=0.5, label=f"{c}. {CLASS_NAMES[c]}")
        for c in range(1, NUM_CLASSES)
    ]
    legend_elements.append(
        Patch(facecolor=colors[0], edgecolor="black", linewidth=0.5, label="0. Vùng ngoài (Nodata)")
    )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
        fontsize=10,
        title="Chú giải Loại đất",
        title_fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Đã xuất bản đồ LULC PNG trực quan: {out_path}")
    return out_path


def export_comparison_maps(
    pred_mask: np.ndarray,
    gt_mask_path: Path | str,
    s2_tif_path: Path | str,
    out_dir: Path | str,
) -> Tuple[Path, Path]:
    """
    Xuất ảnh so sánh song song (Side-by-Side: Sentinel-2 CIR / Ground Truth / Prediction)
    và bản đồ sai khác (Difference Map: Pixel đúng = Xanh/Trắng, Sai = Đỏ).
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    gt_path = Path(gt_mask_path)
    with rasterio.open(gt_path) as src:
        gt_mask = src.read(1).astype(np.uint8)

    with rasterio.open(s2_tif_path) as src:
        # Kênh 4 (B8 - NIR), Kênh 3 (B4 - Red), Kênh 2 (B3 - Green) -> False Color Infrared (CIR)
        nir = src.read(4).astype(np.float32)
        red = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)

    # Chuẩn hóa ảnh màu giả CIR (2% - 98% percentile stretch)
    cir = np.stack([nir, red, green], axis=-1)
    p2, p98 = np.percentile(cir[cir > 0], (2, 98))
    cir_stretched = np.clip((cir - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)

    colors = [
        f"#{CLASS_COLORS[c][0]:02x}{CLASS_COLORS[c][1]:02x}{CLASS_COLORS[c][2]:02x}"
        for c in range(NUM_CLASSES)
    ]
    cmap = ListedColormap(colors)
    bounds = list(range(NUM_CLASSES + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    # 1. Ảnh so sánh song song 3 panel
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=300)

    axes[0].imshow(cir_stretched)
    axes[0].set_title("(a) Ảnh Sentinel-2 Màu giả CIR (NIR-R-G)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("(b) Nhãn chuẩn Ground Truth (WorldCover 7 lớp)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].set_title("(c) Bản đồ Dự đoán U-Net (ResNet34 + MMU)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    legend_elements = [
        Patch(facecolor=colors[c], edgecolor="black", linewidth=0.5, label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=6,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    side_by_side_path = out_dir_path / "lulc_comparison_gt_vs_pred.png"
    plt.savefig(side_by_side_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Bản đồ sai khác (Difference Map)
    fig_diff, ax_diff = plt.subplots(figsize=(12, 7), dpi=300)

    # Mask valid: chỉ xét các pixel foreground ở GT
    valid = (gt_mask > 0)
    correct = (pred_mask == gt_mask) & valid
    incorrect = (pred_mask != gt_mask) & valid

    diff_map = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    # Background ngoài khảo sát: màu xám đen (0.15, 0.15, 0.15)
    diff_map[~valid] = [0.15, 0.15, 0.15]
    # Điểm dự đoán ĐÚNG: màu xanh lục ngọc (0.1, 0.75, 0.3)
    diff_map[correct] = [0.1, 0.75, 0.3]
    # Điểm dự đoán SAI: màu đỏ tươi rực rỡ (0.9, 0.1, 0.1)
    diff_map[incorrect] = [0.9, 0.1, 0.1]

    ax_diff.imshow(diff_map)
    total_valid = np.sum(valid)
    correct_count = np.sum(correct)
    scene_oa = (correct_count / total_valid * 100.0) if total_valid > 0 else 0.0

    ax_diff.set_title(
        f"BẢN ĐỒ SAI KHÁC TOÀN HUYỆN GIAO THỦY (Overall Accuracy = {scene_oa:.2f}%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_diff.axis("off")

    diff_legends = [
        Patch(facecolor=(0.1, 0.75, 0.3), edgecolor="black", label=f"Dự đoán ĐÚNG ({correct_count:,} px - {scene_oa:.1f}%)"),
        Patch(facecolor=(0.9, 0.1, 0.1), edgecolor="black", label=f"Dự đoán SAI ({np.sum(incorrect):,} px - {100.0 - scene_oa:.1f}%)"),
        Patch(facecolor=(0.15, 0.15, 0.15), edgecolor="black", label="Nền / Ngoài ranh giới (Nodata)"),
    ]
    ax_diff.legend(handles=diff_legends, loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    diff_path = out_dir_path / "lulc_difference_map.png"
    plt.savefig(diff_path, dpi=300, bbox_inches="tight")
    plt.close(fig_diff)

    logger.info(f"Đã xuất bản đồ so sánh 3-panel: {side_by_side_path}")
    logger.info(f"Đã xuất bản đồ sai khác toàn cảnh: {diff_path}")
    return side_by_side_path, diff_path


# ===========================================================================
# 4. HÀM THỰC THI CHÍNH (MAIN CLI WORKFLOW)
# ===========================================================================

def run_prediction_pipeline(
    config_path: Path | str = "configs/train_config.yaml",
    checkpoint_path: Optional[Path | str] = None,
    input_tif: Optional[Path | str] = None,
    gt_mask_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    enable_postprocess: Optional[bool] = None,
    enable_tta: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Quy trình toàn diện chạy suy luận toàn cảnh, áp dụng hậu xử lý và xuất sản phẩm bản đồ.
    """
    ensure_dirs()

    # 1. Đọc cấu hình YAML
    config_file = Path(config_path)
    cfg: Dict[str, Any] = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    inf_cfg = cfg.get("inference", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Thiết lập đường dẫn mặc định
    ckpt_file = Path(checkpoint_path or "outputs/checkpoints/best_model.pth")
    src_tif = Path(input_tif or "data/raw/giao_thuy_expanded_s2.tif")
    gt_tif = Path(gt_mask_path or "data/processed/giao_thuy_expanded_mask7.tif")
    out_dir = Path(output_dir or MAP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = inf_cfg.get("patch_size", PATCH_SIZE)
    overlap = inf_cfg.get("overlap", 64)
    batch_size = inf_cfg.get("batch_size", 8)
    use_tta = enable_tta if enable_tta is not None else inf_cfg.get("use_tta", True)
    normalize_max = float(data_cfg.get("normalize_max", 10000.0))

    # Tham số hậu xử lý (Bước 2.7)
    do_postprocess = enable_postprocess if enable_postprocess is not None else inf_cfg.get("postprocess", True)
    kernel_size = int(inf_cfg.get("kernel_size", 3))
    morph_method = str(inf_cfg.get("morph_method", "majority"))
    enable_morph = bool(inf_cfg.get("enable_morph", True))
    enable_mmu = bool(inf_cfg.get("enable_mmu", True))
    mmu_pixels = int(inf_cfg.get("mmu_pixels", 5))

    logger.info("=" * 80)
    logger.info(" QUY TRÌNH SUY LUẬN TOÀN CẢNH LULC & HẬU XỬ LÝ (BƯỚC 2.7 & 2.8)")
    logger.info("=" * 80)
    logger.info(f"Ảnh đầu vào        : {src_tif}")
    logger.info(f"Checkpoint mô hình : {ckpt_file}")
    logger.info(f"Kích thước patch   : {patch_size}x{patch_size} (overlap={overlap}px, TTA={use_tta})")
    logger.info(f"Hậu xử lý (2.7)    : {do_postprocess} (Morphology={enable_morph} [{morph_method}, k={kernel_size}], MMU={enable_mmu} [{mmu_pixels}px = 500m²])")
    logger.info(f"Thư mục xuất bản đồ: {out_dir}")

    # 2. Khởi tạo mô hình và nạp trọng số
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_name = model_cfg.get("encoder_name", DEFAULT_ENCODER)
    num_classes = model_cfg.get("num_classes", NUM_CLASSES)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Tự động phát hiện số kênh đầu vào từ trọng số conv1 (4 hoặc 6 kênh)
    conv1_w = state_dict.get("encoder.conv1.weight", None)
    if conv1_w is not None:
        in_channels = int(conv1_w.shape[1])
    else:
        in_channels = model_cfg.get("in_channels", IN_CHANNELS)

    model = build_unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Đã nạp thành công mô hình từ {ckpt_file} (in_channels={in_channels})")

    add_indices = (in_channels == 6)

    # 3. Chạy suy luận cửa sổ trượt
    raw_pred_mask, prob_map = predict_sliding_window(
        model=model,
        tif_path=src_tif,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        use_tta=use_tta,
        normalize_max=normalize_max,
        add_indices=add_indices,
    )

    # 4. Hậu xử lý (Bước 2.7)
    if do_postprocess:
        logger.info(
            f"Bắt đầu hậu xử lý: Morphological Filter (k={kernel_size}) -> MMU Filter (ngưỡng {mmu_pixels} px)..."
        )
        final_mask = postprocess_mask(
            mask=raw_pred_mask,
            enable_morph=enable_morph,
            kernel_size=kernel_size,
            morph_method=morph_method,
            enable_mmu=enable_mmu,
            mmu_pixels=mmu_pixels,
            preserve_background=True,
        )
        # Đếm số pixel đã được làm mịn
        diff_count = np.sum(raw_pred_mask != final_mask)
        logger.info(f"Hậu xử lý hoàn thành: đã làm mịn và sáp nhập {diff_count:,} pixels ({diff_count / raw_pred_mask.size * 100:.3f}% tổng ảnh).")
    else:
        final_mask = raw_pred_mask
        logger.info("Bỏ qua bước hậu xử lý (postprocess = False).")

    # 5. Xuất bản đồ GeoTIFF và PNG
    out_geotiff = out_dir / "giao_thuy_lulc_prediction.tif"
    export_lulc_geotiff(final_mask, src_tif, out_geotiff)

    out_png = out_dir / "giao_thuy_lulc_prediction.png"
    export_lulc_png(final_mask, out_png)

    # 6. Thống kê diện tích (JSON & CSV)
    out_json = out_dir / "lulc_area_statistics.json"
    out_csv = out_dir / "lulc_area_statistics.csv"
    stats = compute_area_statistics(
        pred_mask=final_mask,
        pixel_size_m=10.0,
        output_json=out_json,
        output_csv=out_csv,
    )

    # 7. So sánh với Ground Truth (nếu có)
    if gt_tif.exists():
        export_comparison_maps(
            pred_mask=final_mask,
            gt_mask_path=gt_tif,
            s2_tif_path=src_tif,
            out_dir=out_dir,
        )

    # 8. In tóm tắt diện tích ra console
    print("\n" + "=" * 80)
    print(" BẢNG THỐNG KÊ DIỆN TÍCH HIỆN TRẠNG SỬ DỤNG ĐẤT - HUYỆN GIAO THỦY")
    print("=" * 80)
    print(f"{'Mã':<4} | {'Tên Loại Đất':<30} | {'Số Pixel':<12} | {'Diện Tích (ha)':<15} | {'Tỷ Lệ (%)':<10}")
    print("-" * 80)
    for c_id, c_data in stats["classes"].items():
        print(
            f"{c_data['class_id']:<4} | {c_data['class_name']:<30} | {c_data['pixel_count']:<12,}"
            f" | {c_data['area_ha']:<15,.2f} | {c_data['percentage']:<10.2f}%"
        )
    print("-" * 80)
    print(
        f"{'TỔNG':<4} | {'Toàn bộ diện tích khảo sát':<30} | {stats['total_foreground_pixels']:<12,}"
        f" | {stats['total_area_ha']:<15,.2f} | 100.00%"
    )
    print("=" * 80 + "\n")

    return {
        "final_mask": final_mask,
        "stats": stats,
        "geotiff_path": out_geotiff,
        "png_path": out_png,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LULC Inference & Hậu xử lý (Bước 2.7 & 2.8) - Huyện Giao Thủy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Đường dẫn file cấu hình YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn checkpoint (.pth) (mặc định lấy từ outputs/checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--input-tif",
        type=str,
        default=None,
        help="Đường dẫn ảnh Sentinel-2 composite gốc.",
    )
    parser.add_argument(
        "--gt-mask",
        type=str,
        default=None,
        help="Đường dẫn mask nhãn chuẩn đối chiếu (Ground Truth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục lưu trữ bản đồ và thống kê.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Tắt toàn bộ bước hậu xử lý hình thái học và MMU.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Tắt Test-Time Augmentation (chạy nhanh hơn).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enable_pp = False if args.no_postprocess else None
    enable_tta = False if args.no_tta else None

    run_prediction_pipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_tif=args.input_tif,
        gt_mask_path=args.gt_mask,
        output_dir=args.output_dir,
        enable_postprocess=enable_pp,
        enable_tta=enable_tta,
    )
