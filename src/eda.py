"""Bước 1.3 — Phân tích Khám phá Dữ liệu (EDA) và Tính Class Weights.

Module này thực hiện:
1. Tính toán trọng số lớp học (Class Weights) bằng phương pháp Median Frequency Balancing (MFB)
   chỉ áp dụng cho các lớp 1–6 (Lớp 0 / Background không tham gia, cố định weight = 0.0).
2. Lưu trọng số ra file JSON sẵn sàng nạp vào hàm mất mát PyTorch (Giai đoạn 2).
3. Vẽ biểu đồ phân bố diện tích / số pixel của 7 lớp sử dụng bảng màu quy chuẩn.
4. Tạo ảnh chồng lớp không gian (Spatial Overlay) đối chiếu giữa ảnh vệ tinh Sentinel-2 RGB
   và nhãn phân loại 7 lớp.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")  # Chế độ headless, không cần GUI display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .config import (
    CLASS_COLORS,
    CLASS_NAMES,
    GEE_SCALE_M,
    PROCESSED_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# Cấu hình font cho Matplotlib hiển thị tiếng Việt an toàn
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Segoe UI", "Tahoma"]
plt.rcParams["axes.unicode_minus"] = False


def compute_median_frequency_weights(
    hist: Mapping[int, int],
    scale_m: float = GEE_SCALE_M,
) -> dict[str, Any]:
    """Tính toán Class Weights theo phương pháp Median Frequency Balancing (MFB).

    Nguyên tắc bắt buộc theo roadmap:
    - Chỉ tính toán trên các lớp mục tiêu foreground: 1, 2, 3, 4, 5, 6.
    - Lớp 0 (Background) hoàn toàn không tham gia, trọng số gán cứng = 0.0 (khớp với
      ignore_index=0 trong Cross-Entropy Loss).
    - Các lớp không xuất hiện trong AOI (count = 0) được gán weight = 0.0.

    Công thức:
        f_c = count(c) / sum_{i=1..6}(count(i))
        median_freq = median({f_c | c in 1..6, f_c > 0})
        w_c = median_freq / f_c (nếu count(c) > 0, ngược lại 0.0)

    Args:
        hist: Dict {class_id: pixel_count} cho các lớp 0..6.
        scale_m: Độ phân giải pixel tính theo mét (mặc định 10 m).

    Returns:
        Dict chứa thống kê chi tiết, tần suất, và danh sách trọng số 7 lớp.
    """
    pixel_area_ha = (scale_m * scale_m) / 10_000.0  # 100 m² = 0.01 ha

    # 1. Thu thập dữ liệu các lớp foreground (1..6)
    target_classes = [c for c in range(1, 7)]
    fg_counts = {c: int(hist.get(c, 0)) for c in target_classes}
    total_fg_pixels = sum(fg_counts.values())

    if total_fg_pixels == 0:
        logger.warning("Không có pixel foreground nào (lớp 1–6) trong dữ liệu.")
        return {
            "weights_list": [0.0] * 7,
            "raw_weights": {c: 0.0 for c in range(7)},
            "normalized_weights": {c: 0.0 for c in range(7)},
            "median_frequency": 0.0,
            "frequencies": {c: 0.0 for c in range(7)},
            "pixel_counts": {c: int(hist.get(c, 0)) for c in range(7)},
            "area_ha": {c: round(int(hist.get(c, 0)) * pixel_area_ha, 2) for c in range(7)},
        }

    # 2. Tần suất từng lớp foreground
    frequencies: dict[int, float] = {}
    for c in target_classes:
        frequencies[c] = fg_counts[c] / float(total_fg_pixels)
    frequencies[0] = 0.0  # Lớp 0 không tính vào tần suất foreground

    # 3. Tần suất trung vị (chỉ tính trên các lớp có số lượng pixel > 0)
    pos_frequencies = [f for c, f in frequencies.items() if c in target_classes and f > 0]
    median_freq = float(np.median(pos_frequencies)) if pos_frequencies else 0.0

    # 4. Trọng số thô (raw weights)
    raw_weights: dict[int, float] = {0: 0.0}
    for c in target_classes:
        f_c = frequencies[c]
        if f_c > 0 and median_freq > 0:
            raw_weights[c] = round(median_freq / f_c, 4)
        else:
            raw_weights[c] = 0.0

    # 5. Trọng số chuẩn hóa (mean của các lớp có trọng số > 0 bằng 1.0)
    pos_weights = [w for c, w in raw_weights.items() if c in target_classes and w > 0]
    mean_weight = float(np.mean(pos_weights)) if pos_weights else 1.0
    normalized_weights: dict[int, float] = {0: 0.0}
    for c in target_classes:
        if raw_weights[c] > 0 and mean_weight > 0:
            normalized_weights[c] = round(raw_weights[c] / mean_weight, 4)
        else:
            normalized_weights[c] = 0.0

    # Danh sách trọng số 7 phần tử theo thứ tự chỉ số [0..6]
    weights_list = [raw_weights[c] for c in range(7)]
    norm_weights_list = [normalized_weights[c] for c in range(7)]

    pixel_counts = {c: int(hist.get(c, 0)) for c in range(7)}
    area_ha = {c: round(pixel_counts[c] * pixel_area_ha, 2) for c in range(7)}

    return {
        "weights_list": weights_list,
        "normalized_weights_list": norm_weights_list,
        "raw_weights": raw_weights,
        "normalized_weights": normalized_weights,
        "median_frequency": round(median_freq, 6),
        "frequencies": {c: round(frequencies[c], 6) for c in range(7)},
        "pixel_counts": pixel_counts,
        "area_ha": area_ha,
        "class_names": CLASS_NAMES,
    }


def generate_class_distribution_plot(
    hist: Mapping[int, int],
    out_png: str | Path,
    scale_m: float = GEE_SCALE_M,
) -> Path:
    """Tạo biểu đồ cột phân bố diện tích và số pixel của 7 lớp sử dụng đất.

    Args:
        hist: Dict thống kê số pixel {class_id: count}.
        out_png: Đường dẫn lưu ảnh PNG.
        scale_m: Độ phân giải pixel (10m).

    Returns:
        Đường dẫn file PNG đã lưu.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pixel_area_ha = (scale_m * scale_m) / 10_000.0
    classes = list(range(7))
    names = [f"[{c}] {CLASS_NAMES.get(c, '')}" for c in classes]
    counts = [int(hist.get(c, 0)) for c in classes]
    areas = [c * pixel_area_ha for c in counts]
    total_pixels = sum(counts)

    # Chuyển màu từ RGB (0..255) sang RGBA (0..1)
    colors = [
        [c / 255.0 for c in CLASS_COLORS.get(cls, (100, 100, 100))]
        for cls in classes
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(names, areas, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)

    # Thêm số liệu diện tích và tỷ lệ % trên đầu mỗi cột
    max_area = max(areas) if areas else 1
    for bar, count, area in zip(bars, counts, areas):
        pct = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_area * 0.015,
            f"{area:,.1f} ha\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Phân Bố Diện Tích 7 Lớp Quy Hoạch Sử Dụng Đất (Cấp Xã)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Diện Tích (ha)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lớp Sử Dụng Đất", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_area * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved class distribution plot to %s", out_path)
    return out_path


def _normalize_channel(ch: np.ndarray) -> np.ndarray:
    """Chuẩn hóa percentile 2% - 98% cho ảnh quang học Sentinel-2."""
    valid = ch[~np.isnan(ch) & (ch > 0)]
    if len(valid) == 0:
        return np.zeros_like(ch, dtype=np.uint8)
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    if p98 > p2:
        norm = (ch - p2) / (p98 - p2)
    else:
        norm = ch
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def generate_spatial_overlay_plot(
    mask_path: str | Path,
    s2_path: str | Path,
    out_png: str | Path,
) -> Path:
    """Tạo ảnh so sánh không gian 3 bảng: Sentinel-2 RGB, Mask 7 Lớp, và Spatial Overlay.

    Args:
        mask_path: File GeoTIFF mask 7 lớp.
        s2_path: File GeoTIFF Sentinel-2 composite 4 băng (B2, B3, B4, B8).
        out_png: File PNG đầu ra.

    Returns:
        Đường dẫn file PNG đã lưu.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path) as src_m:
        mask = src_m.read(1)

    with rasterio.open(s2_path) as src_s2:
        # Sentinel-2 bands: B2 (1), B3 (2), B4 (3) -> True Color RGB = [B4, B3, B2] = [3, 2, 1]
        if src_s2.count >= 3:
            r = src_s2.read(3).astype(float)
            g = src_s2.read(2).astype(float)
            b = src_s2.read(1).astype(float)
            r_norm = _normalize_channel(r)
            g_norm = _normalize_channel(g)
            b_norm = _normalize_channel(b)
            s2_rgb = np.dstack([r_norm, g_norm, b_norm])
        else:
            gray = _normalize_channel(src_s2.read(1).astype(float))
            s2_rgb = np.dstack([gray, gray, gray])

    # Tạo mảng RGB từ Mask 7 lớp
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        mask_rgb[mask == class_id] = color

    # Tạo ảnh chồng lớp (Alpha Blend: 50% Sentinel-2 + 50% Mask)
    # Riêng vùng Background (0) giữ nguyên màu ảnh Sentinel-2 để dễ quan sát
    alpha = 0.5
    overlay = (s2_rgb * (1.0 - alpha) + mask_rgb * alpha).astype(np.uint8)
    overlay[mask == 0] = s2_rgb[mask == 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=300)

    axes[0].imshow(s2_rgb)
    axes[0].set_title("1. Ảnh Vệ Tinh Sentinel-2 (True Color RGB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("2. Bản Đồ Hiện Trạng 7 Lớp (ESA Remapped)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("3. Không Gian Đối Chiếu (Spatial Overlay 50%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Tạo chú thích (Legend) ở đáy biểu đồ
    legend_patches = [
        mpatches.Patch(
            color=[c / 255.0 for c in CLASS_COLORS[cid]],
            label=f"[{cid}] {CLASS_NAMES[cid]}",
        )
        for cid in range(7)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=7,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.suptitle(
        "Đối Chiếu Dữ Liệu Không Gian: Sentinel-2 vs Bản Đồ Nhãn Đất Đai (Giao Thủy)",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    plt.subplots_adjust(bottom=0.15, top=0.90, wspace=0.08)

    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved spatial overlay plot to %s", out_path)
    return out_path


def run_eda_pipeline(
    mask_path: str | Path,
    s2_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Thực hiện toàn bộ quy trình EDA và tính toán trọng số lớp học.

    Args:
        mask_path: Đường dẫn GeoTIFF mask 7 lớp.
        s2_path: Đường dẫn GeoTIFF Sentinel-2 composite.
        out_dir: Thư mục xuất kết quả (mặc định data/processed).

    Returns:
        Dict tổng hợp kết quả phân tích EDA.
    """
    ensure_dirs()
    mask_path = Path(mask_path)
    s2_path = Path(s2_path)
    target_dir = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if not mask_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mask: {mask_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Sentinel-2: {s2_path}")

    # Đọc mask và tính histogram
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
    unique_vals, counts = np.unique(mask_data, return_counts=True)
    hist = {c: 0 for c in range(7)}
    for u, cnt in zip(unique_vals, counts):
        hist[int(u)] = int(cnt)

    # 1. Tính Median Frequency Weights
    eda_stats = compute_median_frequency_weights(hist)

    # 2. Xuất file class_weights.json chuẩn cho PyTorch Loss
    weights_json_path = target_dir / "class_weights.json"
    weights_payload = {
        "loss_target": "WeightedCrossEntropyLoss / DiceLoss",
        "method": "Median Frequency Balancing (MFB)",
        "contract": "ignore_index=0",
        "description": "Trọng số huấn luyện lớp 0..6 (Lớp 0 = 0.0 không tham gia)",
        "weights": eda_stats["weights_list"],
        "normalized_weights": eda_stats["normalized_weights_list"],
        "raw_dict": {str(k): v for k, v in eda_stats["raw_weights"].items()},
        "normalized_dict": {str(k): v for k, v in eda_stats["normalized_weights"].items()},
        "median_frequency": eda_stats["median_frequency"],
        "classes": {str(k): CLASS_NAMES[k] for k in range(7)},
    }
    with open(weights_json_path, "w", encoding="utf-8") as fh:
        json.dump(weights_payload, fh, ensure_ascii=False, indent=2)
    logger.info("Saved class weights to %s", weights_json_path)

    # 3. Xuất file eda_summary.json
    summary_json_path = target_dir / "eda_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(eda_stats, fh, ensure_ascii=False, indent=2)
    logger.info("Saved EDA summary to %s", summary_json_path)

    # 4. Sinh các đồ thị trực quan hóa
    plot_dist_path = target_dir / "eda_class_distribution.png"
    generate_class_distribution_plot(hist, plot_dist_path)

    plot_overlay_path = target_dir / "eda_spatial_overlay.png"
    generate_spatial_overlay_plot(mask_path, s2_path, plot_overlay_path)

    eda_stats["weights_json_path"] = str(weights_json_path)
    eda_stats["summary_json_path"] = str(summary_json_path)
    eda_stats["distribution_plot_path"] = str(plot_dist_path)
    eda_stats["overlay_plot_path"] = str(plot_overlay_path)

    return eda_stats


def main() -> None:
    """CLI thực thi quy trình EDA."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Bước 1.3: EDA & Tính Class Weights (Median Frequency Balancing)"
    )
    parser.add_argument("mask", help="Đường dẫn GeoTIFF mask 7 lớp (ví dụ: data/processed/giao_thuy_mask7.tif)")
    parser.add_argument("s2", help="Đường dẫn GeoTIFF Sentinel-2 composite (ví dụ: data/raw/giao_thuy_s2.tif)")
    parser.add_argument(
        "--out-dir",
        default=str(PROCESSED_DIR),
        help=f"Thư mục lưu báo cáo và đồ thị (mặc định {PROCESSED_DIR})",
    )
    args = parser.parse_args()

    results = run_eda_pipeline(args.mask, args.s2, args.out_dir)

    print("\n" + "=" * 65)
    print(" KẾT QUẢ TÍNH TOÁN CLASS WEIGHTS (MEDIAN FREQUENCY BALANCING)")
    print("=" * 65)
    print(f"{'Mã':<4} | {'Tên Lớp':<30} | {'Số Pixel':>10} | {'Trọng Số MFB':>12}")
    print("-" * 65)
    for c in range(7):
        name = CLASS_NAMES.get(c, "")
        count = results["pixel_counts"].get(c, 0)
        weight = results["weights_list"][c]
        print(f"{c:<4} | {name:<30} | {count:>10,} | {weight:>12.4f}")
    print("=" * 65)
    print(f"\n[OK] File trọng số PyTorch: {results['weights_json_path']}")
    print(f"[OK] File tóm tắt EDA:     {results['summary_json_path']}")
    print(f"[OK] Biểu đồ phân bố:       {results['distribution_plot_path']}")
    print(f"[OK] Ảnh đối chiếu Overlay: {results['overlay_plot_path']}")


if __name__ == "__main__":
    main()
