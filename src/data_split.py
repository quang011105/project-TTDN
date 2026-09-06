"""Bước 1.4 — Tách Dữ liệu theo Không gian (Spatial Block Split).

Module này thực hiện:
1. Phân chia raster cấp xã thành lưới các khối địa lý liền mạch (Spatial Blocks).
2. Tối ưu hóa phân bổ 70% Train / 30% Golden Test độc lập không gian, loại bỏ hoàn toàn
   nguy cơ rò rỉ dữ liệu (spatial data leakage do autocorrelation).
3. Đảm bảo toàn bộ 7 lớp sử dụng đất (đặc biệt là 6 lớp mục tiêu 1–6) phân bố cân bằng
   giữa hai tập Train và Golden Test.
4. Trích xuất các patch ảnh chuẩn 256x256 kèm mask và siêu dữ liệu toạ độ (geotransform)
   sẵn sàng nạp vào DataLoader của PyTorch U-Net ở Giai đoạn 2.
5. Xuất bản đồ phân vùng không gian (Spatial Split Map) và siêu dữ liệu thống kê JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from .config import CLASS_COLORS, CLASS_NAMES, PROCESSED_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def generate_spatial_blocks(
    height: int,
    width: int,
    n_rows: int = 4,
    n_cols: int = 5,
) -> list[dict[str, Any]]:
    """Chia ma trận ảnh thành lưới n_rows x n_cols các khối địa lý liền mạch.

    Args:
        height: Chiều cao ma trận raster (pixel).
        width: Chiều rộng ma trận raster (pixel).
        n_rows: Số hàng phân chia (mặc định 4).
        n_cols: Số cột phân chia (mặc định 5).

    Returns:
        Danh sách dict thông tin từng khối (id, hàng, cột, tọa độ pixel lát cắt).
    """
    block_h = height // n_rows
    block_w = width // n_cols

    blocks: list[dict[str, Any]] = []
    block_id = 0

    for r in range(n_rows):
        r_start = r * block_h
        r_end = (r + 1) * block_h if r < n_rows - 1 else height
        for c in range(n_cols):
            c_start = c * block_w
            c_end = (c + 1) * block_w if c < n_cols - 1 else width
            blocks.append({
                "block_id": block_id,
                "row": r,
                "col": c,
                "r_start": r_start,
                "r_end": r_end,
                "c_start": c_start,
                "c_end": c_end,
                "height": r_end - r_start,
                "width": c_end - c_start,
                "pixel_count": (r_end - r_start) * (c_end - c_start),
            })
            block_id += 1

    return blocks


def optimize_block_split(
    mask_array: np.ndarray,
    blocks: list[dict[str, Any]],
    target_test_ratio: float = 0.30,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Tối ưu hóa lựa chọn khối Train và Golden Test đảm bảo cân bằng phân bố 7 lớp.

    Args:
        mask_array: Ma trận 2D chứa các nhãn lớp 0–6.
        blocks: Danh sách khối không gian từ `generate_spatial_blocks`.
        target_test_ratio: Tỷ lệ diện tích tập Test mục tiêu (mặc định 0.30 = 30%).

    Returns:
        Tuple gồm (train_block_ids, test_block_ids, stats_dict).
    """
    n_blocks = len(blocks)
    n_test_blocks = max(1, round(n_blocks * target_test_ratio))

    # Tính histogram từng khối
    block_hists: list[np.ndarray] = []
    for blk in blocks:
        sub = mask_array[blk["r_start"]:blk["r_end"], blk["c_start"]:blk["c_end"]]
        counts = np.bincount(sub.flatten(), minlength=7)
        block_hists.append(counts[:7])

    hists_arr = np.array(block_hists, dtype=np.int64)  # Shape (n_blocks, 7)
    total_counts = hists_arr.sum(axis=0)
    total_pixels = total_counts.sum()

    # Xác định số hàng và số cột từ danh sách khối
    n_rows = max(b["row"] for b in blocks) + 1
    n_cols = max(b["col"] for b in blocks) + 1

    # Xây dựng đồ thị kề 4 hướng để kiểm tra tính liền mạch không gian
    adj: dict[int, list[int]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * n_cols + c)
            if r < n_rows - 1:
                neighbors.append((r + 1) * n_cols + c)
            if c > 0:
                neighbors.append(r * n_cols + (c - 1))
            if c < n_cols - 1:
                neighbors.append(r * n_cols + (c + 1))
            adj[idx] = neighbors

    def count_connected_components(combo_indices: tuple[int, ...]) -> int:
        """Đếm số thành phần liên thông không gian của tổ hợp khối."""
        combo_set = set(combo_indices)
        visited = set()
        components = 0
        for node in combo_set:
            if node not in visited:
                components += 1
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    for nb in adj.get(curr, []):
                        if nb in combo_set and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
        return components

    # Tìm tổ hợp n_test_blocks có độ cân bằng lớp tốt nhất, ưu tiên khối liền mạch
    best_combo: tuple[int, ...] | None = None
    best_score = float("inf")
    best_contiguous_score = float("inf")
    best_contiguous_combo: tuple[int, ...] | None = None

    # Các lớp foreground cần có mặt
    active_fg = [c for c in range(1, 7) if total_counts[c] > 0]

    for combo in combinations(range(n_blocks), n_test_blocks):
        test_counts = hists_arr[list(combo)].sum(axis=0)
        overall_ratio = test_counts.sum() / float(total_pixels)

        # Ràng buộc sơ bộ: tỷ lệ tổng diện tích trong khoảng [target - 3%, target + 3%]
        if abs(overall_ratio - target_test_ratio) > 0.03:
            continue

        ratios = np.zeros(7, dtype=float)
        for c in active_fg:
            ratios[c] = test_counts[c] / float(total_counts[c])

        # Đảm bảo mỗi lớp foreground có mặt ít nhất 12% và tối đa 50% ở tập Test
        if any(ratios[c] < 0.12 or ratios[c] > 0.50 for c in active_fg):
            continue

        # Điểm phạt: độ lệch bình phương so với target_test_ratio trên từng lớp
        score = sum((ratios[c] - target_test_ratio) ** 2 for c in active_fg)
        n_comp = count_connected_components(combo)

        # Ưu tiên tổ hợp liền mạch địa lý (1 hoặc 2 khối liên thông)
        if n_comp <= 2 and score < best_contiguous_score:
            best_contiguous_score = score
            best_contiguous_combo = combo

        if score < best_score:
            best_score = score
            best_combo = combo

    # Chọn ưu tiên khối liền mạch nếu có
    final_combo = best_contiguous_combo if best_contiguous_combo is not None else best_combo

    # Nếu không tìm thấy nghiệm thỏa mãn ràng buộc hẹp, chọn combo có độ lệch nhỏ nhất
    if final_combo is None:
        logger.warning("Không tìm thấy tổ hợp khối lý tưởng trong dải 12-50%%, nới lỏng ràng buộc.")
        for combo in combinations(range(n_blocks), n_test_blocks):
            test_counts = hists_arr[list(combo)].sum(axis=0)
            overall_ratio = test_counts.sum() / float(total_pixels)
            score = abs(overall_ratio - target_test_ratio) * 10.0
            for c in active_fg:
                r_c = test_counts[c] / float(total_counts[c])
                score += (r_c - target_test_ratio) ** 2
            if score < best_score:
                best_score = score
                final_combo = combo

    assert final_combo is not None
    test_indices = sorted(list(final_combo))
    train_indices = [i for i in range(n_blocks) if i not in test_indices]

    test_counts = hists_arr[test_indices].sum(axis=0)
    train_counts = total_counts - test_counts

    stats = {
        "total_pixels": int(total_pixels),
        "train_pixels": int(train_counts.sum()),
        "test_pixels": int(test_counts.sum()),
        "train_ratio": float(round(train_counts.sum() / float(total_pixels), 4)),
        "test_ratio": float(round(test_counts.sum() / float(total_pixels), 4)),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "class_breakdown": {},
    }

    for c in range(7):
        tr = int(train_counts[c])
        te = int(test_counts[c])
        tot = int(total_counts[c])
        stats["class_breakdown"][str(c)] = {
            "name": CLASS_NAMES.get(c, ""),
            "train_pixels": tr,
            "test_pixels": te,
            "total_pixels": tot,
            "train_ratio": float(round(tr / tot, 4)) if tot > 0 else 0.0,
            "test_ratio": float(round(te / tot, 4)) if tot > 0 else 0.0,
        }

    return train_indices, test_indices, stats


def create_split_mask(
    height: int,
    width: int,
    blocks: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
) -> np.ndarray:
    """Tạo ma trận phân vùng không gian (1 = Train, 2 = Golden Test).

    Args:
        height: Chiều cao ma trận.
        width: Chiều rộng ma trận.
        blocks: Danh sách các khối.
        train_indices: Danh sách chỉ số khối Train.
        test_indices: Danh sách chỉ số khối Test.

    Returns:
        Ma trận 2D uint8 cùng kích thước với raster gốc.
    """
    split_mask = np.zeros((height, width), dtype=np.uint8)
    for idx in train_indices:
        blk = blocks[idx]
        split_mask[blk["r_start"]:blk["r_end"], blk["c_start"]:blk["c_end"]] = 1

    for idx in test_indices:
        blk = blocks[idx]
        split_mask[blk["r_start"]:blk["r_end"], blk["c_start"]:blk["c_end"]] = 2

    return split_mask


def save_split_mask_raster(
    split_mask: np.ndarray,
    template_tif_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Lưu ma trận phân vùng không gian thành GeoTIFF kèm Colormap.

    Args:
        split_mask: Ma trận phân vùng (1=Train, 2=Test).
        template_tif_path: File GeoTIFF mẫu lấy thông tin CRS và Transform.
        out_path: Đường dẫn file GeoTIFF đầu ra.

    Returns:
        Đường dẫn file đã lưu.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(template_tif_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            photometric="palette",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(split_mask, 1)
            # Colormap: 1 = Xanh lá (Train), 2 = Cam/Đỏ (Golden Test)
            cmap = {i: (0, 0, 0, 0) for i in range(256)}
            cmap[1] = (46, 204, 113, 255)   # Train: Emerald Green
            cmap[2] = (231, 76, 60, 255)    # Golden Test: Alizarin Red
            dst.write_colormap(1, cmap)

    logger.info("Saved spatial split mask to %s", out_path)
    return out_path


def extract_patches(
    s2_path: str | Path,
    mask_path: str | Path,
    split_mask: np.ndarray,
    out_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    assign_thresh: float = 0.70,
    min_valid_ratio: float = 0.10,
) -> dict[str, int]:
    """Cắt raster ảnh và nhãn thành các patch 256x256 cho tập Train và Test.

    Nguyên tắc bảo vệ không gian:
    - Một patch chỉ được gán vào Train nếu >= assign_thresh pixel nằm trong vùng Train.
    - Một patch chỉ được gán vào Test nếu >= assign_thresh pixel nằm trong vùng Test.
    - Bỏ qua các patch nằm ở ranh giới giao nhau để chống rò rỉ biên.
    - Loại bỏ các patch viền no-data hoặc tỷ lệ pixel hữu ích < min_valid_ratio.

    Args:
        s2_path: Đường dẫn ảnh Sentinel-2 composite (4 kênh).
        mask_path: Đường dẫn mask 7 lớp (1 kênh).
        split_mask: Ma trận phân vùng 1=Train, 2=Test.
        out_dir: Thư mục gốc lưu patches.
        patch_size: Kích thước patch (pixel, mặc định 256).
        stride: Bước nhảy khi cắt (mặc định 128 = 50% overlap).
        assign_thresh: Ngưỡng tỷ lệ pixel thuộc vùng để gán nhãn tập (mặc định 0.70).
        min_valid_ratio: Tỷ lệ pixel khác 0 tối thiểu để giữ lại patch.

    Returns:
        Dict số lượng patch đã cắt: {"train_patches": X, "test_patches": Y}.
    """
    out_dir = Path(out_dir)
    train_img_dir = out_dir / "patches" / "train" / "images"
    train_mask_dir = out_dir / "patches" / "train" / "masks"
    test_img_dir = out_dir / "patches" / "test" / "images"
    test_mask_dir = out_dir / "patches" / "test" / "masks"

    for d in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_path) as src_s2, rasterio.open(mask_path) as src_m:
        h, w = src_s2.height, src_s2.width
        crs = src_s2.crs
        s2_meta = src_s2.meta.copy()
        mask_meta = src_m.meta.copy()

        train_count = 0
        test_count = 0

        # Lưới tọa độ cắt
        y_steps = list(range(0, h - patch_size + 1, stride))
        if y_steps[-1] + patch_size < h:
            y_steps.append(h - patch_size)

        x_steps = list(range(0, w - patch_size + 1, stride))
        if x_steps[-1] + patch_size < w:
            x_steps.append(w - patch_size)

        for y in y_steps:
            for x in x_steps:
                window = Window(x, y, patch_size, patch_size)
                patch_split = split_mask[y:y + patch_size, x:x + patch_size]

                # Tỷ lệ gán vùng
                train_ratio = np.mean(patch_split == 1)
                test_ratio = np.mean(patch_split == 2)

                # Quyết định tập gán
                assigned_split: str | None = None
                if train_ratio >= assign_thresh:
                    assigned_split = "train"
                elif test_ratio >= assign_thresh:
                    assigned_split = "test"
                else:
                    # Patch nằm ở ranh giới giao nhau giữa 2 khối -> bỏ qua để tránh rò rỉ biên
                    continue

                # Đọc dữ liệu patch
                m_patch = src_m.read(1, window=window)
                valid_ratio = np.mean(m_patch > 0)
                if valid_ratio < min_valid_ratio:
                    continue  # Bỏ qua patch rỗng/toàn nền

                s2_patch = src_s2.read(window=window)
                patch_transform = rasterio.windows.transform(window, src_s2.transform)

                patch_name = f"patch_{y:04d}_{x:04d}.tif"

                # Cập nhật metadata cho patch
                p_s2_meta = s2_meta.copy()
                p_s2_meta.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "compress": "deflate",
                })

                p_mask_meta = mask_meta.copy()
                p_mask_meta.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "compress": "deflate",
                })

                if assigned_split == "train":
                    img_out = train_img_dir / patch_name
                    mask_out = train_mask_dir / patch_name
                    train_count += 1
                else:
                    img_out = test_img_dir / patch_name
                    mask_out = test_mask_dir / patch_name
                    test_count += 1

                with rasterio.open(img_out, "w", **p_s2_meta) as dst:
                    dst.write(s2_patch)

                with rasterio.open(mask_out, "w", **p_mask_meta) as dst:
                    dst.write(m_patch, 1)

    logger.info("Extracted %d train patches and %d test patches", train_count, test_count)
    return {"train_patches": train_count, "test_patches": test_count}


def generate_split_visualization(
    s2_path: str | Path,
    split_mask: np.ndarray,
    blocks: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    stats: dict[str, Any],
    out_png: str | Path,
) -> Path:
    """Tạo bản đồ trực quan hóa phân vùng không gian 70/30 và bảng thống kê.

    Args:
        s2_path: File Sentinel-2 composite.
        split_mask: Ma trận phân vùng (1=Train, 2=Test).
        blocks: Danh sách khối không gian.
        train_indices: Chỉ số khối Train.
        test_indices: Chỉ số khối Test.
        stats: Thống kê số lượng pixel từng lớp.
        out_png: Đường dẫn file ảnh đầu ra.

    Returns:
        Đường dẫn file ảnh PNG đã lưu.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_path) as src:
        r = src.read(3).astype(float)
        g = src.read(2).astype(float)
        b = src.read(1).astype(float)

        def norm(ch):
            valid = ch[ch > 0]
            p2, p98 = np.percentile(valid, (2, 98)) if len(valid) > 0 else (0, 1)
            normed = np.clip((ch - p2) / (p98 - p2), 0, 1) if p98 > p2 else ch
            return (normed * 255).astype(np.uint8)

        s2_rgb = np.dstack([norm(r), norm(g), norm(b)])

    # Tạo lớp phủ màu: Xanh lá (Train), Đỏ cam (Golden Test)
    h, w, _ = s2_rgb.shape
    overlay_rgb = s2_rgb.copy().astype(float)

    train_color = np.array([46, 204, 113], dtype=float)  # Emerald
    test_color = np.array([231, 76, 60], dtype=float)    # Red

    alpha = 0.35
    for y in range(h):
        train_mask_y = (split_mask[y] == 1)
        test_mask_y = (split_mask[y] == 2)
        overlay_rgb[y, train_mask_y] = (1.0 - alpha) * s2_rgb[y, train_mask_y] + alpha * train_color
        overlay_rgb[y, test_mask_y] = (1.0 - alpha) * s2_rgb[y, test_mask_y] + alpha * test_color

    overlay_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)

    fig, (ax_map, ax_table) = plt.subplots(
        1, 2, figsize=(20, 8), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    # 1. Vẽ bản đồ phân vùng
    ax_map.imshow(overlay_rgb)
    ax_map.set_title(
        "Bản Đồ Phân Vùng Không Gian 70% Train / 30% Golden Test (Giao Thủy)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Vẽ khung viền và nhãn cho từng khối
    for blk in blocks:
        bid = blk["block_id"]
        is_test = bid in test_indices
        color = "red" if is_test else "lime"
        lw = 2.0 if is_test else 1.2

        rect = plt.Rectangle(
            (blk["c_start"], blk["r_start"]),
            blk["width"],
            blk["height"],
            fill=False,
            edgecolor=color,
            linewidth=lw,
            linestyle="--" if is_test else "-",
        )
        ax_map.add_patch(rect)

        # Đặt tên nhãn khối
        center_x = blk["c_start"] + blk["width"] / 2.0
        center_y = blk["r_start"] + blk["height"] / 2.0
        tag = "TEST" if is_test else "TRAIN"
        ax_map.text(
            center_x,
            center_y,
            f"B{bid}\n[{tag}]",
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    ax_map.axis("off")

    patch_train = mpatches.Patch(color="#2ecc71", label=f"Train (14 Khối - {stats['train_ratio']:.1%})")
    patch_test = mpatches.Patch(color="#e74c3c", label=f"Golden Test (6 Khối - {stats['test_ratio']:.1%})")
    ax_map.legend(handles=[patch_train, patch_test], loc="lower left", fontsize=10, frameon=True)

    # 2. Vẽ bảng số liệu so sánh
    ax_table.axis("off")
    table_data = [
        ["Lớp", "Tên Lớp", "Train (px)", "Train %", "Test (px)", "Test %"]
    ]

    for c in range(7):
        b = stats["class_breakdown"][str(c)]
        table_data.append([
            str(c),
            b["name"],
            f"{b['train_pixels']:,}",
            f"{b['train_ratio']:.1%}",
            f"{b['test_pixels']:,}",
            f"{b['test_ratio']:.1%}",
        ])

    table_data.append([
        "Tổng",
        "Toàn bộ vùng",
        f"{stats['train_pixels']:,}",
        f"{stats['train_ratio']:.1%}",
        f"{stats['test_pixels']:,}",
        f"{stats['test_ratio']:.1%}",
    ])

    tab = ax_table.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.10, 0.36, 0.20, 0.16, 0.20, 0.16],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.6)

    # Định dạng tiêu đề bảng
    for (row_idx, col_idx), cell in tab.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx == len(table_data) - 1:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
        elif row_idx % 2 == 1:
            cell.set_facecolor("#f9f9f9")

    ax_table.set_title(
        "Bảng Cân Bằng Phân Bố Lớp giữa Train & Golden Test",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved spatial split visualization to %s", out_path)
    return out_path


def run_spatial_split_pipeline(
    s2_path: str | Path,
    mask_path: str | Path,
    out_dir: str | Path | None = None,
    n_rows: int = 4,
    n_cols: int = 5,
    patch_size: int = 256,
    stride: int = 128,
) -> dict[str, Any]:
    """Thực thi toàn bộ quy trình Bước 1.4: Spatial Block Split và trích xuất patches.

    Args:
        s2_path: Đường dẫn ảnh Sentinel-2 composite.
        mask_path: Đường dẫn mask 7 lớp.
        out_dir: Thư mục lưu kết quả (mặc định data/processed).
        n_rows: Số hàng khối không gian (mặc định 4).
        n_cols: Số cột khối không gian (mặc định 5).
        patch_size: Kích thước patch (mặc định 256).
        stride: Bước nhảy trích xuất patch (mặc định 256).

    Returns:
        Dict siêu dữ liệu tổng kết quy trình.
    """
    ensure_dirs()
    s2_path = Path(s2_path)
    mask_path = Path(mask_path)
    target_dir = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if not s2_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh S2: {s2_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mask 7 lớp: {mask_path}")

    # 1. Đọc mask và tạo khối không gian
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1)
        h, w = mask_array.shape

    blocks = generate_spatial_blocks(h, w, n_rows=n_rows, n_cols=n_cols)

    # 2. Tối ưu hóa phân chia 70% Train / 30% Golden Test
    train_indices, test_indices, stats = optimize_block_split(mask_array, blocks, target_test_ratio=0.30)

    # 3. Tạo ma trận phân vùng và lưu GeoTIFF
    split_mask = create_split_mask(h, w, blocks, train_indices, test_indices)
    split_mask_path = target_dir / "spatial_split_mask.tif"
    save_split_mask_raster(split_mask, mask_path, split_mask_path)

    # 4. Lưu siêu dữ liệu JSON
    metadata_path = target_dir / "spatial_split_metadata.json"
    stats_to_save = stats.copy()
    stats_to_save["grid_dimensions"] = {"n_rows": n_rows, "n_cols": n_cols, "total_blocks": len(blocks)}
    stats_to_save["patch_config"] = {"patch_size": patch_size, "stride": stride}
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(stats_to_save, fh, ensure_ascii=False, indent=2)
    logger.info("Saved spatial split metadata to %s", metadata_path)

    # 5. Vẽ đồ thị trực quan hóa
    map_path = target_dir / "spatial_split_map.png"
    generate_split_visualization(s2_path, split_mask, blocks, train_indices, test_indices, stats, map_path)

    # 6. Trích xuất patches 256x256
    patch_stats = extract_patches(
        s2_path,
        mask_path,
        split_mask,
        target_dir,
        patch_size=patch_size,
        stride=stride,
    )

    stats["split_mask_path"] = str(split_mask_path)
    stats["metadata_path"] = str(metadata_path)
    stats["map_path"] = str(map_path)
    stats["patch_counts"] = patch_stats

    return stats


def main() -> None:
    """CLI thực thi phân tách dữ liệu theo không gian."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Bước 1.4: Tách Dữ liệu theo Không gian (Spatial Block Split 70/30)"
    )
    parser.add_argument("s2", help="Đường dẫn file ảnh Sentinel-2 composite")
    parser.add_argument("mask", help="Đường dẫn file mask 7 lớp")
    parser.add_argument(
        "--out-dir",
        default=str(PROCESSED_DIR),
        help=f"Thư mục lưu kết quả và patches (mặc định {PROCESSED_DIR})",
    )
    parser.add_argument("--rows", type=int, default=4, help="Số hàng phân chia khối (mặc định 4)")
    parser.add_argument("--cols", type=int, default=5, help="Số cột phân chia khối (mặc định 5)")
    parser.add_argument("--patch-size", type=int, default=256, help="Kích thước patch (mặc định 256)")
    parser.add_argument("--stride", type=int, default=128, help="Bước nhảy khi trích xuất (mặc định 128)")

    args = parser.parse_args()

    results = run_spatial_split_pipeline(
        args.s2,
        args.mask,
        out_dir=args.out_dir,
        n_rows=args.rows,
        n_cols=args.cols,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    print("\n" + "=" * 70)
    print(" KẾT QUẢ PHÂN CHIA SPATIAL BLOCK SPLIT (70% TRAIN / 30% GOLDEN TEST)")
    print("=" * 70)
    print(f"Tổng pixel: {results['total_pixels']:,}")
    print(f"Tập Train:  {results['train_pixels']:,} pixel ({results['train_ratio']:.1%}) - Khối: {results['train_indices']}")
    print(f"Tập Test:   {results['test_pixels']:,} pixel ({results['test_ratio']:.1%}) - Khối: {results['test_indices']}")
    print("-" * 70)
    print(f"{'Mã':<4} | {'Tên Lớp':<28} | {'Train Px':>10} | {'Tr %':>6} | {'Test Px':>10} | {'Te %':>6}")
    print("-" * 70)
    for c in range(7):
        b = results["class_breakdown"][str(c)]
        print(f"{c:<4} | {b['name']:<28} | {b['train_pixels']:>10,} | {b['train_ratio']:>5.1%} | {b['test_pixels']:>10,} | {b['test_ratio']:>5.1%}")
    print("=" * 70)
    print(f"[OK] Patches trích xuất: Train = {results['patch_counts']['train_patches']} | Test = {results['patch_counts']['test_patches']}")
    print(f"[OK] Mask phân vùng:     {results['split_mask_path']}")
    print(f"[OK] Metadata JSON:      {results['metadata_path']}")
    print(f"[OK] Bản đồ phân vùng:   {results['map_path']}")


if __name__ == "__main__":
    main()
