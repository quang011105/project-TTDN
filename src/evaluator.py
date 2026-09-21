"""
Đánh giá Toàn diện Mô hình Phân vùng LULC trên Tập Golden Test (33 patch).

Thực hiện đánh giá chuyên sâu theo đúng tiêu chuẩn thiết kế Phase 2:
    - Bỏ qua lớp 0 (Background / Nodata), chỉ tính toán trên 6 lớp quy hoạch mục tiêu (1–6).
    - Các chỉ số cốt lõi:
        + Overall Accuracy (OA)
        + Mean IoU (mIoU) & Per-class IoU
        + Macro F1-Score, Precision, Recall
        + Ma trận nhầm lẫn (Confusion Matrix 6x6)
        + Hệ số Cohen's Kappa
    - Xuất các sản phẩm trực quan hoá chất lượng cao:
        + outputs/logs/evaluation_report.json
        + outputs/logs/confusion_matrix.png (Heatmap)
        + outputs/logs/per_class_iou.png (Biểu đồ cột nhúng mã màu chuẩn GIS)
        + outputs/logs/training_curves.png (Đồ thị Loss & mIoU qua các epoch)

Cách dùng:
    python -m src.evaluator --checkpoint outputs/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cấu hình UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # Chế độ headless vẽ ảnh không cần màn hình GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CHECKPOINT_DIR,
    CLASS_COLORS,
    CLASS_NAMES,
    LOG_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
)
from src.dataset import GeoTiffPatchDataset
from src.model import build_unet, get_device
from src.transforms import get_val_transform

# Cấu hình font chữ hỗ trợ tiếng Việt trên Windows
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

FOREGROUND_CLASSES: List[int] = [1, 2, 3, 4, 5, 6]
FOREGROUND_NAMES: List[str] = [CLASS_NAMES[c] for c in FOREGROUND_CLASSES]


# ---------------------------------------------------------------------------
# Các hàm tính toán chỉ số đánh giá (Evaluation Metrics)
# ---------------------------------------------------------------------------
def compute_metrics_from_confusion_matrix(cm_6x6: np.ndarray) -> Dict[str, Any]:
    """Tính toán toàn bộ chỉ số đánh giá từ ma trận nhầm lẫn 6x6 (lớp 1–6).

    Args:
        cm_6x6: Ma trận kích thước [6, 6], hàng là nhãn thực tế, cột là nhãn dự đoán.

    Returns:
        Dict chứa OA, mIoU, Macro F1, Cohen's Kappa và chi tiết từng lớp.
    """
    n_classes = len(FOREGROUND_CLASSES)
    total_pixels = int(cm_6x6.sum())

    if total_pixels == 0:
        return {
            "overall_accuracy": 0.0,
            "mean_iou": 0.0,
            "macro_f1": 0.0,
            "cohen_kappa": 0.0,
            "per_class": {},
        }

    # 1. Overall Accuracy (OA)
    total_correct = int(np.trace(cm_6x6))
    oa = total_correct / total_pixels

    # 2. Per-class Precision, Recall, F1, IoU
    per_class: Dict[int, Dict[str, float]] = {}
    ious: List[float] = []
    f1s: List[float] = []

    for idx, c in enumerate(FOREGROUND_CLASSES):
        tp = int(cm_6x6[idx, idx])
        fp = int(cm_6x6[:, idx].sum() - tp)
        fn = int(cm_6x6[idx, :].sum() - tp)
        n_true = int(cm_6x6[idx, :].sum())
        n_pred = int(cm_6x6[:, idx].sum())

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        ious.append(iou)
        f1s.append(f1)

        per_class[c] = {
            "name": CLASS_NAMES[c],
            "iou": float(iou),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "pixel_count_true": n_true,
            "pixel_count_pred": n_pred,
        }

    # 3. Macro metrics
    mean_iou = float(np.mean(ious))
    macro_f1 = float(np.mean(f1s))

    # 4. Cohen's Kappa
    po = oa
    row_sums = cm_6x6.sum(axis=1).astype(np.float64)
    col_sums = cm_6x6.sum(axis=0).astype(np.float64)
    pe = float(np.sum(row_sums * col_sums) / (total_pixels ** 2))
    cohen_kappa = float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 0 else 0.0

    return {
        "overall_accuracy": float(oa),
        "mean_iou": float(mean_iou),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(cohen_kappa),
        "total_evaluated_pixels": total_pixels,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Các hàm vẽ biểu đồ trực quan
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm_6x6: np.ndarray,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Vẽ Heatmap Ma trận nhầm lẫn 6x6 và lưu ra file ảnh."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    if normalize:
        row_sums = cm_6x6.sum(axis=1, keepdims=True).astype(np.float64)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_6x6 / row_sums
        fmt = ".1%"
    else:
        cm_display = cm_6x6
        fmt = ",d"

    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tỷ lệ phân loại (%)" if normalize else "Số lượng pixel", fontsize=11)

    tick_marks = np.arange(len(FOREGROUND_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(FOREGROUND_NAMES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(FOREGROUND_NAMES, fontsize=10)

    # Hiển thị số liệu trong từng ô
    thresh = cm_display.max() / 2.0
    for i in range(cm_6x6.shape[0]):
        for j in range(cm_6x6.shape[1]):
            val_text = f"{cm_display[i, j]:.1%}" if normalize else f"{cm_6x6[i, j]:,}"
            color = "white" if cm_display[i, j] > thresh else "black"
            ax.text(j, i, val_text, ha="center", va="center", color=color, fontsize=9, fontweight="semibold")

    ax.set_title("Ma trận Nhầm lẫn (Confusion Matrix) — Golden Test", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Nhãn Thực tế (Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nhãn Mô hình Dự đoán (Prediction)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Đã lưu biểu đồ Ma trận Nhầm lẫn: {output_path}")


def plot_per_class_iou(
    per_class_dict: Dict[int, Dict[str, float]],
    mean_iou: float,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ cột thể hiện IoU của 6 lớp đất có nhúng mã màu GIS chuẩn."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    classes = [1, 2, 3, 4, 5, 6]
    names = [CLASS_NAMES[c] for c in classes]
    iou_pcts = [per_class_dict[c]["iou"] * 100 for c in classes]

    # Lấy màu GIS từ CLASS_COLORS
    bar_colors = [
        f"#{CLASS_COLORS[c][0]:02x}{CLASS_COLORS[c][1]:02x}{CLASS_COLORS[c][2]:02x}"
        for c in classes
    ]

    bars = ax.bar(names, iou_pcts, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.6)

    # Vẽ đường trung bình mIoU
    line = ax.axhline(mean_iou * 100, color="red", linestyle="--", linewidth=1.5, label=f"Mean IoU: {mean_iou * 100:.2f}%")

    # Hiển thị số liệu trên đỉnh mỗi cột
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Intersection over Union — IoU (%)", fontsize=11, fontweight="bold")
    ax.set_title("Chỉ số IoU Từng Loại Đất trên Tập Golden Test (Giao Thủy)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()


    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Đã lưu biểu đồ IoU từng lớp: {output_path}")


def plot_training_curves(csv_path: Path, output_path: Path) -> None:
    """Vẽ đồ thị Loss và mIoU/OA qua các epoch từ file training_log.csv."""
    if not csv_path.exists():
        logger.warning(f"Không tìm thấy file log: {csv_path}. Bỏ qua vẽ training curves.")
        return

    epochs: List[int] = []
    tr_loss: List[float] = []
    val_loss: List[float] = []
    val_mIoU: List[float] = []
    val_oa: List[float] = []
    is_bests: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_mIoU.append(float(row["val_mIoU"]) * 100)
            val_oa.append(float(row["val_OA"]) * 100)
            is_bests.append(int(row.get("is_best", 0)))

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Đồ thị 1: Loss
    ax1.plot(epochs, tr_loss, label="Train Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.0)
    ax1.axvline(x=5, color="gray", linestyle=":", label="Unfreeze Encoder (Epoch 6)")
    ax1.set_title("Đường cong Hàm Mất mát (Loss Curves)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Đồ thị 2: mIoU & OA
    ax2.plot(epochs, val_mIoU, label="Val mIoU (%)", color="#2ca02c", linewidth=2.0)
    ax2.plot(epochs, val_oa, label="Val OA (%)", color="#9467bd", linewidth=2.0)
    
    # Đánh dấu đỉnh Best mIoU
    best_idx = int(np.argmax(val_mIoU))
    ax2.scatter(
        [epochs[best_idx]],
        [val_mIoU[best_idx]],
        color="red",
        s=80,
        zorder=5,
        label=f"Best mIoU: {val_mIoU[best_idx]:.2f}% (Epoch {epochs[best_idx]})",
    )

    ax2.set_title("Chỉ số Đánh giá (Validation Metrics)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Tỷ lệ (%)", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Đã lưu biểu đồ Quá trình Huấn luyện: {output_path}")


# ---------------------------------------------------------------------------
# Hàm thực thi đánh giá chính (Main Evaluation Engine)
# ---------------------------------------------------------------------------
def evaluate_model(
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pth",
    patches_dir: Path = PROCESSED_DIR / "patches",
    batch_size: int = 8,
    output_dir: Path = LOG_DIR,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Thực hiện đánh giá toàn diện mô hình checkpoint trên tập Golden Test.

    Args:
        checkpoint_path: Đường dẫn file trọng số best_model.pth.
        patches_dir: Thư mục chứa dữ liệu patches.
        batch_size: Batch size khi chạy inference test.
        output_dir: Thư mục xuất các file báo cáo và biểu đồ.
        device: Thiết bị chạy (GPU hoặc CPU).

    Returns:
        Dict tổng hợp toàn bộ kết quả đánh giá.
    """
    device = device or get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path.resolve()}")

    # 1. Nạp mô hình & Checkpoint
    logger.info(f"Nạp trọng số checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Đọc cấu hình từ checkpoint nếu có
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    encoder_name = model_cfg.get("encoder_name", "resnet34")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Tự động phát hiện số kênh đầu vào từ trọng số conv1 (4 kênh hoặc 6 kênh)
    conv1_w = state_dict.get("encoder.conv1.weight", None)
    if conv1_w is not None:
        in_channels = int(conv1_w.shape[1])
    else:
        in_channels = model_cfg.get("in_channels", 4)

    model = build_unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # Không cần tải lại từ mạng vì sẽ nạp từ checkpoint
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        f"Mô hình nạp thành công: {encoder_name}, in_channels={in_channels} "
        f"(Epoch đã lưu: {checkpoint.get('epoch', 'N/A')})"
    )

    # 2. Khởi tạo Test Dataset & DataLoader
    test_ds = GeoTiffPatchDataset(
        split="test",
        patches_dir=patches_dir,
        transform=get_val_transform(),
        normalize=True,
        add_indices=(in_channels == 6),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"Tập Golden Test: {len(test_ds)} patch ({len(test_loader)} batches)")

    # 3. Chạy Inference và Tích lũy Ma trận Nhầm lẫn
    cm_full = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Đang đánh giá Golden Test", unit="batch"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().reshape(-1)
            targets = masks.numpy().reshape(-1)

            valid = (targets >= 0) & (targets < NUM_CLASSES) & (preds >= 0) & (preds < NUM_CLASSES)
            np.add.at(cm_full, (targets[valid], preds[valid]), 1)

    # Cắt ma trận nhầm lẫn lớp 1–6 (bỏ qua lớp 0 Background)
    cm_6x6 = cm_full[1:, 1:]

    # 4. Tính toán toàn bộ chỉ số
    metrics = compute_metrics_from_confusion_matrix(cm_6x6)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch", None)
    metrics["checkpoint_path"] = str(checkpoint_path.resolve())
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["confusion_matrix_6x6"] = cm_6x6.tolist()

    # 5. Xuất các file báo cáo và biểu đồ
    # a. File JSON
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu báo cáo JSON: {json_path}")

    # b. Biểu đồ Confusion Matrix
    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm_6x6, cm_plot_path, normalize=True)

    # c. Biểu đồ Per-Class IoU
    iou_plot_path = output_dir / "per_class_iou.png"
    plot_per_class_iou(metrics["per_class"], metrics["mean_iou"], iou_plot_path)

    # d. Biểu đồ Training Curves
    curves_plot_path = output_dir / "training_curves.png"
    csv_log_path = output_dir / "training_log.csv"
    plot_training_curves(csv_log_path, curves_plot_path)

    # 6. In Báo cáo Tổng kết ra Console
    print("\n" + "=" * 80)
    print("  BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN MÔ HÌNH TRÊN TẬP GOLDEN TEST (BƯỚC 2.6)")
    print("=" * 80)
    print(f"  Checkpoint nạp     : {checkpoint_path.name} (Epoch {metrics['checkpoint_epoch']})")
    print(f"  Độ chính xác tổng thể (Overall Accuracy) : {metrics['overall_accuracy'] * 100:6.2f}%")
    print(f"  Mean IoU (Macro 6 lớp mục tiêu)          : {metrics['mean_iou'] * 100:6.2f}%")
    print(f"  Macro F1-Score                           : {metrics['macro_f1'] * 100:6.2f}%")
    print(f"  Hệ số Cohen's Kappa                      : {metrics['cohen_kappa']:6.4f}")
    print("-" * 80)
    print(f"  {'STT':<4} {'Loại Đất':<30} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 80)

    for c in FOREGROUND_CLASSES:
        item = metrics["per_class"][c]
        print(
            f"  {c:<4} {item['name']:<30} "
            f"{item['iou'] * 100:>7.2f}% "
            f"{item['precision'] * 100:>9.2f}% "
            f"{item['recall'] * 100:>9.2f}% "
            f"{item['f1_score'] * 100:>9.2f}%"
        )

    print("=" * 80)
    print(f"  Các file kết quả đã xuất tại thư mục: {output_dir}")
    print(f"    - Báo cáo số liệu JSON : {json_path.name}")
    print(f"    - Heatmap Ma trận nhầm lẫn : {cm_plot_path.name}")
    print(f"    - Biểu đồ IoU từng lớp : {iou_plot_path.name}")
    print(f"    - Đồ thị huấn luyện   : {curves_plot_path.name}")
    print("=" * 80 + "\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình U-Net trên tập Golden Test.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model.pth"),
        help="Đường dẫn tới file checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LOG_DIR),
        help="Thư mục xuất báo cáo và biểu đồ.",
    )
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
