"""
Vòng lặp Huấn luyện (Training Loop) cho Mô hình U-Net Phân vùng Viễn thám.

Tích hợp các kỹ thuật huấn luyện hiện đại:
    - Optimizer: AdamW với weight decay chống overfitting.
    - Scheduler: OneCycleLR với warm-up tự động và cosine annealing.
    - Mixed Precision (AMP): Tăng tốc và tiết kiệm VRAM trên GPU CUDA (tự tắt trên CPU).
    - Freeze/Unfreeze Encoder: Epoch 1–5 đóng băng backbone ResNet34, Epoch 6+ fine-tune end-to-end.
    - Gradient Accumulation: Hỗ trợ tích lũy gradient khi batch size nhỏ.
    - Early Stopping: Theo dõi val_mIoU (lớp 1–6) trên tập Test, tự động dừng nếu không cải thiện.
    - Thanh tiến trình trực quan: tqdm hiển thị chi tiết Loss, mIoU, LR từng batch.
    - Logging: Lưu lịch sử huấn luyện per-epoch ra CSV (outputs/logs/training_log.csv).
    - Checkpoint: Lưu trọng số mô hình tốt nhất (best_model.pth) và mới nhất (last_model.pth).

Cách chạy:
    python -m src.trainer --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cấu hình UTF-8 cho console Windows để tránh UnicodeEncodeError charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


from src.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    LOG_DIR,
    NUM_CLASSES,
    ensure_dirs,
)
from src.dataset import get_dataloaders
from src.losses import ComboLoss, get_loss_fn
from src.model import (
    build_unet,
    count_parameters,
    freeze_encoder,
    get_device,
    unfreeze_encoder,
)
from src.transforms import get_train_transform, get_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thiết lập Random Seed để Tái lập Kết quả (Reproducibility)
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Lớp Trainer Quản lý Vòng lặp Huấn luyện
# ---------------------------------------------------------------------------
class Trainer:
    """Quản lý toàn bộ quy trình huấn luyện, đánh giá và lưu trữ mô hình."""

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        self.config = config
        self.device = device or get_device()

        # Thiết lập seed
        train_cfg = self.config.get("training", {})
        set_seed(train_cfg.get("seed", 42))

        # Đảm bảo các thư mục đầu ra tồn tại
        self.checkpoint_dir = Path(self.config.get("output", {}).get("checkpoint_dir", CHECKPOINT_DIR))
        self.log_dir = Path(self.config.get("output", {}).get("log_dir", LOG_DIR))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Khởi tạo DataLoaders
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        self.batch_size = data_cfg.get("batch_size", 8)
        self.num_workers = data_cfg.get("num_workers", 0)
        patches_dir = Path(data_cfg.get("patches_dir", "data/processed/patches"))

        in_channels = model_cfg.get("in_channels", 4)
        add_indices = data_cfg.get("add_indices", (in_channels == 6))

        logger.info(
            f"Khởi tạo DataLoader từ: {patches_dir.resolve()} (batch_size={self.batch_size}, "
            f"in_channels={in_channels}, add_indices={add_indices})"
        )
        self.train_loader, self.test_loader = get_dataloaders(
            patches_dir=patches_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_transform=get_train_transform(),
            val_transform=get_val_transform(),
            add_indices=add_indices,
            pin_memory=(self.device.type == "cuda"),
        )
        logger.info(f"Số batch Train: {len(self.train_loader)} | Số batch Val: {len(self.test_loader)}")

        # 2. Khởi tạo Mô hình
        self.model = build_unet(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            in_channels=in_channels,
            num_classes=model_cfg.get("num_classes", 7),
        ).to(self.device)

        # 3. Khởi tạo Hàm Loss
        loss_cfg = self.config.get("loss", {})
        self.criterion = get_loss_fn(
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            weights_path=loss_cfg.get("class_weights_path", "data/processed/class_weights.json"),
            device=self.device,
        )

        # 4. Cấu hình Huấn luyện
        self.max_epochs = train_cfg.get("max_epochs", 60)
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.freeze_encoder_epochs = int(train_cfg.get("freeze_encoder_epochs", 5))
        self.accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))

        # Mixed Precision (AMP) — chỉ bật khi có GPU CUDA
        use_amp = bool(train_cfg.get("mixed_precision", True)) and (self.device.type == "cuda")
        self.use_amp = use_amp
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


        # Khởi tạo Optimizer với Differential Learning Rate (Backbone học chậm hơn 10x để bảo vệ pretrained weights)
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        self.lr_encoder = self.lr * 0.1
        self.lr_decoder = self.lr

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_encoder},
                {"params": decoder_params, "lr": self.lr_decoder},
            ],
            weight_decay=self.weight_decay,
        )

        # Khởi tạo Scheduler: OneCycleLR với max_lr riêng cho từng nhóm tham số
        total_steps = max(1, self.max_epochs * len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.lr_encoder, self.lr_decoder],
            total_steps=total_steps,
            pct_start=0.2,            # 20% epochs warm-up
            div_factor=10.0,          # lr khởi đầu = max_lr / 10
            final_div_factor=100.0,   # lr cuối = max_lr / 1000
        )

        # Khởi tạo file log CSV
        self.csv_path = self.log_dir / "training_log.csv"
        self._init_csv_log()

        # Trạng thái Early Stopping & Quản lý Freeze
        self.best_mIoU = -1.0
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self._encoder_frozen: Optional[bool] = None

    def _init_csv_log(self) -> None:
        """Tạo header cho file log CSV."""
        headers = [
            "epoch",
            "train_loss",
            "train_ce_loss",
            "train_dice_loss",
            "val_loss",
            "val_ce_loss",
            "val_dice_loss",
            "val_mIoU",
            "val_OA",
            "iou_c1_lua",
            "iou_c2_khudancu",
            "iou_c3_thuysan",
            "iou_c4_rungngapman",
            "iou_c5_caylaunam",
            "iou_c6_dongmuoi",
            "lr",
            "is_best",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    # ----------------------------------------------------------------------
    # Huấn luyện 1 Epoch
    # ----------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Thực hiện huấn luyện 1 epoch trên tập Train."""
        self.model.train()

        # Áp dụng chiến lược Freeze/Unfreeze Encoder (chỉ chuyển đổi & ghi log khi thay đổi trạng thái)
        if epoch <= self.freeze_encoder_epochs:
            if not self._encoder_frozen:
                freeze_encoder(self.model)
                self._encoder_frozen = True
        else:
            if self._encoder_frozen is not False:
                unfreeze_encoder(self.model)
                self._encoder_frozen = False

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        n_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        # Thanh tiến trình tqdm cho tập Train
        status = "Frozen" if epoch <= self.freeze_encoder_epochs else "Fine-tune"
        pbar = tqdm(
            enumerate(self.train_loader),
            total=n_batches,
            desc=f"Epoch {epoch:02d}/{self.max_epochs:02d} [Train-{status}]",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, masks) in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Mixed Precision Forward
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                components = self.criterion.forward_with_components(logits, masks)
                loss = components["loss"]
                # Chuẩn hoá loss khi dùng Gradient Accumulation
                loss_scaled = loss / self.accumulation_steps

            # Backward pass với GradScaler
            self.scaler.scale(loss_scaled).backward()

            # Cập nhật tham số khi tích lũy đủ bước hoặc ở batch cuối
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                # Unscale và Gradient Clipping để ngăn chặn bùng nổ gradient
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item()
            total_ce_loss += components["ce_loss"].item()
            total_dice_loss += components["dice_loss"].item()

            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{components['ce_loss'].item():.3f}",
                "dice": f"{components['dice_loss'].item():.3f}",
                "lr": f"{current_lr:.2e}",
            })

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_ce_loss": total_ce_loss / max(n_batches, 1),
            "train_dice_loss": total_dice_loss / max(n_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
        }

    # ----------------------------------------------------------------------
    # Đánh giá 1 Epoch trên Tập Test/Validation
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, Any]:
        """Thực hiện đánh giá trên tập Test/Val: Loss, mIoU (lớp 1–6), Overall Accuracy."""
        self.model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        n_batches = len(self.test_loader)

        # Bộ đếm Confusion Matrix cho lớp 1–6
        # Kích thước ma trận: [7, 7] để map trực tiếp nhãn 0..6
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        val_pbar = tqdm(
            self.test_loader,
            desc=f"Epoch {epoch:02d}/{self.max_epochs:02d} [Val]",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for images, masks in val_pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                components = self.criterion.forward_with_components(logits, masks)

            total_loss += components["loss"].item()
            total_ce_loss += components["ce_loss"].item()
            total_dice_loss += components["dice_loss"].item()

            preds = torch.argmax(logits, dim=1).cpu().numpy().reshape(-1)
            targets_np = masks.cpu().numpy().reshape(-1)

            # Chỉ tích lũy các pixel hợp lệ trong dải nhãn [0, 6]
            mask_valid = (targets_np >= 0) & (targets_np < NUM_CLASSES) & (preds >= 0) & (preds < NUM_CLASSES)
            np.add.at(confusion_matrix, (targets_np[mask_valid], preds[mask_valid]), 1)

            val_pbar.set_postfix({"loss": f"{components['loss'].item():.4f}"})

        # --- Tính toán chỉ số đánh giá (Bỏ lớp 0 — chỉ tính lớp 1–6) ---
        per_class_iou: Dict[int, float] = {}
        foreground_classes = [1, 2, 3, 4, 5, 6]

        for c in foreground_classes:
            tp = float(confusion_matrix[c, c])
            fp = float(confusion_matrix[:, c].sum() - tp)
            fn = float(confusion_matrix[c, :].sum() - tp)
            denominator = tp + fp + fn
            per_class_iou[c] = tp / denominator if denominator > 0 else 0.0

        val_mIoU = float(np.mean([per_class_iou[c] for c in foreground_classes]))

        # Overall Accuracy: pixel đúng trên tổng số pixel hợp lệ (lớp 1–6)
        fg_correct = sum(float(confusion_matrix[c, c]) for c in foreground_classes)
        fg_total = sum(float(confusion_matrix[c, :].sum()) for c in foreground_classes)
        val_oa = fg_correct / fg_total if fg_total > 0 else 0.0

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_ce_loss": total_ce_loss / max(n_batches, 1),
            "val_dice_loss": total_dice_loss / max(n_batches, 1),
            "val_mIoU": val_mIoU,
            "val_OA": val_oa,
            "per_class_iou": per_class_iou,
        }

    # ----------------------------------------------------------------------
    # Lưu Checkpoint
    # ----------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_mIoU: float, is_best: bool = False) -> None:
        """Lưu trạng thái huấn luyện ra file .pth."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mIoU": self.best_mIoU,
            "val_mIoU": val_mIoU,
            "config": self.config,
        }

        # Lưu last checkpoint
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(checkpoint_data, last_path)

        # Lưu best checkpoint nếu đạt kỷ lục mIoU mới
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_data, best_path)
            logger.info(f"[*] Đã lưu kỷ lục mới -> best_model.pth (mIoU: {val_mIoU * 100:.2f}%)")

    # ----------------------------------------------------------------------
    # Vòng lặp Huấn luyện Toàn trình (Full Loop)
    # ----------------------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        """Khởi chạy toàn bộ vòng lặp huấn luyện đến khi kết thúc hoặc Early Stopping."""
        print("=" * 80)
        print("  BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH U-NET (RESNET34) CHO PHÂN VÙNG GIAO THỦY")
        print("=" * 80)
        params = count_parameters(self.model)
        print(f"  Thiết bị chạy      : {self.device.type.upper()} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        print(f"  Mixed Precision    : {'BẬT (AMP FP16)' if self.use_amp else 'TẮT (FP32)'}")
        print(f"  Tổng tham số       : {params['total']:,} (Encoder: 21.2M, Decoder: 3.1M)")
        in_ch = self.model.encoder.conv1.weight.shape[1]
        ch_desc = "6 kênh: B2, B3, B4, B8, NDVI, NDWI" if in_ch == 6 else f"{in_ch} kênh: B2, B3, B4, B8"
        patch_sz = self.config.get("data", {}).get("patch_size", 256)
        print(f"  Kích thước patch   : {patch_sz}x{patch_sz} ({ch_desc})")
        print(f"  Batch size         : {self.batch_size} (Train: {len(self.train_loader)} batches | Val: {len(self.test_loader)} batches)")
        print(f"  Số Epochs tối đa   : {self.max_epochs} (Freeze Encoder 1–{self.freeze_encoder_epochs})")
        print(f"  Early Stopping     : Patience = {self.early_stopping_patience} epochs (theo dõi val_mIoU)")
        print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # 1. Huấn luyện
            train_metrics = self.train_epoch(epoch)

            # 2. Đánh giá
            val_metrics = self.validate_epoch(epoch)

            val_mIoU = val_metrics["val_mIoU"]
            val_oa = val_metrics["val_OA"]
            is_best = val_mIoU > self.best_mIoU

            # 3. Cập nhật Best mIoU & Early Stopping
            marker = ""
            if is_best:
                self.best_mIoU = val_mIoU
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                marker = "[BEST]"
            else:
                self.epochs_without_improvement += 1

            # 4. Lưu Checkpoint
            self.save_checkpoint(epoch, val_mIoU, is_best=is_best)

            # 5. Ghi CSV Log
            iou_dict = val_metrics["per_class_iou"]
            row = [
                epoch,
                f"{train_metrics['train_loss']:.4f}",
                f"{train_metrics['train_ce_loss']:.4f}",
                f"{train_metrics['train_dice_loss']:.4f}",
                f"{val_metrics['val_loss']:.4f}",
                f"{val_metrics['val_ce_loss']:.4f}",
                f"{val_metrics['val_dice_loss']:.4f}",
                f"{val_mIoU:.4f}",
                f"{val_oa:.4f}",
                f"{iou_dict.get(1, 0.0):.4f}",
                f"{iou_dict.get(2, 0.0):.4f}",
                f"{iou_dict.get(3, 0.0):.4f}",
                f"{iou_dict.get(4, 0.0):.4f}",
                f"{iou_dict.get(5, 0.0):.4f}",
                f"{iou_dict.get(6, 0.0):.4f}",
                f"{train_metrics['lr']:.2e}",
                int(is_best),
            ]
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # 6. In tóm tắt Epoch ra Console
            epoch_elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:02d}/{self.max_epochs:02d} ({epoch_elapsed:.1f}s) | "
                f"TrLoss: {train_metrics['train_loss']:.4f} | "
                f"ValLoss: {val_metrics['val_loss']:.4f} | "
                f"mIoU: {val_mIoU * 100:5.2f}% | "
                f"OA: {val_oa * 100:5.2f}% | "
                f"Best: {self.best_mIoU * 100:5.2f}% {marker}"
            )

            # In chi tiết IoU từng lớp khi có kỷ lục mới
            if is_best:
                iou_strs = [
                    f"{CLASS_NAMES[c]}: {iou_dict.get(c, 0.0) * 100:.1f}%"
                    for c in [1, 2, 3, 4, 5, 6]
                ]
                print(f"       -> IoU: {' | '.join(iou_strs)}")

            # 7. Kiểm tra điều kiện Early Stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"\n[!] Early Stopping kích hoạt: mIoU không cải thiện sau "
                    f"{self.early_stopping_patience} epochs liên tiếp."
                )
                print(f"[*] Mô hình tối ưu nhất tại Epoch {self.best_epoch} với mIoU = {self.best_mIoU * 100:.2f}%")
                break

        total_elapsed = time.time() - start_time
        mins, secs = divmod(int(total_elapsed), 60)
        print("\n" + "=" * 80)
        print(f"  HOÀN THÀNH HUẤN LUYỆN — Tổng thời gian: {mins} phút {secs} giây")
        print(f"  Mô hình tốt nhất  : {self.checkpoint_dir / 'best_model.pth'}")
        print(f"  Kỷ lục Golden mIoU : {self.best_mIoU * 100:.2f}% (Epoch {self.best_epoch})")
        print(f"  File log CSV       : {self.csv_path}")
        print("=" * 80)

        return {
            "best_mIoU": self.best_mIoU,
            "best_epoch": self.best_epoch,
            "best_model_path": str(self.checkpoint_dir / "best_model.pth"),
            "log_csv_path": str(self.csv_path),
        }


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình U-Net viễn thám phân vùng LULC Giao Thủy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Đường dẫn tới file cấu hình YAML (mặc định: configs/train_config.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs tối đa.")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Ghi đè learning rate.")
    parser.add_argument("--patience", type=int, default=None, help="Ghi đè patience Early Stopping.")
    parser.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Không tìm thấy file cấu hình: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ghi đè cấu hình từ command-line nếu có
    if args.epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.patience is not None:
        config.setdefault("training", {})["early_stopping_patience"] = args.patience
    if args.no_amp:
        config.setdefault("training", {})["mixed_precision"] = False

    trainer = Trainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
