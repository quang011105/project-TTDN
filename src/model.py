"""
Kiến trúc Mô hình U-Net với Backbone ResNet34 cho Phân vùng Ảnh Vệ tinh 4 kênh.

Sử dụng thư viện ``segmentation_models_pytorch`` (SMP) để xây dựng mạng Semantic Segmentation:
    - Backbone (Encoder): ResNet34 được tiền huấn luyện trên tập ImageNet (Transfer Learning).
    - Đầu vào: 4 kênh phổ Sentinel-2 (B2 Blue, B3 Green, B4 Red, B8 NIR).
      SMP tự động mở rộng lớp Conv đầu tiên (conv1) từ 3 kênh sang 4 kênh.
    - Đầu ra: Logits thô kích thước [B, 7, H, W] tương ứng với 7 lớp nhãn sử dụng đất (0–6).
      Activation được để là ``None`` để kết hợp trực tiếp với hàm Loss (CrossEntropyLoss, DiceLoss).

Cách dùng điển hình::

    from src.model import build_unet, freeze_encoder, unfreeze_encoder

    # Khởi tạo mô hình 4 kênh
    model = build_unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        num_classes=7,
    )

    # Freeze encoder trong 5 epoch đầu
    freeze_encoder(model)

    # Unfreeze encoder để fine-tune toàn diện
    unfreeze_encoder(model)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    import segmentation_models_pytorch as smp

    _SMP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SMP_AVAILABLE = False

from src.config import (
    DEFAULT_ENCODER,
    DEFAULT_ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory khởi tạo mô hình U-Net
# ---------------------------------------------------------------------------
def build_unet(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: Optional[str] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = IN_CHANNELS,
    num_classes: int = NUM_CLASSES,
    activation: Optional[str] = None,
) -> nn.Module:
    """Khởi tạo mô hình U-Net với backbone tiền huấn luyện cho ảnh viễn thám.

    Args:
        encoder_name: Tên encoder trong SMP (mặc định: ``"resnet34"``).
        encoder_weights: Trọng số tiền huấn luyện (mặc định: ``"imagenet"``,
                         hoặc ``None`` để khởi tạo ngẫu nhiên khi kiểm thử).
        in_channels: Số kênh đầu vào (mặc định: 4 cho Sentinel-2 B2, B3, B4, B8).
        num_classes: Số lớp phân loại (mặc định: 7 lớp 0–6).
        activation: Hàm kích hoạt đầu ra (mặc định: ``None`` để trả về logits thô).

    Returns:
        Mô hình PyTorch ``smp.Unet``.

    Raises:
        ImportError: Nếu thư viện ``segmentation_models_pytorch`` chưa được cài đặt.
    """
    _check_smp()

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
    )

    return model


# ---------------------------------------------------------------------------
# Các tiện ích đóng băng (Freeze) & Mở khóa (Unfreeze) Encoder
# ---------------------------------------------------------------------------
def freeze_encoder(model: nn.Module) -> None:
    """Đóng băng toàn bộ trọng số của Encoder (Backbone).

    Chỉ cho phép cập nhật Decoder và Segmentation Head.
    Thường dùng ở các epoch đầu (ví dụ epoch 1–5) để decoder nhanh chóng hội tụ
    với đặc trưng viễn thám trước khi fine-tune toàn mạng.

    Args:
        model: Mô hình U-Net chứa thuộc tính ``encoder``.
    """
    if not hasattr(model, "encoder"):
        raise AttributeError("Mô hình không chứa thuộc tính 'encoder' để freeze.")

    for param in model.encoder.parameters():
        param.requires_grad = False

    logger.info("Đã đóng băng toàn bộ trọng số của Encoder.")


def unfreeze_encoder(model: nn.Module) -> None:
    """Mở khóa toàn bộ trọng số của Encoder để huấn luyện end-to-end.

    Dùng từ epoch 6 trở đi để fine-tune toàn bộ các tầng mạng trên dữ liệu viễn thám.

    Args:
        model: Mô hình U-Net chứa thuộc tính ``encoder``.
    """
    if not hasattr(model, "encoder"):
        raise AttributeError("Mô hình không chứa thuộc tính 'encoder' để unfreeze.")

    for param in model.encoder.parameters():
        param.requires_grad = True

    logger.info("Đã mở khóa trọng số của Encoder cho fine-tuning.")


# ---------------------------------------------------------------------------
# Thống kê tham số mô hình & Quản lý thiết bị
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Thống kê số lượng tham số của mô hình.

    Args:
        model: Mô hình PyTorch.

    Returns:
        Dict gồm::
            - "total": Tổng số tham số.
            - "trainable": Số tham số đang bật gradient (requires_grad=True).
            - "frozen": Số tham số bị đóng băng (requires_grad=False).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Xác định thiết bị tính toán tối ưu (GPU CUDA hoặc CPU).

    Args:
        device_str: Chuỗi thiết bị tường minh (ví dụ: ``"cuda:0"``, ``"cpu"``).
                    Nếu ``None``, tự động kiểm tra ``torch.cuda.is_available()``.

    Returns:
        ``torch.device`` tương ứng.
    """
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def summary_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 4, 256, 256),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Chạy thử forward pass và tóm tắt thông tin kiến trúc mô hình.

    Args:
        model: Mô hình PyTorch.
        input_size: Kích thước tensor giả lập [B, C, H, W].
        device: Thiết bị chạy thử (mặc định: CPU).

    Returns:
        Dict thông tin chi tiết về kích thước đầu vào, đầu ra, số tham số.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    params = count_parameters(model)

    return {
        "model_class": type(model).__name__,
        "input_shape": tuple(dummy_input.shape),
        "output_shape": tuple(output.shape),
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "frozen_params": params["frozen"],
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _check_smp() -> None:
    """Kiểm tra xem thư viện segmentation_models_pytorch đã sẵn sàng chưa."""
    if not _SMP_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "segmentation_models_pytorch chưa được cài đặt.\n"
            "Chạy: pip install segmentation-models-pytorch>=0.3.3"
        )


# ---------------------------------------------------------------------------
# CLI kiểm tra nhanh
# ---------------------------------------------------------------------------
def _cli_check() -> None:
    """Hàm chạy thử nghiệm kiểm tra kiến trúc mô hình từ dòng lệnh."""
    print("=" * 60)
    print("  CHECK MODEL ARCHITECTURE: U-NET (BACKBONE RESNET34)")
    print("=" * 60)

    model = build_unet(encoder_name="resnet34", encoder_weights="imagenet")
    info = summary_model(model)

    print(f"  Model class    : {info['model_class']} (encoder: resnet34)")
    print(f"  Input shape    : {info['input_shape']} (4 channels: B2, B3, B4, B8)")
    print(f"  Output shape   : {info['output_shape']} (7 classes raw logits)")
    print(f"  Total params   : {info['total_params']:,}")
    print(f"  Trainable      : {info['trainable_params']:,}")

    # Thử nghiệm freeze encoder
    freeze_encoder(model)
    params_frozen = count_parameters(model)
    print(f"\n  [After Freeze Encoder]:")
    print(f"  Trainable      : {params_frozen['trainable']:,} (decoder & head only)")
    print(f"  Frozen         : {params_frozen['frozen']:,} (ResNet34 encoder)")

    # Thử nghiệm unfreeze encoder
    unfreeze_encoder(model)
    params_unfrozen = count_parameters(model)
    print(f"\n  [After Unfreeze Encoder]:")
    print(f"  Trainable      : {params_unfrozen['trainable']:,} (full network)")

    print("\n  Architecture check completed successfully!")


if __name__ == "__main__":
    _cli_check()

