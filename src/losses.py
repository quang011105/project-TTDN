"""
Hàm Mất mát Kết hợp (Combo Loss = Weighted Cross-Entropy + Dice Loss) cho Phân vùng Viễn thám.

Công thức tổng quát:
    L = lambda_CE * L_WeightedCE + lambda_Dice * L_Dice

Thành phần:
    1. Weighted Cross-Entropy Loss:
       - Phạt sai số phân loại từng pixel (pixel-wise).
       - Tích hợp trọng số Median Frequency Balancing (MFB) từ Giai đoạn 1 để khắc phục
         mất cân bằng lớp nghiêm trọng (Lúa chiếm ~40% vs Rừng ngập mặn / Đồng muối ~1-6%).
       - Bắt buộc ignore_index=0 (lớp Background/Nodata) không tham gia tính loss & gradient.
    2. Multi-class Dice Loss:
       - Tối ưu hóa trực tiếp độ trùng khớp vùng (region overlap / intersection over union).
       - Chỉ tính trên 6 lớp quy hoạch mục tiêu (lớp 1–6), bỏ qua lớp 0.
       - Hệ số làm mịn smooth=1.0 chống chia cho 0 khi lớp không xuất hiện trong patch.

Cách dùng điển hình::

    from src.losses import ComboLoss, load_class_weights

    # Nạp trọng số MFB
    weights = load_class_weights()

    # Khởi tạo hàm loss
    criterion = ComboLoss(
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=weights,
        ignore_index=0,
    )

    loss = criterion(logits, targets)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    import segmentation_models_pytorch as smp

    _SMP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SMP_AVAILABLE = False

from src.config import PROCESSED_DIR

DEFAULT_CLASS_WEIGHTS_PATH: Path = PROCESSED_DIR / "class_weights.json"
DEFAULT_FOREGROUND_CLASSES: List[int] = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Tiện ích nạp trọng số MFB từ file JSON
# ---------------------------------------------------------------------------
def load_class_weights(
    weights_path: Union[Path, str] = DEFAULT_CLASS_WEIGHTS_PATH,
    use_normalized: bool = False,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Nạp vector trọng số MFB 7 lớp từ file class_weights.json.

    Args:
        weights_path: Đường dẫn tới file JSON đã tạo ở Bước 1.3.
        use_normalized: Nếu True, nạp 'normalized_weights' (tổng = 1 hoặc trung bình = 1).
                        Nếu False, nạp 'weights' gốc [0.0, 0.33, 0.77, 0.52, 2.18, 1.42, 7.59].
        device: Thiết bị đích (CPU hoặc CUDA) để đặt Tensor.

    Returns:
        FloatTensor 1 chiều kích thước [7].
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số lớp: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = "normalized_weights" if use_normalized else "weights"
    if key not in data:
        raise KeyError(f"Không tìm thấy khóa '{key}' trong file {path}")

    weights_list: List[float] = [float(w) for w in data[key]]
    tensor_weights = torch.tensor(weights_list, dtype=torch.float32)

    if device is not None:
        tensor_weights = tensor_weights.to(device)

    return tensor_weights


# ---------------------------------------------------------------------------
# Lớp ComboLoss kết hợp Weighted Cross-Entropy & Dice Loss
# ---------------------------------------------------------------------------
class ComboLoss(nn.Module):
    """Hàm Loss kết hợp giữa Weighted Cross-Entropy và Multi-class Dice Loss.

    Args:
        ce_weight: Trọng số lambda_CE cho thành phần Cross-Entropy (mặc định: 1.0).
        dice_weight: Trọng số lambda_Dice cho thành phần Dice Loss (mặc định: 1.0).
        class_weights: Vector trọng số MFB [7] cho CrossEntropy (hoặc None).
        ignore_index: Chỉ số lớp bỏ qua gradient (mặc định: 0 cho Background).
        dice_smooth: Hệ số làm mịn của Dice Loss (mặc định: 1.0).
        classes: Danh sách các lớp tính Dice (mặc định: [1, 2, 3, 4, 5, 6]).
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Union[Tensor, Sequence[float]]] = None,
        ignore_index: int = 0,
        dice_smooth: float = 1.0,
        classes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.ce_weight: float = float(ce_weight)
        self.dice_weight: float = float(dice_weight)
        self.ignore_index: int = int(ignore_index)
        self.classes: List[int] = classes if classes is not None else DEFAULT_FOREGROUND_CLASSES

        # Đăng ký class_weights dạng buffer để tự động chuyển theo thiết bị của module (.to(device))
        if class_weights is not None:
            if not isinstance(class_weights, Tensor):
                tensor_w = torch.tensor(class_weights, dtype=torch.float32)
            else:
                tensor_w = class_weights.clone().detach().float()
            self.register_buffer("weights", tensor_w)
        else:
            self.weights = None

        # Khởi tạo CrossEntropyLoss
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=self.weights,
            ignore_index=self.ignore_index,
        )

        # Khởi tạo DiceLoss (smp hoặc fallback)
        if _SMP_AVAILABLE:
            self.dice_loss_fn = smp.losses.DiceLoss(
                mode="multiclass",
                classes=self.classes,
                from_logits=True,
                ignore_index=self.ignore_index,
                smooth=dice_smooth,
                eps=1e-7,
            )
        else:  # pragma: no cover
            self.dice_loss_fn = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Tính tổng loss kết hợp.

        Args:
            logits: Logits dự đoán từ mô hình kích thước [B, C, H, W].
            targets: Mask nhãn thực tế [B, H, W] với kiểu torch.long.

        Returns:
            Scalar Tensor đại diện cho tổng loss.
        """
        components = self.forward_with_components(logits, targets)
        return components["loss"]

    def forward_with_components(self, logits: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """Tính loss và trả về chi tiết từng thành phần để phục vụ logging.

        Args:
            logits: Logits dự đoán [B, C, H, W].
            targets: Mask nhãn thực tế [B, H, W].

        Returns:
            Dict chứa::
                - "loss": Tổng loss kết hợp.
                - "ce_loss": Thành phần Weighted Cross-Entropy.
                - "dice_loss": Thành phần Dice Loss.
        """
        # Cập nhật buffer trọng số vào hàm CE nếu có
        if self.weights is not None and self.ce_loss_fn.weight is not self.weights:
            self.ce_loss_fn.weight = self.weights

        # 1. Weighted Cross-Entropy Loss
        # Xử lý trường hợp biên: Nếu batch chỉ toàn pixel ignore_index (không có foreground)
        valid_pixels = (targets != self.ignore_index).sum()
        if valid_pixels == 0:
            ce_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            ce_loss = self.ce_loss_fn(logits, targets)

        # 2. Dice Loss
        if self.dice_loss_fn is not None:
            dice_loss = self.dice_loss_fn(logits, targets)
        else:
            dice_loss = self._manual_dice_loss(logits, targets)

        # 3. Tổng hợp
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "dice_loss": dice_loss,
        }

    def _manual_dice_loss(self, logits: Tensor, targets: Tensor, eps: float = 1e-7) -> Tensor:
        """Hàm Dice Loss tự sinh dự phòng trong trường hợp không có SMP."""
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        B, C, H, W = probs.shape
        dice_scores: List[Tensor] = []

        for c in self.classes:
            target_c = (targets == c).float()  # [B, H, W]
            prob_c = probs[:, c, :, :]         # [B, H, W]

            intersection = (prob_c * target_c).sum(dim=(1, 2))
            cardinality = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice_c = (2.0 * intersection + 1.0) / (cardinality + 1.0 + eps)
            dice_scores.append(dice_c.mean())

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


# ---------------------------------------------------------------------------
# Factory tiện ích
# ---------------------------------------------------------------------------
def get_loss_fn(
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    weights_path: Union[Path, str] = DEFAULT_CLASS_WEIGHTS_PATH,
    use_normalized: bool = False,
    device: Optional[torch.device] = None,
) -> ComboLoss:
    """Hàm tiện ích khởi tạo nhanh ComboLoss kèm nạp trọng số MFB tự động."""
    class_weights = None
    if Path(weights_path).exists():
        class_weights = load_class_weights(
            weights_path=weights_path,
            use_normalized=use_normalized,
            device=device,
        )

    loss_fn = ComboLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        class_weights=class_weights,
        ignore_index=0,
    )
    if device is not None:
        loss_fn = loss_fn.to(device)

    return loss_fn


# ---------------------------------------------------------------------------
# CLI kiểm tra nhanh
# ---------------------------------------------------------------------------
def _cli_check() -> None:
    """Chạy thử nghiệm kiểm tra tính toán loss từ dòng lệnh."""
    print("=" * 60)
    print("  CHECK LOSS FUNCTIONS: WEIGHTED CROSS-ENTROPY + DICE LOSS")
    print("=" * 60)

    loss_fn = get_loss_fn()
    print(f"  ComboLoss loaded: ce_weight={loss_fn.ce_weight}, dice_weight={loss_fn.dice_weight}")
    if loss_fn.weights is not None:
        print(f"  Class weights (MFB): {loss_fn.weights.tolist()}")

    # Chạy thử forward pass với dummy logits [2, 7, 256, 256] và target [2, 256, 256]
    dummy_logits = torch.randn(2, 7, 256, 256, requires_grad=True)
    dummy_targets = torch.randint(0, 7, (2, 256, 256), dtype=torch.long)

    components = loss_fn.forward_with_components(dummy_logits, dummy_targets)
    print(f"\n  [Forward Loss Components]:")
    print(f"  Total Combo Loss   : {components['loss'].item():.4f}")
    print(f"  Weighted CE Loss   : {components['ce_loss'].item():.4f}")
    print(f"  Dice Loss          : {components['dice_loss'].item():.4f}")

    # Chạy backward pass
    components["loss"].backward()
    grad_norm = dummy_logits.grad.norm().item()
    print(f"  Backward gradient norm: {grad_norm:.4f}")

    print("\n  Loss function check completed successfully!")


if __name__ == "__main__":
    _cli_check()
