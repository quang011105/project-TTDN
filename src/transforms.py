"""
Pipeline Data Augmentation cho ảnh vệ tinh Sentinel-2 đa kênh (4 kênh).

Sử dụng ``albumentations`` để đồng bộ biến đổi hình học trên cả ảnh lẫn mask.
Biến đổi phổ (GaussNoise, RandomBrightnessContrast) chỉ áp dụng cho ảnh,
**không** làm thay đổi nhãn phân loại trong mask.

Cách dùng điển hình::

    from src.transforms import get_train_transform, get_val_transform
    from src.dataset import get_dataloaders

    train_loader, test_loader = get_dataloaders(
        train_transform=get_train_transform(),
        val_transform=get_val_transform(),
    )

Lưu ý về kênh ảnh
------------------
albumentations mặc định xử lý ảnh 3 kênh (H×W×3). Khi dùng ảnh 4 kênh
(H×W×4) ta truyền ``is_check_shapes=False`` để albumentations không giới hạn số kênh.
Ở đây ảnh được Dataset truyền vào đã ở dạng H×W×4 float32, mask là H×W int32/int64.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import albumentations as A
    from albumentations.core.composition import Compose

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ALBUMENTATIONS_AVAILABLE = False


# Tắt cảnh báo gợi ý chuyển sang Affine của Albumentations 2.x
warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Kiểu trả về của mỗi transform: dict{"image": ndarray, "mask": ndarray}
# ---------------------------------------------------------------------------
TransformOutput = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Các hàm khởi tạo tương thích đa phiên bản Albumentations (1.x & 2.x)
# ---------------------------------------------------------------------------
def _make_gauss_noise(
    var_limit: Tuple[float, float] = (0.001, 0.01),
    p: float = 0.3,
) -> Any:
    """Tạo GaussNoise tương thích cả albumentations 1.x (var_limit) và 2.x (std_range)."""
    sig = inspect.signature(A.GaussNoise.__init__).parameters
    if "std_range" in sig:
        std_min = float(np.sqrt(max(var_limit[0], 0.0)))
        std_max = float(np.sqrt(max(var_limit[1], 0.0)))
        return A.GaussNoise(
            std_range=(std_min, std_max),
            mean_range=(0.0, 0.0),
            p=p,
        )
    return A.GaussNoise(var_limit=var_limit, mean=0.0, p=p)


def _make_shift_scale_rotate(
    shift_limit: float = 0.05,
    scale_limit: float = 0.1,
    rotate_limit: int = 15,
    p: float = 0.3,
) -> Any:
    """Tạo ShiftScaleRotate tương thích cả 1.x (value, mask_value) và 2.x (fill, fill_mask)."""
    sig = inspect.signature(A.ShiftScaleRotate.__init__).parameters
    kwargs: Dict[str, Any] = {
        "shift_limit": shift_limit,
        "scale_limit": scale_limit,
        "rotate_limit": rotate_limit,
        "border_mode": 0,  # cv2.BORDER_CONSTANT — điền 0 cho pixel ngoài viền
        "p": p,
    }
    if "fill" in sig:
        kwargs["fill"] = 0
        kwargs["fill_mask"] = 0
    else:
        kwargs["value"] = 0
        kwargs["mask_value"] = 0
    return A.ShiftScaleRotate(**kwargs)


def _make_grid_distortion(
    num_steps: int = 5,
    distort_limit: float = 0.1,
    p: float = 0.2,
) -> Any:
    """Tạo GridDistortion tương thích cả 1.x và 2.x."""
    sig = inspect.signature(A.GridDistortion.__init__).parameters
    kwargs: Dict[str, Any] = {
        "num_steps": num_steps,
        "distort_limit": distort_limit,
        "border_mode": 0,
        "p": p,
    }
    if "fill" in sig:
        kwargs["fill"] = 0
        kwargs["fill_mask"] = 0
    else:
        kwargs["value"] = 0
        kwargs["mask_value"] = 0
    return A.GridDistortion(**kwargs)


def _make_coarse_dropout(
    num_holes: Tuple[int, int] = (1, 4),
    hole_height: Tuple[int, int] = (16, 32),
    hole_width: Tuple[int, int] = (16, 32),
    p: float = 0.2,
) -> Any:
    """Tạo CoarseDropout tương thích cả 1.x và 2.x."""
    sig = inspect.signature(A.CoarseDropout.__init__).parameters
    if "num_holes_range" in sig:
        return A.CoarseDropout(
            num_holes_range=num_holes,
            hole_height_range=hole_height,
            hole_width_range=hole_width,
            fill=0.0,
            fill_mask=0,
            p=p,
        )
    return A.CoarseDropout(
        min_holes=num_holes[0],
        max_holes=num_holes[1],
        min_height=hole_height[0],
        max_height=hole_height[1],
        min_width=hole_width[0],
        max_width=hole_width[1],
        fill_value=0.0,
        mask_fill_value=0,
        p=p,
    )


# ---------------------------------------------------------------------------
# Hàm tạo transform cho tập TRAIN
# ---------------------------------------------------------------------------
def get_train_transform(
    p_flip_h: float = 0.5,
    p_flip_v: float = 0.5,
    p_rotate90: float = 0.5,
    p_noise: float = 0.3,
    noise_var_limit: Tuple[float, float] = (0.001, 0.01),
    p_shift_scale_rotate: float = 0.3,
    shift_limit: float = 0.05,
    scale_limit: float = 0.1,
    rotate_limit: int = 15,
    seed: Optional[int] = 42,
) -> "Compose":
    """Tạo pipeline augmentation cho tập **train**.

    Biến đổi hình học (đồng bộ ảnh + mask):
        - HorizontalFlip: lật ngang — phù hợp ảnh nadir (chụp thẳng đứng).
        - VerticalFlip: lật dọc — bất biến theo hướng bắc/nam.
        - RandomRotate90: xoay 0°/90°/180°/270° — Sentinel-2 không có hướng cố định.
        - ShiftScaleRotate: dịch, phóng to/thu nhỏ, xoay nhỏ — mô phỏng sai số căn chỉnh.

    Biến đổi phổ (chỉ ảnh, **không** áp dụng mask):
        - GaussNoise: mô phỏng nhiễu cảm biến; phương sai nhỏ để không bóp méo phổ.

    Args:
        p_flip_h: Xác suất lật ngang.
        p_flip_v: Xác suất lật dọc.
        p_rotate90: Xác suất xoay 90°.
        p_noise: Xác suất thêm nhiễu Gaussian.
        noise_var_limit: Khoảng phương sai nhiễu Gaussian (min, max).
        p_shift_scale_rotate: Xác suất ShiftScaleRotate (tắt bằng p=0.0).
        shift_limit: Giới hạn dịch chuyển (tỷ lệ theo kích thước ảnh).
        scale_limit: Giới hạn thay đổi tỷ lệ.
        rotate_limit: Góc xoay tối đa (độ).
        seed: Random seed để tái lập kết quả.

    Returns:
        ``albumentations.Compose`` pipeline.
    """
    _check_albumentations()

    transforms_list = [
        # --- Hình học bất biến nadir ---
        A.HorizontalFlip(p=p_flip_h),
        A.VerticalFlip(p=p_flip_v),
        A.RandomRotate90(p=p_rotate90),
        # --- Hình học nâng cao (tắt bằng p=0.0 nếu cần) ---
        _make_shift_scale_rotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            p=p_shift_scale_rotate,
        ),
        # --- Phổ: nhiễu cảm biến (chỉ ảnh) ---
        _make_gauss_noise(
            var_limit=noise_var_limit,
            p=p_noise,
        ),
    ]

    compose_kwargs: Dict[str, Any] = {
        "additional_targets": {"mask": "mask"},
        "is_check_shapes": False,  # Cho phép ảnh 4 kênh
    }
    sig = inspect.signature(A.Compose.__init__).parameters
    if "seed" in sig and seed is not None:
        compose_kwargs["seed"] = seed

    return A.Compose(transforms_list, **compose_kwargs)


# ---------------------------------------------------------------------------
# Hàm tạo transform cho tập TEST / VALIDATION
# ---------------------------------------------------------------------------
def get_val_transform() -> "Compose":
    """Tạo pipeline augmentation cho tập **test/val** — không biến đổi.

    Tập test cần đánh giá chính xác trên phân bố dữ liệu gốc nên
    không áp dụng bất kỳ biến đổi ngẫu nhiên nào.

    Returns:
        ``albumentations.Compose`` trống — chỉ pass-through ảnh và mask.
    """
    _check_albumentations()
    return A.Compose(
        [],
        additional_targets={"mask": "mask"},
        is_check_shapes=False,
    )


# ---------------------------------------------------------------------------
# Hàm tạo transform mạnh hơn — dùng khi muốn tăng mạnh dữ liệu
# ---------------------------------------------------------------------------
def get_strong_train_transform(seed: Optional[int] = 42) -> "Compose":
    """Pipeline augmentation mạnh hơn — dùng khi tập train ít patch hoặc cần regularization cao.

    Bổ sung thêm so với ``get_train_transform()``:
        - GridDistortion: biến dạng lưới nhỏ — mô phỏng sai số hình học nhẹ.
        - CoarseDropout: che khuất một số vùng nhỏ — buộc model học đặc trưng phân tán.
        - RandomBrightnessContrast: biến đổi độ sáng / tương phản nhẹ trên dải phổ.

    Args:
        seed: Random seed.

    Returns:
        ``albumentations.Compose`` pipeline mạnh.
    """
    _check_albumentations()

    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        _make_shift_scale_rotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.4,
        ),
        _make_gauss_noise(var_limit=(0.001, 0.015), p=0.3),
        _make_grid_distortion(
            num_steps=5,
            distort_limit=0.1,
            p=0.2,
        ),
        _make_coarse_dropout(
            num_holes=(1, 4),
            hole_height=(16, 32),
            hole_width=(16, 32),
            p=0.2,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
    ]

    compose_kwargs: Dict[str, Any] = {
        "additional_targets": {"mask": "mask"},
        "is_check_shapes": False,
    }
    sig = inspect.signature(A.Compose.__init__).parameters
    if "seed" in sig and seed is not None:
        compose_kwargs["seed"] = seed

    return A.Compose(transforms_list, **compose_kwargs)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _check_albumentations() -> None:
    """Kiểm tra albumentations đã cài đặt chưa."""
    if not _ALBUMENTATIONS_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "albumentations chưa được cài đặt.\n"
            "Chạy: pip install albumentations>=1.3.0"
        )


def describe_transform(transform: "Compose") -> Dict[str, Any]:
    """Trả về mô tả dạng dict của pipeline transform.

    Args:
        transform: Pipeline albumentations.

    Returns:
        Dict chứa tên và xác suất của từng biến đổi.
    """
    if not hasattr(transform, "transforms"):
        return {"transforms": []}
    return {
        "transforms": [
            {
                "name": type(t).__name__,
                "p": getattr(t, "p", None),
            }
            for t in transform.transforms
        ]
    }
