"""
PyTorch Dataset và DataLoader cho ảnh vệ tinh Sentinel-2 patch 4 kênh.

Cấu trúc thư mục kỳ vọng:
    data/processed/patches/
    ├── train/
    │   ├── images/   <- patch_RRRR_CCCC.tif  (4 kênh, uint16)
    │   └── masks/    <- patch_RRRR_CCCC.tif  (1 kênh, uint8, giá trị 0–6)
    └── test/
        ├── images/
        └── masks/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import CLASS_NAMES, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Hằng số chuẩn hoá Sentinel-2 Surface Reflectance
# ---------------------------------------------------------------------------
S2_NORMALIZE_MAX: float = 10_000.0  # Giá trị SR tối đa Sentinel-2 L2A

# ---------------------------------------------------------------------------
# Thống kê per-channel (Mean / Std) tính trên toàn tập train Giao Thủy
# (tính bằng script scripts/compute_band_stats.py — đơn vị SR /10000)
# Dùng cho chuẩn hoá z-score nếu muốn thay thế min-max.
# ---------------------------------------------------------------------------
S2_BAND_MEAN: List[float] = [0.1051, 0.1127, 0.1333, 0.2971]  # B2,B3,B4,B8
S2_BAND_STD: List[float] = [0.0513, 0.0527, 0.0617, 0.0892]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GeoTiffPatchDataset(Dataset):
    """Dataset đọc cặp (ảnh vệ tinh, mask nhãn) từ thư mục patch GeoTIFF.

    Args:
        split: ``"train"`` hoặc ``"test"``.
        patches_dir: Đường dẫn thư mục gốc chứa ``train/`` và ``test/``.
        transform: Callable nhận ``(image: np.ndarray H×W×C, mask: np.ndarray H×W)``
                   và trả về dict ``{"image": np.ndarray, "mask": np.ndarray}``.
                   Thường là ``albumentations.Compose``.
        normalize: Nếu ``True``, chia pixel cho ``S2_NORMALIZE_MAX`` (min-max [0,1]).
        use_zscore: Nếu ``True``, áp dụng z-score thay cho min-max (dùng khi
                    normalize=True đã chia). Tắt theo mặc định.

    Returns:
        ``(image, mask)`` với:
        - ``image``: ``FloatTensor[4, 256, 256]`` trong khoảng [0, 1].
        - ``mask`` : ``LongTensor[256, 256]`` giá trị 0–6.
    """

    def __init__(
        self,
        split: str,
        patches_dir: Path = PROCESSED_DIR / "patches",
        transform: Optional[Callable] = None,
        normalize: bool = True,
        use_zscore: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split phải là 'train' hoặc 'test', nhận được: {split!r}")

        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.use_zscore = use_zscore

        self.images_dir = patches_dir / split / "images"
        self.masks_dir = patches_dir / split / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục mask: {self.masks_dir}")

        # Thu thập và sắp xếp tên file
        self.image_paths: List[Path] = sorted(self.images_dir.glob("*.tif"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Không tìm thấy file .tif trong {self.images_dir}")

        # Kiểm tra từng mask tương ứng tồn tại
        missing: List[str] = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / img_path.name
            if not mask_path.exists():
                missing.append(img_path.name)
        if missing:
            raise FileNotFoundError(
                f"Thiếu {len(missing)} mask tương ứng, ví dụ: {missing[:3]}"
            )

        # Thống kê z-score (chỉ dùng nếu use_zscore=True)
        self._band_mean = np.array(S2_BAND_MEAN, dtype=np.float32).reshape(4, 1, 1)
        self._band_std = np.array(S2_BAND_STD, dtype=np.float32).reshape(4, 1, 1)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        # --- Đọc ảnh vệ tinh 4 kênh uint16 → float32 ---
        with rasterio.open(img_path) as src:
            image: np.ndarray = src.read().astype(np.float32)  # (4, H, W)

        # --- Đọc mask nhãn 1 kênh uint8 ---
        with rasterio.open(mask_path) as src:
            mask: np.ndarray = src.read(1).astype(np.int64)  # (H, W)

        # --- Chuẩn hoá ---
        if self.normalize:
            image = image / S2_NORMALIZE_MAX  # [0, 1]
            image = np.clip(image, 0.0, 1.0)  # đề phòng outlier SR > 10000
            if self.use_zscore:
                image = (image - self._band_mean) / (self._band_std + 1e-8)

        # --- Augmentation (albumentations nhận H×W×C) ---
        if self.transform is not None:
            image_hwc = image.transpose(1, 2, 0)  # (H, W, 4) cho albumentations
            mask_hw = mask.astype(np.int32)
            augmented = self.transform(image=image_hwc, mask=mask_hw)
            image_hwc = augmented["image"]
            mask = augmented["mask"].astype(np.int64)
            image = image_hwc.transpose(2, 0, 1)  # (4, H, W)

        # --- Chuyển sang Tensor ---
        image_tensor: Tensor = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        )  # FloatTensor[4, H, W]
        mask_tensor: Tensor = torch.from_numpy(
            np.ascontiguousarray(mask, dtype=np.int64)
        )  # LongTensor[H, W]

        return image_tensor, mask_tensor

    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """Trả về thông tin tóm tắt dataset."""
        return {
            "split": self.split,
            "n_patches": len(self),
            "images_dir": str(self.images_dir),
            "masks_dir": str(self.masks_dir),
            "normalize": self.normalize,
            "use_zscore": self.use_zscore,
            "n_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
        }


# ---------------------------------------------------------------------------
# Hàm tiện ích tạo DataLoader
# ---------------------------------------------------------------------------
def get_dataloaders(
    patches_dir: Path = PROCESSED_DIR / "patches",
    batch_size: int = 8,
    num_workers: int = 2,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    normalize: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Tạo cặp (train_loader, test_loader) sẵn sàng đưa vào training loop.

    Args:
        patches_dir: Thư mục gốc chứa train/ và test/.
        batch_size: Số lượng patch mỗi batch.
        num_workers: Số luồng DataLoader (0 = main thread, an toàn cho Windows).
        train_transform: Augmentation pipeline cho tập train (albumentations.Compose).
        val_transform: Augmentation pipeline cho tập test (thường là None hoặc chỉ resize).
        normalize: Chuẩn hoá pixel /10000.
        pin_memory: ``True`` khi có GPU (giảm latency CPU→GPU).
        persistent_workers: ``True`` khi ``num_workers > 0`` và train nhiều epoch.

    Returns:
        ``(train_loader, test_loader)``.

    Example::

        from src.dataset import get_dataloaders
        from src.transforms import get_train_transform, get_val_transform

        train_loader, test_loader = get_dataloaders(
            batch_size=8,
            num_workers=2,
            train_transform=get_train_transform(),
            val_transform=get_val_transform(),
        )
        for images, masks in train_loader:
            # images: FloatTensor[B, 4, 256, 256]
            # masks:  LongTensor[B, 256, 256]
            ...
    """
    train_ds = GeoTiffPatchDataset(
        split="train",
        patches_dir=patches_dir,
        transform=train_transform,
        normalize=normalize,
    )
    test_ds = GeoTiffPatchDataset(
        split="test",
        patches_dir=patches_dir,
        transform=val_transform,
        normalize=normalize,
    )

    _persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,               # Bỏ batch cuối nếu không đủ số lượng
        persistent_workers=_persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=_persistent,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CLI tiện ích — kiểm tra nhanh dataset
# ---------------------------------------------------------------------------
def _cli_check() -> None:
    """Kiểm tra dataset và in thông tin tóm tắt.

    Usage::
        python -m src.dataset
    """
    import sys

    warnings.filterwarnings("ignore", category=UserWarning)

    for split in ("train", "test"):
        try:
            ds = GeoTiffPatchDataset(split=split)
            info = ds.summary()
            print(f"\n{'='*50}")
            print(f"  Split: {info['split'].upper()}")
            print(f"  Số patch     : {info['n_patches']}")
            print(f"  Thư mục ảnh  : {info['images_dir']}")
            print(f"  Thư mục mask : {info['masks_dir']}")
            print(f"  Normalize    : {info['normalize']}")

            # Đọc thử 1 sample
            img, msk = ds[0]
            print(f"  image.shape  : {tuple(img.shape)}  dtype={img.dtype}")
            print(f"  mask.shape   : {tuple(msk.shape)}  dtype={msk.dtype}")
            print(f"  image range  : [{img.min():.4f}, {img.max():.4f}]")
            print(f"  mask labels  : {msk.unique().tolist()}")

            # Thống kê per-class pixel count trên toàn split
            all_labels: List[int] = []
            label_counts: Dict[int, int] = {c: 0 for c in range(7)}
            for i in range(len(ds)):
                _, m = ds[i]
                for c in range(7):
                    label_counts[c] += int((m == c).sum())

            total_px = sum(label_counts.values())
            print(f"\n  Phân bố nhãn (toàn split):")
            for c, cnt in label_counts.items():
                pct = cnt / total_px * 100 if total_px > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    {c} {CLASS_NAMES[c]:<30s}: {cnt:>8,} px  ({pct:5.2f}%)  {bar}")

        except Exception as exc:
            print(f"[ERROR] split={split}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    _cli_check()
