"""
Unit tests cho src.dataset — GeoTiffPatchDataset và get_dataloaders.

Chạy:
    python -m unittest tests/test_dataset.py -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import torch


def _make_fake_patch_dir(tmp_dir: Path, n_train: int = 4, n_test: int = 2) -> Path:
    """Tạo thư mục patch giả (fake) để kiểm thử mà không cần dữ liệu thực."""
    patches_dir = tmp_dir / "patches"
    H, W = 256, 256
    transform = from_bounds(0, 0, 1, 1, W, H)
    crs = "EPSG:32648"

    for split, n in (("train", n_train), ("test", n_test)):
        (patches_dir / split / "images").mkdir(parents=True)
        (patches_dir / split / "masks").mkdir(parents=True)
        for i in range(n):
            fname = f"patch_{i:04d}_0000.tif"
            # --- Tạo ảnh 4 kênh uint16 với giá trị ngẫu nhiên ---
            img_data = np.random.randint(100, 5000, (4, H, W), dtype=np.uint16)
            with rasterio.open(
                patches_dir / split / "images" / fname,
                "w",
                driver="GTiff",
                count=4,
                height=H,
                width=W,
                dtype="uint16",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(img_data)

            # --- Tạo mask 1 kênh uint8 với nhãn ngẫu nhiên 0–6 ---
            mask_data = np.random.randint(0, 7, (H, W), dtype=np.uint8)
            with rasterio.open(
                patches_dir / split / "masks" / fname,
                "w",
                driver="GTiff",
                count=1,
                height=H,
                width=W,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(mask_data[np.newaxis, ...])

    return patches_dir


class TestGeoTiffPatchDataset(unittest.TestCase):
    """Kiểm thử GeoTiffPatchDataset."""

    def setUp(self) -> None:
        from src.dataset import GeoTiffPatchDataset

        self.GeoTiffPatchDataset = GeoTiffPatchDataset
        self.tmp = Path(tempfile.mkdtemp())
        self.patches_dir = _make_fake_patch_dir(self.tmp, n_train=4, n_test=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- 1. Kiểm tra __len__ ---
    def test_len_train(self) -> None:
        ds = self.GeoTiffPatchDataset("train", patches_dir=self.patches_dir)
        self.assertEqual(len(ds), 4)

    def test_len_test(self) -> None:
        ds = self.GeoTiffPatchDataset("test", patches_dir=self.patches_dir)
        self.assertEqual(len(ds), 2)

    # --- 2. Kiểm tra output shape và dtype ---
    def test_output_shape_dtype(self) -> None:
        ds = self.GeoTiffPatchDataset("train", patches_dir=self.patches_dir)
        img, msk = ds[0]
        self.assertEqual(img.shape, (4, 256, 256), "image shape phải là (4, 256, 256)")
        self.assertEqual(msk.shape, (256, 256), "mask shape phải là (256, 256)")
        self.assertEqual(img.dtype, torch.float32, "image dtype phải là float32")
        self.assertEqual(msk.dtype, torch.int64, "mask dtype phải là int64 (Long)")

    # --- 3. Kiểm tra chuẩn hoá ---
    def test_normalize_range(self) -> None:
        """Pixel ảnh sau chuẩn hoá phải nằm trong [0, 1]."""
        ds = self.GeoTiffPatchDataset(
            "train", patches_dir=self.patches_dir, normalize=True
        )
        for i in range(len(ds)):
            img, _ = ds[i]
            self.assertGreaterEqual(float(img.min()), 0.0 - 1e-6)
            self.assertLessEqual(float(img.max()), 1.0 + 1e-6)

    def test_no_normalize(self) -> None:
        """Khi normalize=False, giá trị pixel phải > 1.0 (uint16 gốc)."""
        ds = self.GeoTiffPatchDataset(
            "train", patches_dir=self.patches_dir, normalize=False
        )
        img, _ = ds[0]
        self.assertGreater(float(img.max()), 1.0)

    # --- 4. Kiểm tra nhãn hợp lệ ---
    def test_mask_labels_valid(self) -> None:
        """Mọi giá trị pixel mask phải nằm trong {0, 1, 2, 3, 4, 5, 6}."""
        valid_labels = set(range(7))
        ds = self.GeoTiffPatchDataset("train", patches_dir=self.patches_dir)
        for i in range(len(ds)):
            _, msk = ds[i]
            unique = set(msk.unique().tolist())
            self.assertTrue(
                unique.issubset(valid_labels),
                f"Patch {i}: nhãn lạ {unique - valid_labels}",
            )

    # --- 5. Kiểm tra split không hợp lệ ---
    def test_invalid_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.GeoTiffPatchDataset("validation", patches_dir=self.patches_dir)

    # --- 6. Kiểm tra summary() ---
    def test_summary_keys(self) -> None:
        ds = self.GeoTiffPatchDataset("train", patches_dir=self.patches_dir)
        info = ds.summary()
        for key in ("split", "n_patches", "images_dir", "masks_dir", "n_classes"):
            self.assertIn(key, info)
        self.assertEqual(info["split"], "train")
        self.assertEqual(info["n_patches"], 4)
        self.assertEqual(info["n_classes"], 7)

    # --- 7. Kiểm tra thiếu mask → FileNotFoundError ---
    def test_missing_mask_raises(self) -> None:
        # Xóa 1 mask
        first_mask = next((self.patches_dir / "train" / "masks").glob("*.tif"))
        first_mask.unlink()
        with self.assertRaises(FileNotFoundError):
            self.GeoTiffPatchDataset("train", patches_dir=self.patches_dir)


class TestGetDataloaders(unittest.TestCase):
    """Kiểm thử hàm get_dataloaders."""

    def setUp(self) -> None:
        from src.dataset import get_dataloaders

        self.get_dataloaders = get_dataloaders
        self.tmp = Path(tempfile.mkdtemp())
        self.patches_dir = _make_fake_patch_dir(self.tmp, n_train=4, n_test=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- 8. Kiểm tra DataLoader trả về đúng batch ---
    def test_dataloader_batch_shape(self) -> None:
        train_dl, test_dl = self.get_dataloaders(
            patches_dir=self.patches_dir,
            batch_size=2,
            num_workers=0,  # 0 để tránh lỗi multiprocessing khi test
            pin_memory=False,
        )
        imgs, msks = next(iter(train_dl))
        self.assertEqual(imgs.shape, (2, 4, 256, 256))
        self.assertEqual(msks.shape, (2, 256, 256))

    # --- 9. Kiểm tra train shuffle vs test không shuffle ---
    def test_train_loader_not_empty(self) -> None:
        train_dl, test_dl = self.get_dataloaders(
            patches_dir=self.patches_dir,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        # Cả hai loader phải có ít nhất 1 batch
        self.assertGreater(len(train_dl), 0)
        self.assertGreater(len(test_dl), 0)

    # --- 10. Kiểm tra augmentation transform hook được gọi ---
    def test_transform_called(self) -> None:
        """Kiểm tra transform được gọi bằng cách dùng mock đếm số lần gọi."""
        call_count = {"n": 0}

        def fake_transform(image, mask):
            call_count["n"] += 1
            return {"image": image, "mask": mask}

        train_dl, _ = self.get_dataloaders(
            patches_dir=self.patches_dir,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            train_transform=fake_transform,
        )
        for imgs, msks in train_dl:
            pass
        # Với 4 patch, batch_size=2, drop_last=True → 2 batch × 2 = 4 lần gọi
        self.assertEqual(call_count["n"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
