"""
Unit tests cho src.transforms — Pipeline Data Augmentation cho ảnh vệ tinh 4 kênh.

Chạy:
    python -m unittest tests/test_transforms.py -v
"""

import unittest
from typing import Set

import numpy as np
import torch

from src.transforms import (
    describe_transform,
    get_strong_train_transform,
    get_train_transform,
    get_val_transform,
)


class TestTransforms(unittest.TestCase):
    """Kiểm thử các hàm tạo và thực thi Augmentation."""

    def setUp(self) -> None:
        self.H, self.W, self.C = 256, 256, 4
        # Tạo dữ liệu giả lập ảnh 4 kênh [0, 1] float32 và mask int32 nhãn 0-6
        np.random.seed(42)
        self.dummy_image = np.random.rand(self.H, self.W, self.C).astype(np.float32)
        self.dummy_mask = np.random.randint(0, 7, (self.H, self.W), dtype=np.int32)

    def test_train_transform_structure(self) -> None:
        """Kiểm tra cấu trúc của train transform pipeline."""
        t = get_train_transform()
        desc = describe_transform(t)
        names = [item["name"] for item in desc["transforms"]]
        expected_transforms = [
            "HorizontalFlip",
            "VerticalFlip",
            "RandomRotate90",
            "ShiftScaleRotate",
            "GaussNoise",
        ]
        for exp in expected_transforms:
            self.assertIn(exp, names, f"Thiếu transform: {exp}")

    def test_val_transform_empty(self) -> None:
        """Tập validation/test không được biến đổi ngẫu nhiên."""
        t = get_val_transform()
        desc = describe_transform(t)
        self.assertEqual(len(desc["transforms"]), 0)

        # Chạy thử trên ảnh giả lập, kết quả phải giữ nguyên giá trị
        res = t(image=self.dummy_image, mask=self.dummy_mask)
        np.testing.assert_array_equal(res["image"], self.dummy_image)
        np.testing.assert_array_equal(res["mask"], self.dummy_mask)

    def test_strong_transform_structure(self) -> None:
        """Kiểm tra pipeline tăng cường mạnh (strong transform)."""
        t = get_strong_train_transform()
        desc = describe_transform(t)
        names = [item["name"] for item in desc["transforms"]]
        for exp in ["GridDistortion", "CoarseDropout", "RandomBrightnessContrast"]:
            self.assertIn(exp, names, f"Thiếu strong transform: {exp}")

    def test_apply_train_transform_shapes_and_types(self) -> None:
        """Transform phải bảo toàn shape (256, 256, 4) và dtype float32 cho ảnh."""
        t = get_train_transform(
            p_flip_h=1.0,
            p_flip_v=1.0,
            p_rotate90=1.0,
            p_noise=1.0,
            p_shift_scale_rotate=1.0,
        )
        res = t(image=self.dummy_image, mask=self.dummy_mask)

        self.assertEqual(res["image"].shape, (self.H, self.W, self.C))
        self.assertEqual(res["mask"].shape, (self.H, self.W))
        self.assertEqual(res["image"].dtype, np.float32)

    def test_mask_labels_remain_valid(self) -> None:
        """Biến đổi hình học không được sinh ra nhãn ngoài phạm vi [0, 6]."""
        t = get_train_transform(
            p_flip_h=1.0,
            p_flip_v=1.0,
            p_rotate90=1.0,
            p_noise=1.0,
            p_shift_scale_rotate=1.0,
        )
        for _ in range(5):
            res = t(image=self.dummy_image, mask=self.dummy_mask)
            unique_labels: Set[int] = set(np.unique(res["mask"]).tolist())
            self.assertTrue(
                unique_labels.issubset(set(range(7))),
                f"Phát hiện nhãn lạ sau khi transform: {unique_labels}",
            )

    def test_apply_strong_transform(self) -> None:
        """Kiểm tra thực thi strong transform trên ảnh 4 kênh."""
        t = get_strong_train_transform()
        res = t(image=self.dummy_image, mask=self.dummy_mask)
        self.assertEqual(res["image"].shape, (self.H, self.W, self.C))
        self.assertEqual(res["mask"].shape, (self.H, self.W))
        self.assertEqual(res["image"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
