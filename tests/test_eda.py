"""Unit tests cho module src.eda (Bước 1.3)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.eda import compute_median_frequency_weights, run_eda_pipeline


class TestEDA(unittest.TestCase):
    """Bộ kiểm thử cho EDA và tính Class Weights."""

    def test_compute_median_frequency_weights_standard(self) -> None:
        # Giả lập số pixel 7 lớp:
        # Lớp 0: 500 (background)
        # Lớp 1: 100 (freq = 100/1000 = 0.10)
        # Lớp 2: 200 (freq = 200/1000 = 0.20)
        # Lớp 3: 300 (freq = 300/1000 = 0.30)
        # Lớp 4: 0   (lớp vắng mặt)
        # Lớp 5: 250 (freq = 250/1000 = 0.25)
        # Lớp 6: 150 (freq = 150/1000 = 0.15)
        # Tổng foreground = 100 + 200 + 300 + 0 + 250 + 150 = 1000
        # Positive frequencies: [0.10, 0.15, 0.20, 0.25, 0.30] -> median = 0.20
        hist = {0: 500, 1: 100, 2: 200, 3: 300, 4: 0, 5: 250, 6: 150}

        results = compute_median_frequency_weights(hist)

        # Lớp 0 phải luôn luôn có weight = 0.0
        self.assertEqual(results["weights_list"][0], 0.0)
        self.assertEqual(results["raw_weights"][0], 0.0)

        # Lớp 4 vắng mặt (count=0) phải có weight = 0.0
        self.assertEqual(results["weights_list"][4], 0.0)

        # Median frequency = 0.20
        self.assertAlmostEqual(results["median_frequency"], 0.20, places=4)

        # Trọng số: w_c = 0.20 / f_c
        # Lớp 1: 0.20 / 0.10 = 2.00
        self.assertAlmostEqual(results["weights_list"][1], 2.0, places=2)
        # Lớp 2: 0.20 / 0.20 = 1.00 (lớp trung vị)
        self.assertAlmostEqual(results["weights_list"][2], 1.0, places=2)
        # Lớp 3: 0.20 / 0.30 = 0.6667
        self.assertAlmostEqual(results["weights_list"][3], 0.6667, places=2)
        # Lớp 5: 0.20 / 0.25 = 0.80
        self.assertAlmostEqual(results["weights_list"][5], 0.80, places=2)
        # Lớp 6: 0.20 / 0.15 = 1.3333
        self.assertAlmostEqual(results["weights_list"][6], 1.3333, places=2)

    def test_compute_median_frequency_all_zero(self) -> None:
        # Trường hợp không có pixel foreground nào
        hist = {0: 1000, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        results = compute_median_frequency_weights(hist)
        self.assertEqual(results["weights_list"], [0.0] * 7)

    def test_run_eda_pipeline_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mask_path = tmp_path / "mock_mask7.tif"
            s2_path = tmp_path / "mock_s2.tif"
            out_dir = tmp_path / "output"

            # Tạo raster mask 20x20
            h, w = 20, 20
            transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            mask_arr[:10, :10] = 1  # Lúa
            mask_arr[:10, 10:] = 2  # Khu dân cư
            mask_arr[10:, :10] = 3  # Thủy sản
            mask_arr[10:, 10:] = 5  # Cây lâu năm

            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=rasterio.uint8,
                crs="EPSG:32648",
                transform=transform,
            ) as dst:
                dst.write(mask_arr, 1)

            # Tạo raster S2 4 bands
            s2_arr = np.random.randint(100, 3000, size=(4, h, w), dtype=np.uint16)
            with rasterio.open(
                s2_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=4,
                dtype=rasterio.uint16,
                crs="EPSG:32648",
                transform=transform,
            ) as dst:
                dst.write(s2_arr)

            # Chạy pipeline
            summary = run_eda_pipeline(mask_path, s2_path, out_dir=out_dir)

            # Kiểm tra các file đầu ra
            self.assertTrue((out_dir / "class_weights.json").exists())
            self.assertTrue((out_dir / "eda_summary.json").exists())
            self.assertTrue((out_dir / "eda_class_distribution.png").exists())
            self.assertTrue((out_dir / "eda_spatial_overlay.png").exists())

            # Kiểm tra nội dung class_weights.json
            with open(out_dir / "class_weights.json", "r", encoding="utf-8") as fh:
                data = json.load(fh)
                self.assertIn("weights", data)
                self.assertEqual(len(data["weights"]), 7)
                self.assertEqual(data["weights"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
