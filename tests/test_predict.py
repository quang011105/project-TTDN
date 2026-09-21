"""
Unit tests cho src.predict — Hậu xử lý kết quả dự đoán (Bước 2.7) & Inference toàn cảnh (Bước 2.8).

Chạy:
    python -m unittest tests/test_predict.py -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.predict import (
    apply_mmu_filter,
    apply_morphological_filter,
    compute_area_statistics,
    create_2d_window,
    export_lulc_geotiff,
    export_lulc_png,
    postprocess_mask,
)


class TestPredictPostprocessing(unittest.TestCase):
    """Kiểm thử Bước 2.7: Thuật toán lọc hình thái học và MMU filter."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_morphological_filter_majority(self) -> None:
        """Kiểm tra bộ lọc đa số 3x3 loại bỏ pixel rác cô lập."""
        # Tạo ma trận 5x5 toàn lớp 1 (Lúa), có 1 điểm nhiễu lớp 2 (Khu dân cư) ở tâm
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[2, 2] = 2

        filtered = apply_morphological_filter(mask, kernel_size=3, method="majority")
        # Điểm tâm phải được đa số xung quanh (lớp 1) sáp nhập
        self.assertEqual(filtered[2, 2], 1)
        self.assertTrue(np.all(filtered == 1))

    def test_morphological_filter_opening_closing(self) -> None:
        """Kiểm tra bộ lọc opening/closing loại bỏ nhiễu đơn lẻ."""
        mask = np.ones((7, 7), dtype=np.uint8)
        mask[3, 3] = 2

        filtered = apply_morphological_filter(mask, kernel_size=3, method="opening_closing")
        self.assertEqual(filtered[3, 3], 1)

    def test_morphological_filter_preserves_background(self) -> None:
        """Kiểm tra cờ preserve_background bảo toàn vùng nodata (0)."""
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:5, 1:5] = 1
        filtered = apply_morphological_filter(mask, kernel_size=3, preserve_background=True)
        self.assertEqual(filtered[0, 0], 0)
        self.assertEqual(filtered[5, 5], 0)

    def test_mmu_filter_eliminates_small_clusters(self) -> None:
        """Kiểm tra MMU filter: cụm < 5 pixels bị xóa và sáp nhập vào láng giềng."""
        # Nền lớp 1 (Lúa), có 1 cụm 3 pixel của lớp 2 (Dân cư) và 1 cụm 8 pixel của lớp 3 (Thủy sản)
        mask = np.ones((10, 10), dtype=np.uint8)
        # Cụm nhỏ: 3 pixels (< 5)
        mask[1, 1:4] = 2
        # Cụm lớn: 8 pixels (>= 5)
        mask[5:7, 5:9] = 3

        filtered = apply_mmu_filter(mask, mmu_pixels=5, background_val=0)

        # Cụm 3 pixel của lớp 2 phải bị xóa (thành lớp 1)
        self.assertTrue(np.all(filtered[1, 1:4] == 1))
        # Cụm 8 pixel của lớp 3 phải được giữ nguyên hoàn toàn
        self.assertTrue(np.all(filtered[5:7, 5:9] == 3))

    def test_mmu_filter_noise_in_background(self) -> None:
        """Kiểm tra điểm nhiễu cô lập nằm trong đại dương/nodata (0) bị sáp nhập về 0."""
        mask = np.zeros((8, 8), dtype=np.uint8)
        # 1 pixel lớp 4 (Rừng ngập mặn) lạc lõng giữa biển
        mask[4, 4] = 4

        filtered = apply_mmu_filter(mask, mmu_pixels=5, background_val=0)
        self.assertEqual(filtered[4, 4], 0)
        self.assertTrue(np.all(filtered == 0))

    def test_postprocess_mask_flags(self) -> None:
        """Kiểm tra hàm wrapper postprocess_mask với các cờ bật/tắt."""
        mask = np.ones((8, 8), dtype=np.uint8)
        mask[2, 2] = 2

        # Bật đầy đủ
        res_full = postprocess_mask(mask, enable_morph=True, enable_mmu=True, mmu_pixels=5)
        self.assertEqual(res_full[2, 2], 1)

        # Tắt toàn bộ: phải giữ nguyên gốc
        res_disabled = postprocess_mask(mask, enable_morph=False, enable_mmu=False)
        self.assertEqual(res_disabled[2, 2], 2)

    def test_create_2d_window(self) -> None:
        """Kiểm tra tạo cửa sổ trọng số 2D Hann soft blending."""
        w = create_2d_window(patch_size=256, min_weight=0.05)
        self.assertEqual(w.shape, (256, 256))
        self.assertGreaterEqual(float(np.min(w)), 0.05)
        self.assertLessEqual(float(np.max(w)), 1.0)
        # Điểm tâm phải có trọng số cao nhất (~1.0)
        center_val = w[128, 128]
        self.assertAlmostEqual(center_val, 1.0, places=2)
        # Điểm góc mép phải bằng min_weight
        self.assertAlmostEqual(w[0, 0], 0.05, places=2)

    def test_compute_area_statistics(self) -> None:
        """Kiểm tra tính toán diện tích đất (pixel, ha, km², %)."""
        # Tạo ma trận giả lập: 100 px lớp 1 (Lúa), 200 px lớp 2 (Dân cư), 100 px nền 0
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:10, :] = 1   # 200 px
        mask[10:15, :] = 2 # 100 px
        # Tổng foreground = 300 px

        json_file = self.tmp_dir / "stats.json"
        csv_file = self.tmp_dir / "stats.csv"

        stats = compute_area_statistics(
            mask,
            pixel_size_m=10.0,
            output_json=json_file,
            output_csv=csv_file,
        )

        self.assertEqual(stats["total_foreground_pixels"], 300)
        # 300 px * 100 m² = 30,000 m² = 3.0 ha = 0.03 km²
        self.assertAlmostEqual(stats["total_area_ha"], 3.0)
        self.assertAlmostEqual(stats["total_area_km2"], 0.03)

        # Lớp 1: 200 / 300 = 66.67%
        self.assertEqual(stats["classes"]["1"]["pixel_count"], 200)
        self.assertAlmostEqual(stats["classes"]["1"]["percentage"], 66.67, places=2)

        # Lớp 2: 100 / 300 = 33.33%
        self.assertEqual(stats["classes"]["2"]["pixel_count"], 100)
        self.assertAlmostEqual(stats["classes"]["2"]["percentage"], 33.33, places=2)

        # Kiểm tra file đã tạo thành công
        self.assertTrue(json_file.exists())
        self.assertTrue(csv_file.exists())

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["total_foreground_pixels"], 300)

    def test_export_lulc_geotiff(self) -> None:
        """Kiểm tra xuất GeoTIFF nhúng Colormap chuẩn GDAL."""
        # Tạo một file GeoTIFF nguồn giả lập
        src_path = self.tmp_dir / "synthetic_src.tif"
        out_path = self.tmp_dir / "lulc_out.tif"

        transform = from_origin(106.0, 20.0, 10.0, 10.0)
        data = np.random.randint(0, 1000, size=(4, 20, 20), dtype=np.uint16)

        with rasterio.open(
            src_path,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=4,
            dtype="uint16",
            crs="EPSG:32648",
            transform=transform,
        ) as dst:
            dst.write(data)

        pred_mask = np.random.randint(0, 7, size=(20, 20), dtype=np.uint8)
        export_lulc_geotiff(pred_mask, src_path, out_path)

        self.assertTrue(out_path.exists())
        with rasterio.open(out_path) as dst:
            self.assertEqual(dst.count, 1)
            self.assertEqual(dst.dtypes[0], "uint8")
            self.assertEqual(str(dst.crs), "EPSG:32648")
            # Kiểm tra Colormap đã nhúng thành công
            cmap = dst.colormap(1)
            self.assertIn(1, cmap)
            # Lớp 1 là Lúa: Vàng (255, 215, 0, 255)
            self.assertEqual(cmap[1], (255, 215, 0, 255))

    def test_export_lulc_png(self) -> None:
        """Kiểm tra xuất ảnh bản đồ màu PNG chất lượng cao kèm chú giải."""
        pred_mask = np.random.randint(0, 7, size=(30, 30), dtype=np.uint8)
        out_png = self.tmp_dir / "lulc_map.png"

        export_lulc_png(pred_mask, out_png)
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
