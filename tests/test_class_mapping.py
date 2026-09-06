"""Unit tests cho module src.class_mapping."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.class_mapping import (
    VALID_CLASSES,
    build_lookup_table,
    compute_class_histogram,
    remap_array,
    remap_worldcover_raster,
)
from src.config import ESA_TO_LOCAL_CLASS


class TestClassMapping(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho các chức năng ánh xạ nhãn."""

    def test_build_lookup_table_default(self) -> None:
        lut = build_lookup_table()
        self.assertEqual(lut.shape, (256,))
        self.assertEqual(lut.dtype, np.uint8)

        # Kiểm tra các ánh xạ theo chuẩn
        self.assertEqual(lut[40], 1)  # Lúa
        self.assertEqual(lut[50], 2)  # Khu dân cư
        self.assertEqual(lut[80], 3)  # Thủy sản
        self.assertEqual(lut[95], 4)  # Rừng ngập mặn
        self.assertEqual(lut[10], 5)  # Tree cover
        self.assertEqual(lut[20], 5)  # Shrubland
        self.assertEqual(lut[30], 5)  # Grassland
        self.assertEqual(lut[60], 6)  # Đồng muối / Đất trống

        # Các mã nền / không được ánh xạ phải bằng 0
        self.assertEqual(lut[0], 0)
        self.assertEqual(lut[70], 0)
        self.assertEqual(lut[90], 0)
        self.assertEqual(lut[100], 0)
        self.assertEqual(lut[255], 0)

    def test_build_lookup_table_validation(self) -> None:
        # Mã đích ngoài 0..6 phải raise ValueError
        with self.assertRaises(ValueError):
            build_lookup_table({40: 7})

        # Mã nguồn ngoài 0..255 phải raise ValueError
        with self.assertRaises(ValueError):
            build_lookup_table({300: 1})

    def test_remap_array_success(self) -> None:
        lut = build_lookup_table()
        input_arr = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [80, 90, 95],
                [0, 70, 100],
            ],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                [5, 5, 5],
                [1, 2, 6],
                [3, 0, 4],  # 90 là wetland -> 0
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )

        remapped = remap_array(input_arr, lut)
        self.assertEqual(remapped.dtype, np.uint8)
        self.assertTrue(np.array_equal(remapped, expected))
        self.assertTrue(set(np.unique(remapped)).issubset(VALID_CLASSES))

    def test_compute_class_histogram(self) -> None:
        arr = np.array([0, 1, 1, 2, 5, 5, 5, 6], dtype=np.uint8)
        hist = compute_class_histogram(arr)

        self.assertEqual(set(hist.keys()), set(range(7)))
        self.assertEqual(hist[0], 1)
        self.assertEqual(hist[1], 2)
        self.assertEqual(hist[2], 1)
        self.assertEqual(hist[3], 0)
        self.assertEqual(hist[4], 0)
        self.assertEqual(hist[5], 3)
        self.assertEqual(hist[6], 1)

    def test_remap_raster_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            in_tif = tmp_path / "synthetic_worldcover.tif"
            out_tif = tmp_path / "synthetic_mask7.tif"
            out_png = tmp_path / "synthetic_preview.png"

            # Tạo raster giả lập 10x10 với CRS EPSG:32648
            h, w = 10, 10
            synthetic_data = np.array([[40] * 5 + [50] * 5] * 5 + [[80] * 5 + [10] * 5] * 5, dtype=np.uint8)
            transform = Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 2200000.0)

            profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": 1,
                "dtype": rasterio.uint8,
                "crs": "EPSG:32648",
                "transform": transform,
            }

            with rasterio.open(in_tif, "w", **profile) as dst:
                dst.write(synthetic_data, 1)

            # Chạy remap
            out_result, hist = remap_worldcover_raster(
                in_tif,
                out_tif,
                preview_path=out_png,
            )

            self.assertTrue(out_result.exists())
            self.assertTrue(out_png.exists())

            # Kiểm tra dữ liệu raster đầu ra
            with rasterio.open(out_tif) as dst:
                self.assertEqual(dst.crs.to_string(), "EPSG:32648")
                self.assertEqual(dst.transform, transform)
                self.assertEqual(dst.dtypes[0], "uint8")
                out_data = dst.read(1)
                unique_vals = set(np.unique(out_data))
                self.assertTrue(unique_vals.issubset(VALID_CLASSES))

                # Kiểm tra colormap đã được nhúng
                colormap = dst.colormap(1)
                self.assertIsNotNone(colormap)
                self.assertIn(1, colormap)


if __name__ == "__main__":
    unittest.main()
