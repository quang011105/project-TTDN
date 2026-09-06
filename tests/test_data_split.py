"""Unit tests cho module src.data_split (Bước 1.4)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.data_split import (
    create_split_mask,
    extract_patches,
    generate_spatial_blocks,
    optimize_block_split,
    run_spatial_split_pipeline,
    save_split_mask_raster,
)


class TestDataSplit(unittest.TestCase):
    """Bộ kiểm thử cho phân tách không gian và trích xuất patches."""

    def test_generate_spatial_blocks_coverage(self) -> None:
        h, w = 1000, 2000
        n_rows, n_cols = 4, 5
        blocks = generate_spatial_blocks(h, w, n_rows=n_rows, n_cols=n_cols)

        self.assertEqual(len(blocks), n_rows * n_cols)

        # Kiểm tra tính bao phủ 100% diện tích không đè lấn
        coverage = np.zeros((h, w), dtype=np.uint8)
        for blk in blocks:
            r_s, r_e = blk["r_start"], blk["r_end"]
            c_s, c_e = blk["c_start"], blk["c_end"]
            coverage[r_s:r_e, c_s:c_e] += 1

        self.assertTrue(np.all(coverage == 1))

    def test_optimize_block_split_disjoint(self) -> None:
        h, w = 200, 200
        blocks = generate_spatial_blocks(h, w, n_rows=2, n_cols=2)

        # Tạo mask giả lập với 4 lớp ở 4 khối
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:100, :100] = 1
        mask[:100, 100:] = 2
        mask[100:, :100] = 3
        mask[100:, 100:] = 4

        train_idx, test_idx, stats = optimize_block_split(mask, blocks, target_test_ratio=0.25)

        # Tập Train và Test phải tách rời nhau hoàn toàn
        self.assertEqual(len(set(train_idx).intersection(set(test_idx))), 0)
        self.assertEqual(len(train_idx) + len(test_idx), len(blocks))
        self.assertAlmostEqual(stats["test_ratio"], 0.25, delta=0.05)

    def test_create_and_save_split_mask(self) -> None:
        h, w = 100, 100
        blocks = generate_spatial_blocks(h, w, n_rows=2, n_cols=2)
        train_idx = [0, 1, 2]
        test_idx = [3]

        split_mask = create_split_mask(h, w, blocks, train_idx, test_idx)
        unique_vals = set(np.unique(split_mask))
        self.assertEqual(unique_vals, {1, 2})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_tif = tmp_path / "template.tif"
            out_split_tif = tmp_path / "split_mask.tif"

            # Tạo template raster
            profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": 1,
                "dtype": rasterio.uint8,
                "crs": "EPSG:32648",
                "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
            }
            with rasterio.open(template_tif, "w", **profile) as dst:
                dst.write(np.zeros((h, w), dtype=np.uint8), 1)

            save_split_mask_raster(split_mask, template_tif, out_split_tif)

            self.assertTrue(out_split_tif.exists())
            with rasterio.open(out_split_tif) as dst:
                self.assertEqual(dst.crs.to_string(), "EPSG:32648")
                self.assertEqual(dst.read(1).shape, (h, w))
                self.assertEqual(set(np.unique(dst.read(1))), {1, 2})

    def test_extract_patches_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            s2_path = tmp_path / "s2.tif"
            mask_path = tmp_path / "mask.tif"

            h, w = 512, 512
            transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)

            # S2 4 bands
            s2_data = np.ones((4, h, w), dtype=np.uint16) * 1000
            with rasterio.open(
                s2_path, "w", driver="GTiff", height=h, width=w, count=4, dtype=rasterio.uint16,
                crs="EPSG:32648", transform=transform
            ) as dst:
                dst.write(s2_data)

            # Mask 1 band
            mask_data = np.ones((h, w), dtype=np.uint8) * 1
            with rasterio.open(
                mask_path, "w", driver="GTiff", height=h, width=w, count=1, dtype=rasterio.uint8,
                crs="EPSG:32648", transform=transform
            ) as dst:
                dst.write(mask_data, 1)

            # Split mask: nửa trái là Train (1), nửa phải là Test (2)
            split_mask = np.zeros((h, w), dtype=np.uint8)
            split_mask[:, :256] = 1
            split_mask[:, 256:] = 2

            stats = extract_patches(
                s2_path, mask_path, split_mask, tmp_path, patch_size=256, stride=256
            )

            self.assertGreater(stats["train_patches"], 0)
            self.assertGreater(stats["test_patches"], 0)

            # Kiểm tra file patch được tạo
            train_patches = list((tmp_path / "patches" / "train" / "images").glob("*.tif"))
            self.assertEqual(len(train_patches), stats["train_patches"])

            with rasterio.open(train_patches[0]) as p_ds:
                self.assertEqual(p_ds.shape, (256, 256))
                self.assertEqual(p_ds.count, 4)


if __name__ == "__main__":
    unittest.main()
