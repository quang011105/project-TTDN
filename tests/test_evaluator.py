"""
Unit tests cho src.evaluator — Đánh giá toàn diện trên tập Golden Test.

Chạy:
    python -m unittest tests/test_evaluator.py -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluator import (
    compute_metrics_from_confusion_matrix,
    plot_confusion_matrix,
    plot_per_class_iou,
    plot_training_curves,
)


class TestEvaluator(unittest.TestCase):
    """Kiểm thử tính toán chỉ số đánh giá và xuất biểu đồ."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_compute_metrics_perfect(self) -> None:
        """Ma trận đường chéo hoàn hảo (100% đúng) phải cho OA=1, mIoU=1, Kappa=1."""
        cm_perfect = np.diag([100, 200, 300, 400, 500, 600]).astype(np.int64)
        metrics = compute_metrics_from_confusion_matrix(cm_perfect)

        self.assertAlmostEqual(metrics["overall_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_iou"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["cohen_kappa"], 1.0)

        for c in range(1, 7):
            self.assertAlmostEqual(metrics["per_class"][c]["iou"], 1.0)
            self.assertAlmostEqual(metrics["per_class"][c]["f1_score"], 1.0)

    def test_compute_metrics_zeros(self) -> None:
        """Ma trận toàn 0 không gây ra lỗi chia cho 0."""
        cm_zeros = np.zeros((6, 6), dtype=np.int64)
        metrics = compute_metrics_from_confusion_matrix(cm_zeros)

        self.assertEqual(metrics["overall_accuracy"], 0.0)
        self.assertEqual(metrics["mean_iou"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["cohen_kappa"], 0.0)

    def test_compute_metrics_known_case(self) -> None:
        """Kiểm tra tính toán chính xác trên ma trận đã biết trước đáp án."""
        cm = np.array([
            [80, 20, 0, 0, 0, 0],
            [10, 90, 0, 0, 0, 0],
            [0, 0, 100, 0, 0, 0],
            [0, 0, 0, 100, 0, 0],
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 100],
        ], dtype=np.int64)

        metrics = compute_metrics_from_confusion_matrix(cm)
        # Lớp 1: TP=80, FP=10, FN=20 -> IoU = 80 / (80+10+20) = 80/110 ≈ 0.7273
        self.assertAlmostEqual(metrics["per_class"][1]["iou"], 80 / 110, places=4)
        self.assertAlmostEqual(metrics["per_class"][1]["precision"], 80 / 90, places=4)
        self.assertAlmostEqual(metrics["per_class"][1]["recall"], 80 / 100, places=4)

    def test_plot_confusion_matrix(self) -> None:
        """Kiểm tra xuất file ảnh Heatmap Ma trận nhầm lẫn."""
        cm = np.diag([10, 20, 30, 40, 50, 60]).astype(np.int64)
        out_png = self.tmp_dir / "cm.png"
        plot_confusion_matrix(cm, out_png, normalize=True)
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_per_class_iou(self) -> None:
        """Kiểm tra xuất file ảnh biểu đồ cột IoU."""
        per_class = {
            c: {"iou": 0.5 + c * 0.05}
            for c in range(1, 7)
        }
        out_png = self.tmp_dir / "iou.png"
        plot_per_class_iou(per_class, mean_iou=0.65, output_path=out_png)
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
