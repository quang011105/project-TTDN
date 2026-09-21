"""
Unit tests cho src.losses — ComboLoss (Weighted Cross-Entropy + Dice Loss).

Chạy:
    python -m unittest tests/test_losses.py -v
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.config import NUM_CLASSES
from src.losses import ComboLoss, get_loss_fn, load_class_weights


class TestComboLoss(unittest.TestCase):
    """Kiểm thử tính toán Loss, nạp trọng số MFB và cơ chế ignore_index=0."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_class_weights()

    def test_load_class_weights_shape_and_values(self) -> None:
        """Kiểm tra nạp trọng số MFB: 7 lớp, lớp 0 = 0.0 (ignore)."""
        w = load_class_weights()
        self.assertIsInstance(w, torch.Tensor)
        self.assertEqual(w.shape, (NUM_CLASSES,))
        self.assertEqual(w.dtype, torch.float32)
        self.assertEqual(float(w[0]), 0.0, "Trọng số lớp 0 (Background) phải là 0.0.")
        # Lớp 4 (Rừng ngập mặn) và lớp 6 (Đồng muối) phải có trọng số cao do hiếm
        self.assertGreater(float(w[4]), 1.0)
        self.assertGreater(float(w[6]), 5.0)

    def test_load_class_weights_normalized(self) -> None:
        """Kiểm tra nạp trọng số đã chuẩn hoá."""
        w_norm = load_class_weights(use_normalized=True)
        self.assertEqual(w_norm.shape, (NUM_CLASSES,))
        self.assertEqual(float(w_norm[0]), 0.0)

    def test_load_class_weights_nonexistent_file_raises(self) -> None:
        """Kiểm tra raise FileNotFoundError khi đường dẫn không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            load_class_weights(weights_path="nonexistent_path/weights.json")

    def test_combo_loss_scalar_and_components(self) -> None:
        """Kiểm tra ComboLoss trả về scalar và tổng hợp đúng các thành phần."""
        criterion = ComboLoss(
            ce_weight=1.5,
            dice_weight=0.8,
            class_weights=self.weights,
            ignore_index=0,
        )

        dummy_logits = torch.randn(2, NUM_CLASSES, 64, 64)
        dummy_targets = torch.randint(0, NUM_CLASSES, (2, 64, 64), dtype=torch.long)

        # 1. Forward scalar
        loss = criterion(dummy_logits, dummy_targets)
        self.assertEqual(loss.ndim, 0, "Loss phải là scalar tensor.")
        self.assertGreater(float(loss), 0.0)

        # 2. Forward components
        components = criterion.forward_with_components(dummy_logits, dummy_targets)
        for key in ("loss", "ce_loss", "dice_loss"):
            self.assertIn(key, components)

        # Kiểm tra công thức L = ce_weight * L_ce + dice_weight * L_dice
        expected_total = 1.5 * components["ce_loss"] + 0.8 * components["dice_loss"]
        self.assertAlmostEqual(
            float(components["loss"]),
            float(expected_total),
            places=5,
            msg="Tổng loss không khớp với công thức kết hợp trọng số.",
        )

    def test_ignore_index_zero_does_not_affect_ce_loss(self) -> None:
        """Kiểm tra pixel mang nhãn 0 (Background) không đóng góp vào gradient CE."""
        criterion = ComboLoss(
            ce_weight=1.0,
            dice_weight=0.0,  # Tắt Dice để cô lập kiểm tra riêng Cross-Entropy
            class_weights=self.weights,
            ignore_index=0,
        )

        logits = torch.zeros(1, NUM_CLASSES, 16, 16, requires_grad=True)
        targets = torch.ones(1, 16, 16, dtype=torch.long)  # Tất cả đều nhãn 1

        loss1 = criterion(logits, targets)
        loss1.backward()
        grad1 = logits.grad.clone()

        # Đổi 1 nửa pixel sang nhãn 0 (Background)
        logits.grad.zero_()
        targets[:, :8, :] = 0

        loss2 = criterion(logits, targets)
        loss2.backward()
        grad2 = logits.grad.clone()

        # Gradient ở vùng nhãn 0 phải hoàn toàn bằng 0
        self.assertTrue(
            torch.all(grad2[:, :, :8, :] == 0.0),
            "Gradient tại các pixel nhãn 0 (ignore_index) phải bằng 0.",
        )

    def test_all_background_edge_case(self) -> None:
        """Kiểm tra trường hợp biên: patch chỉ toàn nhãn 0 không gây lỗi hoặc NaN."""
        criterion = ComboLoss(
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights=self.weights,
            ignore_index=0,
        )

        dummy_logits = torch.randn(1, NUM_CLASSES, 32, 32)
        all_zeros_target = torch.zeros(1, 32, 32, dtype=torch.long)

        loss = criterion(dummy_logits, all_zeros_target)
        self.assertFalse(torch.isnan(loss), "Loss không được là NaN khi target toàn 0.")
        self.assertGreaterEqual(float(loss), 0.0)

    def test_perfect_prediction_dice_loss_near_zero(self) -> None:
        """Kiểm tra khi mô hình dự đoán hoàn hảo, Dice Loss tiến gần về 0."""
        criterion = ComboLoss(
            ce_weight=0.0,
            dice_weight=1.0,
            ignore_index=0,
        )

        # Tạo mask nhãn ngẫu nhiên 1-6 (bỏ qua 0)
        target = torch.randint(1, NUM_CLASSES, (1, 32, 32), dtype=torch.long)

        # Tạo logits cực lớn tại đúng nhãn (confidence cao)
        logits = torch.full((1, NUM_CLASSES, 32, 32), -20.0)
        logits.scatter_(1, target.unsqueeze(1), 20.0)

        loss = criterion(logits, target)
        self.assertLess(
            float(loss),
            0.05,
            f"Dự đoán hoàn hảo phải có Dice Loss gần 0, nhận được: {float(loss):.4f}",
        )


    def test_get_loss_fn_factory(self) -> None:
        """Kiểm tra factory function get_loss_fn."""
        loss_fn = get_loss_fn(ce_weight=1.0, dice_weight=1.0)
        self.assertIsInstance(loss_fn, ComboLoss)
        self.assertIsNotNone(loss_fn.weights)

    def test_backward_gradient_flow(self) -> None:
        """Kiểm tra backward pass tính gradient cho logits đầu vào."""
        criterion = ComboLoss(
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights=self.weights,
            ignore_index=0,
        )

        logits = torch.randn(2, NUM_CLASSES, 32, 32, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (2, 32, 32), dtype=torch.long)

        loss = criterion(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any(), "Gradient không được chứa NaN.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
