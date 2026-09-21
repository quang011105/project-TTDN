"""
Unit tests cho src.model — Kiến trúc mô hình U-Net (backbone ResNet34) 4 kênh.

Chạy:
    python -m unittest tests/test_model.py -v
"""

import unittest
from typing import Dict

import torch
import torch.nn as nn

from src.config import IN_CHANNELS, NUM_CLASSES
from src.model import (
    build_unet,
    count_parameters,
    freeze_encoder,
    get_device,
    summary_model,
    unfreeze_encoder,
)


class TestUnetModel(unittest.TestCase):
    """Kiểm thử khởi tạo mô hình, forward pass, freeze/unfreeze và gradient."""

    @classmethod
    def setUpClass(cls) -> None:
        # Sử dụng encoder_weights=None để unit test chạy nhanh và độc lập mạng Internet
        cls.model = build_unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
        )

    def test_model_type_and_conv1_channels(self) -> None:
        """Kiểm tra lớp Conv đầu vào của encoder thích ứng đúng 4 kênh (B2, B3, B4, B8)."""
        conv1 = self.model.encoder.conv1
        self.assertIsInstance(conv1, nn.Conv2d)
        self.assertEqual(conv1.in_channels, 4, "Lớp conv1 phải nhận đúng 4 kênh đầu vào.")
        self.assertEqual(conv1.weight.shape, (64, 4, 7, 7))

    def test_forward_pass_output_shape(self) -> None:
        """Kiểm tra shape logits đầu ra là [B, 7, H, W] với đầu vào [B, 4, H, W]."""
        self.model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 4, 256, 256)

        with torch.no_grad():
            output = self.model(dummy_input)

        expected_shape = (batch_size, 7, 256, 256)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Kỳ vọng shape {expected_shape}, nhận được {output.shape}",
        )
        self.assertEqual(output.dtype, torch.float32)

    def test_count_parameters(self) -> None:
        """Kiểm tra hàm count_parameters trả về đủ các trường và tổng khớp."""
        params = count_parameters(self.model)
        for key in ("total", "trainable", "frozen"):
            self.assertIn(key, params)
        self.assertEqual(
            params["total"],
            params["trainable"] + params["frozen"],
            "Tổng tham số phải bằng trainable + frozen.",
        )
        # ResNet34 + UNet decoder ~ 24.4 triệu tham số
        self.assertGreater(params["total"], 20_000_000)

    def test_freeze_and_unfreeze_encoder(self) -> None:
        """Kiểm tra logic đóng băng (freeze) và mở khóa (unfreeze) encoder."""
        # 1. Freeze encoder
        freeze_encoder(self.model)
        for name, param in self.model.encoder.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Tham số encoder {name} phải có requires_grad=False khi bị freeze.",
            )

        # Kiểm tra decoder và segmentation_head vẫn còn mở khóa để train
        decoder_trainable = any(
            p.requires_grad for p in self.model.decoder.parameters()
        )
        self.assertTrue(decoder_trainable, "Decoder phải giữ requires_grad=True.")

        params_frozen = count_parameters(self.model)
        self.assertGreater(params_frozen["frozen"], 20_000_000)
        self.assertGreater(params_frozen["trainable"], 3_000_000)

        # 2. Unfreeze encoder
        unfreeze_encoder(self.model)
        for name, param in self.model.encoder.named_parameters():
            self.assertTrue(
                param.requires_grad,
                f"Tham số encoder {name} phải có requires_grad=True khi được unfreeze.",
            )

        params_unfrozen = count_parameters(self.model)
        self.assertEqual(params_unfrozen["frozen"], 0)
        self.assertEqual(params_unfrozen["trainable"], params_unfrozen["total"])

    def test_backward_pass_gradients_flow(self) -> None:
        """Kiểm tra lan truyền ngược (backward pass) và tính gradient chuẩn xác."""
        self.model.train()
        unfreeze_encoder(self.model)

        # Khởi tạo optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        dummy_input = torch.randn(1, 4, 128, 128)
        dummy_target = torch.randint(0, 7, (1, 128, 128), dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        output = self.model(dummy_input)
        loss = criterion(output, dummy_target)

        loss.backward()

        # Kiểm tra gradient xuất hiện ở cả conv1 của encoder và segmentation head
        self.assertIsNotNone(self.model.encoder.conv1.weight.grad)
        self.assertIsNotNone(self.model.segmentation_head[0].weight.grad)

    def test_get_device(self) -> None:
        """Kiểm tra tiện ích lấy thiết bị phần cứng."""
        device_cpu = get_device("cpu")
        self.assertEqual(device_cpu.type, "cpu")

        device_auto = get_device()
        self.assertIn(device_auto.type, ("cuda", "cpu"))

    def test_summary_model(self) -> None:
        """Kiểm tra hàm summary_model trả về đầy đủ metadata."""
        info = summary_model(self.model, input_size=(1, 4, 128, 128))
        self.assertEqual(info["model_class"], "Unet")
        self.assertEqual(info["input_shape"], (1, 4, 128, 128))
        self.assertEqual(info["output_shape"], (1, 7, 128, 128))
        self.assertGreater(info["total_params"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
