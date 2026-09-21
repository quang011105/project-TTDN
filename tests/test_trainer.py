"""
Unit tests cho src.trainer — Vòng lặp huấn luyện, Early Stopping và Checkpoint.

Chạy:
    python -m unittest tests/test_trainer.py -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.trainer import Trainer, set_seed


class TestTrainer(unittest.TestCase):
    """Kiểm thử khởi tạo Trainer và các phương thức train/validate."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "model": {
                "encoder_name": "resnet34",
                "encoder_weights": None,
                "in_channels": 4,
                "num_classes": 7,
            },
            "data": {
                "patches_dir": "data/processed/patches",
                "batch_size": 4,
                "num_workers": 0,
            },
            "training": {
                "max_epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "freeze_encoder_epochs": 1,
                "gradient_accumulation_steps": 1,
                "mixed_precision": False,
                "early_stopping_patience": 2,
                "seed": 42,
            },
            "loss": {
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "ignore_index": 0,
                "class_weights_path": "data/processed/class_weights.json",
            },
            "output": {
                "checkpoint_dir": str(self.tmp_dir / "checkpoints"),
                "log_dir": str(self.tmp_dir / "logs"),
            },
        }
        self.device = torch.device("cpu")
        self.trainer = Trainer(config=self.config, device=self.device)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_trainer_initialization(self) -> None:
        """Kiểm tra khởi tạo các module của Trainer."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
        self.assertIsNotNone(self.trainer.criterion)
        self.assertTrue(self.trainer.csv_path.exists())

    def test_save_checkpoint(self) -> None:
        """Kiểm tra lưu checkpoint best_model.pth và last_model.pth."""
        self.trainer.save_checkpoint(epoch=1, val_mIoU=0.55, is_best=True)
        best_ckpt = self.tmp_dir / "checkpoints" / "best_model.pth"
        last_ckpt = self.tmp_dir / "checkpoints" / "last_model.pth"

        self.assertTrue(best_ckpt.exists())
        self.assertTrue(last_ckpt.exists())

        loaded = torch.save, torch.load(best_ckpt, map_location="cpu")
        data = loaded[1]
        self.assertEqual(data["epoch"], 1)
        self.assertEqual(data["val_mIoU"], 0.55)
        self.assertIn("model_state_dict", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
