import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from micro_rectifier.dataset import MicroRectifierDataset, build_record_from_crop_pair  # noqa: E402
from micro_rectifier.model import MicroRectifier  # noqa: E402
from micro_rectifier.geometry import apply_parametric_warp_bgr  # noqa: E402


class MicroRectifierDatasetTests(unittest.TestCase):
    def test_build_record_and_dataset_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = np.zeros((48, 160, 3), dtype=np.uint8)
            cv2.rectangle(src, (20, 10), (140, 38), (0, 255, 0), -1)
            tgt = src.copy()
            src_path = td / "src.png"
            tgt_path = td / "tgt.png"
            cv2.imwrite(str(src_path), src)
            cv2.imwrite(str(tgt_path), tgt)

            rec = build_record_from_crop_pair(
                sample_id="x",
                input_path=src_path,
                target_path=tgt_path,
                text="京AD06088",
                split="train",
                source_name="unit",
                dx=0.0,
                dy=0.0,
                sx=1.0,
                sy=1.0,
                shx=0.0,
            )
            ds = MicroRectifierDataset([rec], input_size=(160, 48), grayscale=False)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 48, 160))
            self.assertEqual(tuple(item["target_image"].shape), (3, 48, 160))
            self.assertEqual(tuple(item["params"].shape), (5,))
            self.assertEqual(item["text"], "京AD06088")


class MicroRectifierGeometryTests(unittest.TestCase):
    def test_identity_warp_keeps_image_close(self):
        img = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
        warped = apply_parametric_warp_bgr(img, dx=0.0, dy=0.0, sx=1.0, sy=1.0, shx=0.0)
        self.assertLess(np.abs(warped.astype(np.int16) - img.astype(np.int16)).mean(), 1.0)


class MicroRectifierModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        net = MicroRectifier(in_channels=3, width_mult=1.0)
        x = torch.randn(2, 3, 48, 160)
        out = net(x)
        self.assertEqual(tuple(out["params"].shape), (2, 5))
        self.assertEqual(tuple(out["rectified"].shape), (2, 3, 48, 160))


if __name__ == "__main__":
    unittest.main()
