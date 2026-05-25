import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quad_refiner.dataset import (  # noqa: E402
    build_ccpd_record,
    build_crpd_records,
    generate_corner_heatmaps,
    rasterize_quad_mask,
)
from quad_refiner.decode import decode_corner_heatmaps  # noqa: E402
from quad_refiner.geometry import (  # noqa: E402
    build_patch_box_from_quad,
    gate_refined_quad,
    map_quad_from_patch,
    map_quad_to_patch,
)
from quad_refiner.model import QuadHeatmapRefiner  # noqa: E402


class QuadRefinerGeometryTests(unittest.TestCase):
    def test_patch_box_roundtrip_preserves_quad(self):
        quad = np.asarray([[100, 40], [180, 44], [178, 78], [98, 74]], dtype=np.float32)
        patch_box = build_patch_box_from_quad(quad, img_w=300, img_h=120, pad_x=0.2, pad_y=0.25)
        quad_patch = map_quad_to_patch(quad, patch_box, out_w=256, out_h=128)
        quad_back = map_quad_from_patch(quad_patch, patch_box, in_w=256, in_h=128)
        self.assertTrue(np.allclose(quad, quad_back, atol=1.0), msg=f"quad={quad} quad_back={quad_back}")

    def test_gate_accepts_reasonable_refine_and_rejects_bad_quad(self):
        coarse = np.asarray([[100, 40], [180, 44], [178, 78], [98, 74]], dtype=np.float32)
        refined_good = np.asarray([[102, 41], [179, 43], [177, 77], [100, 75]], dtype=np.float32)
        decision_good = gate_refined_quad(coarse, refined_good, corner_conf=[0.95, 0.92, 0.91, 0.90], patch_diag=286.0)
        self.assertTrue(decision_good.accepted, msg=decision_good.reason)

        refined_bad = np.asarray([[70, 20], [230, 80], [60, 90], [210, 30]], dtype=np.float32)
        decision_bad = gate_refined_quad(coarse, refined_bad, corner_conf=[0.99, 0.99, 0.99, 0.99], patch_diag=286.0)
        self.assertFalse(decision_bad.accepted)


class QuadRefinerDatasetTests(unittest.TestCase):
    def test_corner_heatmaps_and_mask_have_expected_shapes(self):
        quad_patch = np.asarray([[32, 10], [92, 12], [90, 24], [30, 22]], dtype=np.float32)
        heatmaps = generate_corner_heatmaps(quad_patch, in_w=128, in_h=32, out_w=64, out_h=16, sigma=1.5)
        mask = rasterize_quad_mask(quad_patch, in_w=128, in_h=32, out_w=64, out_h=16)
        self.assertEqual(heatmaps.shape, (4, 16, 64))
        self.assertEqual(mask.shape, (1, 16, 64))
        self.assertGreater(float(heatmaps.max()), 0.8)
        self.assertGreater(int(mask.sum()), 20)

    def test_build_ccpd_and_crpd_records_parse_quads(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ccpd_img = td_path / "0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg"
            cv2.imwrite(str(ccpd_img), np.full((600, 800, 3), 180, dtype=np.uint8))
            rec = build_ccpd_record(ccpd_img, split="train", source_name="ccpd2019")
            self.assertEqual(rec["source_name"], "ccpd2019")
            self.assertEqual(len(rec["gt_quad"]), 4)
            self.assertEqual(rec["split"], "train")

            crpd_img = td_path / "61_0686.jpg"
            crpd_label = td_path / "61_0686.txt"
            cv2.imwrite(str(crpd_img), np.full((1200, 1600, 3), 100, dtype=np.uint8))
            crpd_label.write_text("1389 926 1483 925 1483 959 1389 959 0 川FKX755\n811 162 933 163 928 176 808 166 0 川B87763\n", encoding="utf-8")
            records = build_crpd_records(crpd_img, crpd_label, split="train", source_name="crpd_raw")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["text"], "川FKX755")
            self.assertEqual(records[1]["source_name"], "crpd_raw")


class QuadRefinerModelTests(unittest.TestCase):
    def test_model_forward_and_decode_shapes(self):
        import torch

        model = QuadHeatmapRefiner(pretrained=False)
        x = torch.randn(2, 3, 128, 256)
        outputs = model(x)
        self.assertEqual(tuple(outputs["heatmaps"].shape), (2, 4, 32, 64))
        self.assertEqual(tuple(outputs["mask"].shape), (2, 1, 32, 64))

        heatmaps = outputs["heatmaps"][0].detach().cpu().numpy()
        points, confs = decode_corner_heatmaps(heatmaps, in_w=256, in_h=128)
        self.assertEqual(points.shape, (4, 2))
        self.assertEqual(len(confs), 4)


if __name__ == "__main__":
    unittest.main()
