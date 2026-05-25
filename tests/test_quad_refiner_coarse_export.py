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

from quad_refiner.coarse_export import (  # noqa: E402
    build_ccpd_coarse_record,
    build_crpd_coarse_records,
    choose_best_obb,
    extract_obb_detections,
    load_split_paths,
    match_detections_to_gt,
)


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeOBB:
    def __init__(self, conf, quads, cls=None):
        self.conf = FakeTensor(conf)
        self.xyxyxyxy = FakeTensor(quads)
        self.cls = FakeTensor(cls if cls is not None else [0] * len(conf))


class FakeResult:
    def __init__(self, conf, quads, cls=None):
        self.obb = FakeOBB(conf, quads, cls)


class QuadRefinerCoarseExportTests(unittest.TestCase):
    def test_choose_best_obb_selects_highest_conf_and_orders_quad(self):
        result = FakeResult(
            conf=[0.3, 0.9],
            quads=[
                [[10, 10], [30, 10], [30, 20], [10, 20]],
                [[100, 50], [180, 52], [178, 80], [98, 78]],
            ],
            cls=[1, 2],
        )
        picked = choose_best_obb(result)
        self.assertIsNotNone(picked)
        self.assertAlmostEqual(picked["conf"], 0.9, places=6)
        self.assertEqual(picked["cls"], 2)
        self.assertEqual(picked["det_count"], 2)
        self.assertEqual(np.asarray(picked["quad"]).shape, (4, 2))

    def test_extract_obb_detections_returns_all_quads(self):
        result = FakeResult(
            conf=[0.5, 0.7],
            quads=[
                [[10, 10], [30, 10], [30, 20], [10, 20]],
                [[100, 50], [180, 52], [178, 80], [98, 78]],
            ],
        )
        dets = extract_obb_detections(result)
        self.assertEqual(len(dets), 2)
        self.assertAlmostEqual(dets[1]["conf"], 0.7, places=6)
        self.assertEqual(np.asarray(dets[0]["quad"]).shape, (4, 2))

    def test_build_ccpd_coarse_record_uses_expected_sample_id(self):
        img_path = Path("/data/CCPD2019/ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg")
        det = {
            "conf": 0.87,
            "cls": 0,
            "det_count": 1,
            "quad": [[363, 554], [189, 540], [190, 484], [364, 498]],
        }
        rec = build_ccpd_coarse_record(img_path, source_name="ccpd2019", det=det)
        self.assertEqual(rec["sample_id"], f"ccpd2019:{img_path.stem}")
        self.assertEqual(rec["image_path"], str(img_path))
        self.assertEqual(rec["source_name"], "ccpd2019")
        self.assertEqual(rec["coarse_quad"], det["quad"])
        self.assertAlmostEqual(rec["det_conf"], 0.87, places=6)

    def test_load_split_paths_reads_ccpd_train_txt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "CCPD2019"
            root.mkdir(parents=True, exist_ok=True)
            rel = "ccpd_base/abc-0-1&1_2&2-1&1_2&1_2_1&2-0-0.jpg"
            img = root / rel
            img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img), np.full((50, 100, 3), 127, dtype=np.uint8))
            split = root / "train.txt"
            split.write_text(rel + " 皖A12345\n", encoding="utf-8")
            paths = load_split_paths(root, split)
            self.assertEqual(paths, [img])

    def test_match_detections_to_gt_greedily(self):
        gt_records = [
            {"sample_id": "crpd_raw/CRPD_double:img_obj1", "gt_quad": [[100, 100], [150, 100], [150, 120], [100, 120]]},
            {"sample_id": "crpd_raw/CRPD_double:img_obj2", "gt_quad": [[300, 200], [360, 200], [360, 225], [300, 225]]},
        ]
        detections = [
            {"conf": 0.90, "cls": 0, "det_count": 2, "quad": [[98, 98], [152, 98], [152, 122], [98, 122]]},
            {"conf": 0.85, "cls": 0, "det_count": 2, "quad": [[302, 202], [358, 202], [358, 224], [302, 224]]},
        ]
        matched = match_detections_to_gt(gt_records, detections, min_iou=0.1)
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched[0]["sample_id"], "crpd_raw/CRPD_double:img_obj1")
        self.assertAlmostEqual(matched[1]["det_conf"], 0.85, places=6)

    def test_build_crpd_coarse_records_uses_native_gt_ids(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            img_path = td_path / "demo.jpg"
            label_path = td_path / "demo.txt"
            cv2.imwrite(str(img_path), np.full((600, 800, 3), 128, dtype=np.uint8))
            label_path.write_text(
                "100 100 150 100 150 120 100 120 0 苏A12345\n300 200 360 200 360 225 300 225 0 皖B54321\n",
                encoding="utf-8",
            )
            detections = [
                {"conf": 0.90, "cls": 0, "det_count": 2, "quad": [[98, 98], [152, 98], [152, 122], [98, 122]]},
                {"conf": 0.85, "cls": 0, "det_count": 2, "quad": [[302, 202], [358, 202], [358, 224], [302, 224]]},
            ]
            rows = build_crpd_coarse_records(img_path, label_path, split="train", source_name="crpd_raw/CRPD_double", detections=detections)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "crpd_raw/CRPD_double:demo_obj1")
            self.assertEqual(rows[1]["sample_id"], "crpd_raw/CRPD_double:demo_obj2")
            self.assertEqual(rows[0]["text"], "苏A12345")
            self.assertEqual(rows[1]["text"], "皖B54321")


if __name__ == "__main__":
    unittest.main()
