import csv
import os
import tempfile
import unittest
from pathlib import Path

from build_unified_manifest import build_row_from_label_entry, write_manifest
from load_data import UnifiedManifestDataset


class UnifiedManifestTests(unittest.TestCase):
    def test_build_row_for_ccpd_board_has_geometry_flags(self):
        rel_path = "ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg"
        row = build_row_from_label_entry(
            img_root="/data/CCPD2019",
            rel_path=rel_path,
            text="皖A45S58",
            split="train",
            family="normal7",
            sub_type="blue",
            source="real",
            is_real=1,
            need_tilt_aug=1,
            dataset_name="ccpd2019",
        )
        self.assertEqual(row["preprocess_group"], "ccpd_board")
        self.assertEqual(row["has_bbox"], 1)
        self.assertEqual(row["has_quad"], 1)
        self.assertEqual(row["can_perspective"], 1)
        self.assertEqual(row["can_parse_ccpd_geom"], 1)

    def test_build_row_for_plain_plate_disables_perspective(self):
        rel_path = "plain/plate001.jpg"
        row = build_row_from_label_entry(
            img_root="/data/plain",
            rel_path=rel_path,
            text="京A12345",
            split="train",
            family="normal7",
            sub_type="blue",
            source="real",
            is_real=1,
            need_tilt_aug=0,
            dataset_name="plain_blue",
        )
        self.assertEqual(row["preprocess_group"], "plain_plate")
        self.assertEqual(row["has_bbox"], 0)
        self.assertEqual(row["has_quad"], 0)
        self.assertEqual(row["can_perspective"], 0)
        self.assertEqual(row["can_parse_ccpd_geom"], 0)

    def test_unified_manifest_dataset_reads_mixed_groups(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ccpd_root = td_path / "CCPD2019"
            plain_root = td_path / "plain"
            ccpd_root.mkdir()
            plain_root.mkdir()

            import cv2
            import numpy as np

            ccpd_name = "ccpd_base/0092816091954-94_82-181&490_358&548-363&554_189&540_190&484_364&498-0_0_28_29_16_29_32-133-13.jpg"
            ccpd_img = ccpd_root / ccpd_name
            ccpd_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ccpd_img), np.full((600, 800, 3), 180, dtype=np.uint8))

            plain_rel = "plate001.jpg"
            plain_img = plain_root / plain_rel
            cv2.imwrite(str(plain_img), np.full((24, 94, 3), 200, dtype=np.uint8))

            rows = [
                build_row_from_label_entry(
                    img_root=str(ccpd_root),
                    rel_path=ccpd_name,
                    text="皖A45S58",
                    split="train",
                    family="normal7",
                    sub_type="blue",
                    source="real",
                    is_real=1,
                    need_tilt_aug=1,
                    dataset_name="ccpd2019",
                ),
                build_row_from_label_entry(
                    img_root=str(plain_root),
                    rel_path=plain_rel,
                    text="京A12345",
                    split="train",
                    family="normal7",
                    sub_type="blue",
                    source="real",
                    is_real=1,
                    need_tilt_aug=0,
                    dataset_name="plain_blue",
                ),
            ]
            manifest_path = td_path / "manifest.csv"
            write_manifest(rows, manifest_path)

            dataset = UnifiedManifestDataset(
                manifest_path=str(manifest_path),
                img_size=[94, 24],
                lpr_max_len=8,
                split_filter="train",
                ocr_channel_order="bgr",
                ocr_crop_mode="obb_warp",
                ocr_resize_mode="letterbox",
                ocr_resize_kernel="nn",
                ocr_preproc="none",
                ocr_min_occ_ratio=0.90,
                ocr_quad_pad_ratio=0.0,
            )

            self.assertEqual(len(dataset), 2)
            img0, label0, len0 = dataset[0]
            img1, label1, len1 = dataset[1]
            self.assertEqual(tuple(img0.shape), (3, 24, 94))
            self.assertEqual(tuple(img1.shape), (3, 24, 94))
            self.assertEqual(len0, 7)
            self.assertEqual(len1, 7)
            self.assertEqual(len(label0), 7)
            self.assertEqual(len(label1), 7)


if __name__ == "__main__":
    unittest.main()
