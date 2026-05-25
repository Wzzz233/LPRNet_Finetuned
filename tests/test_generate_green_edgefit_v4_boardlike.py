import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import sys

ROOT = Path('/home/wzzz/LPRNet')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'src' / 'utils'))
SCRIPT = ROOT / 'src' / 'utils' / 'generate_green_edgefit_v4_boardlike.py'
spec = importlib.util.spec_from_file_location('generate_green_edgefit_v4_boardlike', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class GreenEdgefitV4BoardlikeTests(unittest.TestCase):
    def test_build_sample_record_contains_board_metrics(self):
        img = np.full((72, 246, 3), 180, dtype=np.uint8)
        exact_quad = np.array([[10, 10], [235, 8], [238, 60], [12, 64]], dtype=np.float32)
        board_quad = np.array([[18, 12], [210, 10], [188, 58], [20, 62]], dtype=np.float32)

        def fake_prepare_fn(image, quad, in_w, in_h, resize_mode, resize_kernel, preproc_mode, channel_order, quad_pad_ratio=0.0):
            warped = np.full((52, 180, 3), 127, dtype=np.uint8)
            prepared = np.full((24, 94, 3), 120, dtype=np.uint8)
            return prepared, 0.69, warped, np.asarray(quad, dtype=np.float32), np.eye(3, dtype=np.float32)

        record = mod.build_sample_record(
            img=img,
            split='train',
            bucket='board_low_occ',
            province='苏',
            text='苏AD12345',
            exact_quad=exact_quad,
            board_quad=board_quad,
            asym_mode='left_compressed',
            appearance_meta={'blur_strength': 1.2, 'jpeg_quality': 42, 'appearance_mode': 'board_low_occ'},
            out_root=Path(tempfile.mkdtemp(prefix='edgefit_v4_test_')),
            uid='unit-case',
            prepare_fn=fake_prepare_fn,
        )

        self.assertEqual(record['bucket'], 'board_low_occ')
        self.assertEqual(record['quad_mode'], 'board_like')
        self.assertIn('occ_ratio', record)
        self.assertIn('warped_aspect', record)
        self.assertIn('left_right_width_ratio', record)
        self.assertGreater(record['occ_ratio'], 0.0)
        self.assertTrue(Path(record['abs_path']).exists())

    def test_accept_bucket_for_occ_ratio(self):
        self.assertTrue(mod.accept_bucket('board_mid_occ', 0.80))
        self.assertFalse(mod.accept_bucket('board_mid_occ', 0.68))
        self.assertTrue(mod.accept_bucket('board_low_occ', 0.68))
        self.assertTrue(mod.accept_bucket('board_extreme_tail', 0.58))
        self.assertFalse(mod.accept_bucket('geometry_clean', 0.70))

    def test_write_outputs_emits_extended_tsv(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            img = np.full((72, 246, 3), 200, dtype=np.uint8)
            rel = 'images/train/board_mid_occ/p00_u4eac/sample.jpg'
            abs_path = out_root / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(abs_path), img)
            rows = [{
                'split': 'train',
                'bucket': 'board_mid_occ',
                'province': '京',
                'text': '京AD12345',
                'rel_path': rel,
                'abs_path': str(abs_path),
                'exact_quad': [[0, 0], [245, 0], [245, 71], [0, 71]],
                'board_quad': [[5, 4], [220, 3], [210, 62], [6, 63]],
                'quad_mode': 'board_like',
                'occ_ratio': 0.79,
                'warped_w': 220,
                'warped_h': 60,
                'warped_aspect': 3.66,
                'left_right_width_ratio': 1.25,
                'asym_mode': 'left_compressed',
                'blur_strength': 1.1,
                'jpeg_quality': 40,
                'appearance_mode': 'board_mid_occ',
                'source_tag': 'synthetic_edgefit_v4min',
                'manifest_row': {
                    'img_path': str(abs_path),
                    'img_rel_path': rel,
                    'dataset_name': 'green_edgefit_v4_boardlike_a3000',
                    'split': 'train',
                    'text': '京AD12345',
                    'plate_len': 8,
                    'family': 'green8',
                    'sub_type': 'green_small',
                    'source': 'v4_boardlike_edgefit',
                    'is_real': 0,
                    'need_tilt_aug': 1,
                    'preprocess_group': 'ccpd_board',
                    'has_bbox': 1,
                    'has_quad': 1,
                    'can_parse_ccpd_geom': 1,
                    'can_perspective': 1,
                    'bbox_source': 'v4_boardlike_edgefit',
                    'quad_source': 'v4_boardlike_edgefit',
                    'ocr_channel_order': 'bgr',
                    'ocr_crop_mode': 'obb_warp',
                    'ocr_resize_mode': 'letterbox',
                    'ocr_resize_kernel': 'nn',
                    'ocr_preproc': 'none',
                    'ocr_min_occ_ratio': 0.9,
                    'ocr_quad_pad_ratio': 0.0,
                },
            }]
            split_texts = {'train': {'京AD12345'}, 'val': set(), 'test': set()}
            report = mod.write_outputs(rows, split_texts, out_root)
            self.assertEqual(report['total'], 1)
            tsv_path = out_root / 'details' / 'accepted.tsv'
            self.assertTrue(tsv_path.exists())
            text = tsv_path.read_text(encoding='utf-8')
            self.assertIn('occ_ratio', text.splitlines()[0])
            self.assertIn('board_mid_occ', text)
            manifest_path = out_root / 'manifests' / 'train_manifest_v4.csv'
            self.assertTrue(manifest_path.exists())
            self.assertIn('dataset_name', manifest_path.read_text(encoding='utf-8').splitlines()[0])

    def test_train_target_by_province_caps_anhui(self):
        targets = mod.train_target_by_province(3000, 0.15)
        self.assertEqual(sum(targets.values()), 3000)
        self.assertEqual(targets['皖'], 450)
        self.assertEqual(sum(v for k, v in targets.items() if k != '皖'), 2550)

    def test_allocate_bucket_anhui_counts_exact_total(self):
        quotas = {
            'geometry_clean': 450,
            'board_mid_occ': 1050,
            'board_low_occ': 900,
            'board_extreme_tail': 600,
        }
        counts = mod.allocate_bucket_anhui_counts(quotas, 450)
        self.assertEqual(sum(counts.values()), 450)
        self.assertEqual(counts['geometry_clean'], 67)
        self.assertEqual(counts['board_mid_occ'], 158)
        self.assertEqual(counts['board_low_occ'], 135)
        self.assertEqual(counts['board_extreme_tail'], 90)

    def test_enforce_vertical_margins_caps_top_bottom_black_bars(self):
        quad = np.array([[25, 1], [170, 2], [165, 69], [28, 68]], dtype=np.float32)
        rng = random.Random(123)
        fixed = mod.enforce_vertical_margins(quad, 'board_extreme_tail', rng)
        top_margin = float(min(fixed[0, 1], fixed[1, 1]))
        bottom_margin = float((mod.CANVAS_H - 1) - max(fixed[2, 1], fixed[3, 1]))
        self.assertGreaterEqual(top_margin, 0.2)
        self.assertLessEqual(top_margin, 1.2)
        self.assertGreaterEqual(bottom_margin, 0.2)
        self.assertLessEqual(bottom_margin, 1.2)

    def test_attempt_budget_for_bucket_scales_extreme_more_aggressively(self):
        self.assertEqual(mod.attempt_budget_for_bucket('geometry_clean', 1), 50)
        self.assertEqual(mod.attempt_budget_for_bucket('board_mid_occ', 1), 80)
        self.assertEqual(mod.attempt_budget_for_bucket('board_low_occ', 1), 120)
        self.assertEqual(mod.attempt_budget_for_bucket('board_extreme_tail', 1), 800)
        self.assertEqual(mod.attempt_budget_for_bucket('board_extreme_tail', 9), 2880)


if __name__ == '__main__':
    unittest.main()
