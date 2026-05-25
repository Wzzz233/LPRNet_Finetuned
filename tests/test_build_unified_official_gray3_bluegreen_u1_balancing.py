import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SCRIPT = ROOT / 'src' / 'utils' / 'build_unified_official_gray3_bluegreen_u1.py'
spec = importlib.util.spec_from_file_location('build_unified_official_gray3_bluegreen_u1', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class U1BalancedManifestTests(unittest.TestCase):
    def test_balance_province_weights_downweights_anhui_without_deleting_rows(self):
        rows = [
            {'split': 'train', 'family': 'green8', 'text': '皖AD00001', 'difficulty_bucket': 'real'},
            {'split': 'train', 'family': 'green8', 'text': '皖AD00002', 'difficulty_bucket': 'simple'},
            {'split': 'train', 'family': 'green8', 'text': '苏AD00001', 'difficulty_bucket': 'extreme'},
            {'split': 'train', 'family': 'normal7', 'text': '皖A12345', 'difficulty_bucket': 'real'},
            {'split': 'train', 'family': 'normal7', 'text': '皖B12345', 'difficulty_bucket': 'real'},
            {'split': 'train', 'family': 'normal7', 'text': '粤B12345', 'difficulty_bucket': 'real'},
        ]
        balanced = mod.apply_balanced_sample_weights(rows, province_power=1.0, family_power=0.0, difficulty_boosts={'extreme': 3.0})
        self.assertEqual(len(balanced), len(rows))
        green_anhui = [r for r in balanced if r['family'] == 'green8' and r['text'].startswith('皖')]
        green_su = [r for r in balanced if r['family'] == 'green8' and r['text'].startswith('苏')][0]
        blue_anhui = [r for r in balanced if r['family'] == 'normal7' and r['text'].startswith('皖')]
        blue_yue = [r for r in balanced if r['family'] == 'normal7' and r['text'].startswith('粤')][0]
        self.assertTrue(all(float(r['sample_weight']) < float(green_su['sample_weight']) for r in green_anhui))
        self.assertTrue(all(float(r['sample_weight']) < float(blue_yue['sample_weight']) for r in blue_anhui))
        self.assertEqual(green_su['difficulty_bucket'], 'extreme')
        self.assertGreater(float(green_su['sample_weight']), 1.0)

    def test_attach_extra_extreme_rows_marks_extreme_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            extra = td_path / 'extra.csv'
            extra.write_text(
                'img_path,img_rel_path,dataset_name,split,text,plate_len,family,sub_type,source,is_real,need_tilt_aug,preprocess_group,has_bbox,has_quad,can_parse_ccpd_geom,can_perspective,bbox_source,quad_source,ocr_channel_order,ocr_crop_mode,ocr_resize_mode,ocr_resize_kernel,ocr_preproc,ocr_min_occ_ratio,ocr_quad_pad_ratio\n'
                '/tmp/x.jpg,images/train/extreme/x.jpg,extra_extreme,train,苏AD12345,8,green8,green_small,v4_boardlike_edgefit,0,1,ccpd_board,1,1,1,1,det,det,bgr,obb_warp,letterbox,nn,gray3,0.9,0.0\n',
                encoding='utf-8'
            )
            base_rows = [{'split': 'train', 'family': 'green8', 'text': '皖AD00001', 'source_root_tag': 'e2_green', 'difficulty_bucket': 'simple'}]
            merged = mod.attach_extra_manifest(base_rows, extra, 'u1_extra_extreme', forced_difficulty='extreme')
            self.assertEqual(len(merged), 2)
            self.assertEqual(merged[-1]['difficulty_bucket'], 'extreme')
            self.assertEqual(merged[-1]['source_root_tag'], 'u1_extra_extreme')
            self.assertEqual(merged[0]['text'], '皖AD00001')

    def test_attach_directory_tree_rows_scans_boardlike_and_tier3_extremes(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tier3 = td_path / 'green_edgefit_tier3_full_v2' / 'images' / 'train' / 'extreme' / 'p10_u82cf'
            tier3.mkdir(parents=True)
            tier3_img = tier3 / 'edgefit-tier3-苏AD12345-sample.jpg'
            tier3_img.write_bytes(b'x')
            a3000 = td_path / 'green_edgefit_v4_boardlike_a3000' / 'images' / 'train' / 'board_extreme_tail' / 'p01_u6caa'
            a3000.mkdir(parents=True)
            a3000_img = a3000 / 'edgefit4-board_extreme_tail-100&71_245&71-100&71_245&71_245&71_140&71-沪AD54321-board_extreme_tail-001.jpg'
            a3000_img.write_bytes(b'y')

            rows = mod.attach_extra_tree_rows(
                [],
                td_path / 'green_edgefit_tier3_full_v2',
                source_tag='u1_tier3_v3_extreme',
                dataset_name='green_edgefit_tier3_full_v2',
                source_name='synthetic_edgefit_tier3_v3',
                bucket_to_difficulty={'simple': 'simple', 'hard': 'hard', 'extreme': 'extreme'},
                split_allowlist={'train'},
            )
            rows = mod.attach_extra_tree_rows(
                rows,
                td_path / 'green_edgefit_v4_boardlike_a3000',
                source_tag='u1_a3000_extreme',
                dataset_name='green_edgefit_v4_boardlike_a3000',
                source_name='v4_boardlike_edgefit_a3000',
                bucket_to_difficulty={'geometry_clean': 'simple', 'board_mid_occ': 'hard', 'board_low_occ': 'hard', 'board_extreme_tail': 'extreme'},
                split_allowlist={'train'},
            )
            self.assertEqual(len(rows), 2)
            by_tag = {r['source_root_tag']: r for r in rows}
            self.assertEqual(by_tag['u1_tier3_v3_extreme']['difficulty_bucket'], 'extreme')
            self.assertEqual(by_tag['u1_a3000_extreme']['difficulty_bucket'], 'extreme')
            self.assertEqual(by_tag['u1_tier3_v3_extreme']['text'], '苏AD12345')
            self.assertEqual(by_tag['u1_a3000_extreme']['text'], '沪AD54321')
            self.assertEqual(by_tag['u1_a3000_extreme']['preprocess_group'], 'ccpd_board')

    def test_summarize_weight_stats_reports_family_province_spread(self):
        rows = [
            {'split': 'train', 'family': 'green8', 'text': '皖AD00001', 'sample_weight': '0.5'},
            {'split': 'train', 'family': 'green8', 'text': '苏AD00001', 'sample_weight': '1.5'},
            {'split': 'train', 'family': 'normal7', 'text': '皖A12345', 'sample_weight': '0.6'},
            {'split': 'train', 'family': 'normal7', 'text': '粤B12345', 'sample_weight': '1.4'},
        ]
        stats = mod.summarize_weight_stats(rows)
        self.assertIn('green8', stats['family_weight_stats'])
        self.assertIn('normal7', stats['family_weight_stats'])
        self.assertLess(stats['province_weight_stats']['green8']['皖']['mean_weight'], stats['province_weight_stats']['green8']['苏']['mean_weight'])
        self.assertLess(stats['province_weight_stats']['normal7']['皖']['mean_weight'], stats['province_weight_stats']['normal7']['粤']['mean_weight'])


if __name__ == '__main__':
    unittest.main()
