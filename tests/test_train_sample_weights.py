import unittest

from train_LPRNet import build_sample_weights


class TrainSampleWeightTests(unittest.TestCase):
    def test_manifest_sample_weight_multiplies_existing_main_weight(self):
        texts = ['皖AD00001', '苏AD00001']
        sample_sources = ['main', 'main']
        sample_metas = [
            {'sample_weight': '0.5', 'family': 'green8'},
            {'sample_weight': '2.0', 'family': 'green8'},
        ]
        sample_weights, _, _, debug = build_sample_weights(
            texts,
            sample_sources,
            sample_metas,
            province_mode='none',
            province_clip=0.0,
            strata_mode='none',
            strata_clip=0.0,
            board_anchor_sample_weight=1.0,
            pseudo_anchor_sample_weight=1.0,
            secondary_train_sample_weight=1.0,
            adj_repeat_sample_weight=1.0,
            main_group_by='none',
            main_group_ratios='',
            main_group_clip=0.0,
        )
        self.assertAlmostEqual(float(sample_weights[0]), 0.5, places=6)
        self.assertAlmostEqual(float(sample_weights[1]), 2.0, places=6)
        self.assertEqual(debug['manifest_sample_weight_applied'], 2)

    def test_manifest_sample_weight_compounds_with_main_group_ratio(self):
        texts = ['皖AD00001', '苏AD00001', '粤A12345']
        sample_sources = ['main', 'main', 'main']
        sample_metas = [
            {'family': 'green8', 'sample_weight': '1.0'},
            {'family': 'green8', 'sample_weight': '2.0'},
            {'family': 'normal7', 'sample_weight': '3.0'},
        ]
        sample_weights, _, _, debug = build_sample_weights(
            texts,
            sample_sources,
            sample_metas,
            province_mode='none',
            province_clip=0.0,
            strata_mode='none',
            strata_clip=0.0,
            board_anchor_sample_weight=1.0,
            pseudo_anchor_sample_weight=1.0,
            secondary_train_sample_weight=1.0,
            adj_repeat_sample_weight=1.0,
            main_group_by='family',
            main_group_ratios='green8=0.5,normal7=0.5',
            main_group_clip=0.0,
        )
        self.assertAlmostEqual(float(debug['main_group_weights']['green8']), 1.0, places=6)
        self.assertAlmostEqual(float(debug['main_group_weights']['normal7']), 1.0, places=6)
        self.assertAlmostEqual(float(sample_weights[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sample_weights[1]), 2.0, places=6)
        self.assertAlmostEqual(float(sample_weights[2]), 3.0, places=6)

    def test_invalid_manifest_sample_weight_falls_back_to_one(self):
        texts = ['皖AD00001']
        sample_sources = ['main']
        sample_metas = [{'sample_weight': 'not-a-number', 'family': 'green8'}]
        sample_weights, _, _, debug = build_sample_weights(
            texts,
            sample_sources,
            sample_metas,
            province_mode='none',
            province_clip=0.0,
            strata_mode='none',
            strata_clip=0.0,
            board_anchor_sample_weight=1.0,
            pseudo_anchor_sample_weight=1.0,
            secondary_train_sample_weight=1.0,
            adj_repeat_sample_weight=1.0,
            main_group_by='none',
            main_group_ratios='',
            main_group_clip=0.0,
        )
        self.assertAlmostEqual(float(sample_weights[0]), 1.0, places=6)
        self.assertEqual(debug['manifest_sample_weight_invalid'], 1)


if __name__ == '__main__':
    unittest.main()
