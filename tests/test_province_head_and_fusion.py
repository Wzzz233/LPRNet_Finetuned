import sys
import unittest
from pathlib import Path

import torch

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src' / 'utils', ROOT / 'src' / 'training', ROOT / 'src' / 'evaluation']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from LPRNet_multihead import build_lprnet_multihead_from_state_dict, infer_checkpoint_multihead_config, load_multihead_state_dict_compat
from firstchar_fusion import extract_province_logits, fuse_first_char
from train_LPRNet import get_parser


class ProvinceHeadAndFusionTests(unittest.TestCase):
    def test_infers_family_specific_province_heads_from_checkpoint(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
            'province_family_heads.green8.4.weight': torch.zeros(31, 128),
        }

        cfg = infer_checkpoint_multihead_config(state)

        self.assertEqual(cfg['province_target_families'], ['green8'])
        self.assertTrue(cfg['has_any_province'])
        self.assertFalse(cfg['has_shared_province'])
        self.assertEqual(cfg['province_num_classes'], 31)

    def test_prefers_primary_class_num_when_aux_head_is_legacy(self):
        state = {
            'backbone.20.weight': torch.zeros(68, 256, 13, 1),
            'backbone.21.weight': torch.zeros(68),
            'containers.green8.0.weight': torch.zeros(256, 448 + 68, 3, 3),
            'province_family_heads.green8.0.weight': torch.zeros(128, 448 + 66, 3, 3),
            'province_family_heads.green8.4.weight': torch.zeros(31, 128),
        }

        cfg = infer_checkpoint_multihead_config(state)

        self.assertEqual(cfg['class_num'], 68)

    def test_build_model_restores_province_family_head(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
            'province_family_heads.green8.4.weight': torch.zeros(31, 128),
        }

        net, cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=66,
            dropout_rate=0.0,
        )

        self.assertEqual(cfg['province_target_families'], ['green8'])
        self.assertIn('green8', net.province_family_heads)

    def test_build_model_uses_checkpoint_class_num_for_province_head(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
            'province_family_heads.green8.0.weight': torch.zeros(128, 448 + 66, 3, 3),
            'province_family_heads.green8.4.weight': torch.zeros(31, 128),
        }

        net, cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=68,
            dropout_rate=0.0,
        )

        self.assertEqual(cfg['class_num'], 66)
        self.assertEqual(net.class_num, 66)
        self.assertEqual(net.province_family_heads['green8'][0].weight.shape[1], 448 + 66)

    def test_load_multihead_state_dict_compat_pads_legacy_province_head_channels(self):
        state = {
            'backbone.20.weight': torch.zeros(68, 256, 13, 1),
            'backbone.21.weight': torch.zeros(68),
            'containers.green8.0.weight': torch.zeros(256, 448 + 68, 3, 3),
            'province_family_heads.green8.0.weight': torch.ones(128, 448 + 66, 3, 3),
            'province_family_heads.green8.0.bias': torch.ones(128),
            'province_family_heads.green8.4.weight': torch.ones(31, 128),
            'province_family_heads.green8.4.bias': torch.ones(31),
        }

        net, cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=68,
            dropout_rate=0.0,
        )
        _, adapted_keys = load_multihead_state_dict_compat(net, state, strict=False)

        self.assertEqual(cfg['class_num'], 68)
        self.assertTrue(any(key == 'province_family_heads.green8.0.weight' for key, _, _ in adapted_keys))
        self.assertEqual(net.province_family_heads['green8'][0].weight.shape[1], 448 + 68)

    def test_extract_province_logits_prefers_family_specific_outputs(self):
        raw = {
            'province_green8': torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        }
        logits = extract_province_logits(raw, ['green8', 'green8'])
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertAlmostEqual(float(logits[0, 1]), 3.0)
        self.assertAlmostEqual(float(logits[1, 1]), 4.0)

    def test_extract_province_logits_tolerates_missing_non_target_family_head(self):
        raw = {
            'province_green8': torch.tensor([[1.0, 3.0]]),
        }
        logits = extract_province_logits(raw, ['green8', 'normal7'])
        self.assertIsNotNone(logits)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertAlmostEqual(float(logits[0, 1]), 3.0)

    def test_extract_province_logits_tolerates_batch_without_target_family(self):
        raw = {
            'province_green8': torch.tensor([[1.0, 3.0]]),
        }
        logits = extract_province_logits(raw, ['normal7'])
        self.assertIsNotNone(logits)
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_fuse_first_char_replace_if_confident(self):
        fused, changed, reason = fuse_first_char('皖AD06088', '京', 0.83, 'replace_if_confident', 0.55)
        self.assertEqual(fused, '京AD06088')
        self.assertTrue(changed)
        self.assertEqual(reason, 'replace_if_confident')

    def test_train_parser_accepts_province_head_args(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                'train_LPRNet.py',
                '--data_mode', 'manifest',
                '--train_manifest', '/tmp/train.csv',
                '--test_manifest', '/tmp/test.csv',
                '--province_head_weight', '0.30',
                '--province_num_classes', '31',
                '--province_target_families', 'green8',
            ]
            args = get_parser()
        finally:
            sys.argv = old_argv

        self.assertAlmostEqual(args.province_head_weight, 0.30)
        self.assertEqual(args.province_num_classes, 31)
        self.assertEqual(args.province_target_families, 'green8')


if __name__ == '__main__':
    unittest.main()
