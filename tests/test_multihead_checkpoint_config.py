import sys
import unittest
from pathlib import Path

import torch

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src' / 'utils', ROOT / 'src' / 'training', ROOT / 'src' / 'evaluation']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from LPRNet_multihead import (
    DEFAULT_POS0_NUM_CLASSES,
    build_lprnet_multihead_from_state_dict,
    infer_checkpoint_multihead_config,
)


class MultiheadCheckpointConfigTests(unittest.TestCase):
    def test_infers_expd_shared_pos0_and_adapter_family(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
            'pos0_head.4.weight': torch.zeros(31, 128),
            'family_adapters.green8.0.weight': torch.zeros(128, 448 + 66, 3, 3),
        }

        cfg = infer_checkpoint_multihead_config(state)

        self.assertEqual(cfg['enhanced_green_head'], 'expD')
        self.assertEqual(cfg['adapter_families'], ['green8'])
        self.assertEqual(cfg['pos0_target_families'], [])
        self.assertTrue(cfg['has_shared_pos0'])
        self.assertTrue(cfg['has_any_pos0'])
        self.assertEqual(cfg['pos0_head_cols'], 4)
        self.assertEqual(cfg['pos0_num_classes'], 31)

    def test_infers_expe_and_multiple_family_specific_pos0_heads(self):
        state = {
            'containers.green8.0.weight': torch.zeros(512, 448 + 66, 3, 3),
            'pos0_family_heads.green8.4.weight': torch.zeros(34, 128),
            'pos0_family_heads.special.4.weight': torch.zeros(34, 128),
            'family_adapters.green8.0.weight': torch.zeros(128, 448 + 66, 3, 3),
            'family_adapters.special.0.weight': torch.zeros(128, 448 + 66, 3, 3),
        }

        cfg = infer_checkpoint_multihead_config(state)

        self.assertEqual(cfg['enhanced_green_head'], 'expE')
        self.assertEqual(cfg['adapter_families'], ['green8', 'special'])
        self.assertEqual(cfg['pos0_target_families'], ['green8', 'special'])
        self.assertFalse(cfg['has_shared_pos0'])
        self.assertTrue(cfg['has_any_pos0'])
        self.assertEqual(cfg['pos0_num_classes'], 34)

    def test_builds_network_with_detected_family_specific_pos0_heads(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
            'pos0_family_heads.green8.4.weight': torch.zeros(31, 128),
            'family_adapters.green8.0.weight': torch.zeros(128, 448 + 66, 3, 3),
        }

        net, cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=66,
            dropout_rate=0.0,
        )

        self.assertEqual(cfg['enhanced_green_head'], 'expD')
        self.assertIn('green8', net.family_adapters)
        self.assertIn('green8', net.pos0_family_heads)
        self.assertEqual(net.pos0_head_cols, 4)
        self.assertIsNotNone(net.pos0_head)

    def test_disables_pos0_when_checkpoint_has_no_pos0_weights(self):
        state = {
            'containers.green8.0.weight': torch.zeros(256, 448 + 66, 3, 3),
        }

        net, cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=66,
            dropout_rate=0.0,
        )

        self.assertFalse(cfg['has_any_pos0'])
        self.assertEqual(cfg['pos0_num_classes'], DEFAULT_POS0_NUM_CLASSES)
        self.assertEqual(net.pos0_head_cols, 0)
        self.assertEqual(len(net.pos0_family_heads), 0)


if __name__ == '__main__':
    unittest.main()
