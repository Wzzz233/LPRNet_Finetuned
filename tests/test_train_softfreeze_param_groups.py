import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
TRAINING = SRC / 'training'
for p in (str(SRC), str(TRAINING)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from train_LPRNet import (
    adjust_learning_rate,
    build_optimizer_param_groups,
    count_trainable_params,
    freeze_batchnorm_eval,
    get_parser,
)
from LPRNet_multihead import build_lprnet_multihead


class TrainSoftFreezeParamGroupTests(unittest.TestCase):
    def test_parser_accepts_soft_freeze_lr_multipliers(self):
        old = sys.argv
        try:
            sys.argv = [
                'train_LPRNet.py',
                '--backbone_lr_mult', '0.1',
                '--head_lr_mult', '1.0',
                '--adapter_lr_mult', '0.5',
                '--aux_lr_mult', '2.0',
                '--freeze_bn_stats', 'True',
            ]
            args = get_parser()
            self.assertEqual(args.backbone_lr_mult, 0.1)
            self.assertEqual(args.head_lr_mult, 1.0)
            self.assertEqual(args.adapter_lr_mult, 0.5)
            self.assertEqual(args.aux_lr_mult, 2.0)
            self.assertTrue(args.freeze_bn_stats)
        finally:
            sys.argv = old

    def test_adjust_learning_rate_preserves_group_lr_multipliers(self):
        p_backbone = nn.Parameter(torch.ones(1))
        p_head = nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([
            {'params': [p_backbone], 'lr_mult': 0.1, 'name': 'backbone'},
            {'params': [p_head], 'lr_mult': 1.0, 'name': 'head'},
        ], lr=0.001)

        lr = adjust_learning_rate(opt, cur_epoch=1, base_lr=0.002, lr_schedule=[3, 5])
        self.assertEqual(lr, 0.002)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.0002)
        self.assertAlmostEqual(opt.param_groups[1]['lr'], 0.002)

        lr = adjust_learning_rate(opt, cur_epoch=3, base_lr=0.002, lr_schedule=[3, 5])
        self.assertEqual(lr, 0.0002)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.00002)
        self.assertAlmostEqual(opt.param_groups[1]['lr'], 0.0002)

    def test_build_optimizer_param_groups_classifies_params_without_overlap(self):
        net = build_lprnet_multihead(
            phase=True,
            class_num=66,
            adapter_families=['green8'],
            pos0_head_cols=4,
        )
        net.enable_family_specific_pos0(['green8'], pos0_num_classes=31)
        net.enable_family_specific_province(['green8'], province_num_classes=31)
        aux_second = nn.Linear(66, 66)
        aux_ne = nn.Linear(66, 2)

        groups = build_optimizer_param_groups(
            net,
            aux_second,
            aux_ne,
            second_char_aux_weight=0.3,
            ne_type_aux_weight=0.2,
            base_lr=0.002,
            backbone_lr_mult=0.1,
            head_lr_mult=1.0,
            adapter_lr_mult=0.5,
            aux_lr_mult=2.0,
        )
        by_name = {g['name']: g for g in groups}
        self.assertIn('backbone', by_name)
        self.assertIn('head', by_name)
        self.assertIn('adapter', by_name)
        self.assertIn('aux', by_name)
        self.assertAlmostEqual(by_name['backbone']['lr'], 0.0002)
        self.assertAlmostEqual(by_name['head']['lr'], 0.002)
        self.assertAlmostEqual(by_name['adapter']['lr'], 0.001)
        self.assertAlmostEqual(by_name['aux']['lr'], 0.004)

        ids = []
        for group in groups:
            ids.extend(id(p) for p in group['params'])
        self.assertEqual(len(ids), len(set(ids)))

        expected = count_trainable_params(net)
        expected += count_trainable_params(aux_second)
        expected += count_trainable_params(aux_ne)
        actual = sum(p.numel() for group in groups for p in group['params'])
        self.assertEqual(actual, expected)

    def test_freeze_batchnorm_stats_can_be_applied_without_freezing_conv_weights(self):
        net = build_lprnet_multihead(phase=True, class_num=66)
        first_conv = next(m for m in net.backbone.modules() if isinstance(m, nn.Conv2d))
        self.assertTrue(first_conv.weight.requires_grad)
        freeze_batchnorm_eval(net.backbone, freeze_params=False)
        bn_modules = [m for m in net.backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.assertGreater(len(bn_modules), 0)
        self.assertTrue(all(not m.training for m in bn_modules))
        self.assertTrue(first_conv.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for m in bn_modules for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
