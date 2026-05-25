import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src' / 'training']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import train_tiny_province_net as tiny_mod


class TinyProvinceNetGrayTests(unittest.TestCase):
    def test_parser_accepts_gray3_preproc(self):
        parser = tiny_mod.build_parser()
        args = parser.parse_args([
            '--train_manifest', '/tmp/train.csv',
            '--test_manifest', '/tmp/test.csv',
            '--save_dir', '/tmp/out',
            '--ocr_preproc', 'gray3',
        ])
        self.assertEqual(args.ocr_preproc, 'gray3')

    def test_parser_accepts_log_interval(self):
        parser = tiny_mod.build_parser()
        args = parser.parse_args([
            '--train_manifest', '/tmp/train.csv',
            '--test_manifest', '/tmp/test.csv',
            '--save_dir', '/tmp/out',
            '--log_interval', '17',
        ])
        self.assertEqual(args.log_interval, 17)

    def test_make_datasets_passes_ocr_preproc_to_unified_manifest(self):
        class DummyDataset:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            def __len__(self):
                return 0

        parser = tiny_mod.build_parser()
        args = parser.parse_args([
            '--train_manifest', '/tmp/train.csv',
            '--test_manifest', '/tmp/test.csv',
            '--save_dir', '/tmp/out',
            '--ocr_preproc', 'gray3',
        ])

        made = []
        def _factory(*a, **kw):
            ds = DummyDataset(*a, **kw)
            made.append(ds)
            return ds

        with patch.object(tiny_mod, 'UnifiedManifestDataset', side_effect=_factory):
            ds_train, ds_test = tiny_mod.make_datasets(args)

        self.assertEqual(len(made), 2)
        self.assertIs(ds_train, made[0])
        self.assertIs(ds_test, made[1])
        self.assertEqual(made[0].kwargs['ocr_preproc'], 'gray3')
        self.assertEqual(made[1].kwargs['ocr_preproc'], 'gray3')
        self.assertEqual(made[0].kwargs['split_filter'], 'train')
        self.assertEqual(made[1].kwargs['split_filter'], 'test')

    def test_make_datasets_uses_full_crop_height(self):
        class DummyDataset:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            def __len__(self):
                return 0

        parser = tiny_mod.build_parser()
        args = parser.parse_args([
            '--train_manifest', '/tmp/train.csv',
            '--test_manifest', '/tmp/test.csv',
            '--save_dir', '/tmp/out',
            '--input_mode', 'full_crop',
            '--full_crop_height', '64',
            '--ocr_preproc', 'gray3',
        ])

        made = []
        def _factory(*a, **kw):
            ds = DummyDataset(*a, **kw)
            made.append(ds)
            return ds

        with patch.object(tiny_mod, 'FirstCharCropDataset', side_effect=_factory):
            ds_train, ds_test = tiny_mod.make_datasets(args)

        self.assertEqual(len(made), 2)
        self.assertIs(ds_train, made[0])
        self.assertIs(ds_test, made[1])
        self.assertEqual(made[0].kwargs['resize_to'], (171, 64))
        self.assertEqual(made[1].kwargs['resize_to'], (171, 64))

    def test_build_model_uses_single_channel_for_gray_input(self):
        model = tiny_mod.build_model(gray_input=True)
        self.assertEqual(model.features[0].in_channels, 1)
        x = torch.randn(2, 1, 24, 94)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 31))

    def test_format_seconds(self):
        self.assertEqual(tiny_mod._format_seconds(59), '00:59')
        self.assertEqual(tiny_mod._format_seconds(61), '01:01')
        self.assertEqual(tiny_mod._format_seconds(3661), '01:01:01')

    def test_convert_batch_inputs_gray3_matches_gray(self):
        x = torch.rand(2, 3, 24, 94)
        y = tiny_mod.convert_batch_inputs(x, 'gray3')
        self.assertEqual(tuple(y.shape), (2, 1, 24, 94))

    def test_build_model_uses_single_channel_for_gray3_input(self):
        model = tiny_mod.build_model(gray_input=True)
        self.assertEqual(model.features[0].in_channels, 1)

    def test_main_emits_heartbeat_and_summary(self):
        class DummyDataset:
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                image = torch.zeros(3, 24, 94, dtype=torch.float32)
                label = torch.tensor([idx % 31], dtype=torch.long)
                return image, label, 1, 'blue'

        def fake_collate(batch):
            images, labels, lengths, families = zip(*batch)
            return torch.stack(images), torch.cat(labels), list(lengths), list(families)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / 'out'
            outputs = []
            parser = tiny_mod.build_parser()
            args = parser.parse_args([
                '--train_manifest', '/tmp/train.csv',
                '--test_manifest', '/tmp/test.csv',
                '--save_dir', str(save_dir),
                '--epochs', '1',
                '--batch_size', '2',
                '--num_workers', '0',
                '--log_interval', '1',
            ])

            with patch.object(tiny_mod, 'build_parser', return_value=parser), \
                 patch.object(argparse.ArgumentParser, 'parse_args', return_value=args), \
                 patch.object(tiny_mod, 'configure_runtime'), \
                 patch.object(tiny_mod, 'make_datasets', return_value=(DummyDataset(4), DummyDataset(2))), \
                 patch.object(tiny_mod, 'collate_fn', side_effect=fake_collate), \
                 patch('builtins.print', side_effect=lambda *a, **k: outputs.append(a[0]) if a else None):
                tiny_mod.main()

            events = [json.loads(line) for line in outputs if isinstance(line, str) and line.startswith('{')]
            event_names = [e.get('event') for e in events if 'event' in e]
            self.assertIn('train_start', event_names)
            self.assertIn('heartbeat', event_names)
            self.assertIn('new_best', event_names)
            self.assertIn('train_done', event_names)
            summary = json.loads((save_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['log_interval'], 1)
            self.assertIn('total_elapsed_sec', summary)


if __name__ == '__main__':
    unittest.main()
