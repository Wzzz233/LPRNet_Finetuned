import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path('/home/wzzz/LPRNet/src/utils/generate_green_e15a_prewarp_slot_probe.py')


def load_module():
    spec = importlib.util.spec_from_file_location('generate_green_e15a_prewarp_slot_probe', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestE15APrewarpSlotProbe(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.char_quads = []
        x = 6
        for _ in range(8):
            self.char_quads.append(np.asarray([
                [x, 6],
                [x + 7, 6],
                [x + 7, 17],
                [x, 17],
            ], dtype=np.float32))
            x += 10
        self.char_quads = np.asarray(self.char_quads, dtype=np.float32)

        img = np.zeros((24, 94, 3), dtype=np.uint8)
        for idx, quad in enumerate(self.char_quads):
            x1 = int(quad[:, 0].min())
            y1 = int(quad[:, 1].min())
            x2 = int(quad[:, 0].max())
            y2 = int(quad[:, 1].max())
            img[y1:y2 + 1, x1:x2 + 1] = 20 + idx * 20
        self.img = img

    def test_slot_bbox_targets_positions_3_to_5(self):
        bbox = self.mod.slot_bbox_from_char_quads(self.char_quads, pad=0, image_shape=self.img.shape)
        self.assertEqual(bbox, (26, 6, 53, 17))

    def test_apply_prewarp_none_is_identity(self):
        out_img, out_quads, meta = self.mod.apply_prewarp_slot_mode(self.img, self.char_quads, 'none', np.random.default_rng(123))
        self.assertTrue(np.array_equal(out_img, self.img))
        self.assertTrue(np.array_equal(out_quads, self.char_quads))
        self.assertEqual(meta['mode'], 'none')

    def test_apply_prewarp_changes_only_aa0_region(self):
        bbox = self.mod.slot_bbox_from_char_quads(self.char_quads, pad=1, image_shape=self.img.shape)
        out_img, out_quads, meta = self.mod.apply_prewarp_slot_mode(
            self.img,
            self.char_quads,
            'aa0_prewarp_right_pull',
            np.random.default_rng(7),
        )
        self.assertEqual(meta['mode'], 'aa0_prewarp_right_pull')
        x1, y1, x2, y2 = bbox

        mask = np.ones(self.img.shape[:2], dtype=bool)
        mask[y1:y2 + 1, x1:x2 + 1] = False
        self.assertTrue(np.array_equal(out_img[mask], self.img[mask]))
        self.assertFalse(np.array_equal(out_img[y1:y2 + 1, x1:x2 + 1], self.img[y1:y2 + 1, x1:x2 + 1]))

        self.assertTrue(np.array_equal(out_quads[:2], self.char_quads[:2]))
        self.assertTrue(np.array_equal(out_quads[5:], self.char_quads[5:]))
        self.assertFalse(np.array_equal(out_quads[2:5], self.char_quads[2:5]))


if __name__ == '__main__':
    unittest.main()
