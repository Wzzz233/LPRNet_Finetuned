import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from quad_refiner import dataset as mod


class QuadRefinerFamilyClassificationTests(unittest.TestCase):
    def test_classify_family_from_text_blue(self):
        self.assertEqual(mod.classify_family_from_text('川A12345'), ('normal7', 'blue'))

    def test_classify_family_from_text_green_small(self):
        self.assertEqual(mod.classify_family_from_text('川AD12345'), ('green8', 'green_small'))

    def test_classify_family_from_text_green_large(self):
        self.assertEqual(mod.classify_family_from_text('川A12345D'), ('green8', 'green_large'))

    def test_build_crpd_records_sets_family(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / 'x.jpg'
            lbl = Path(td) / 'x.txt'
            img.write_bytes(b'fake')
            lbl.write_text('1389 926 1483 925 1483 959 1389 959 0 川FKX755\n', encoding='utf-8')
            rec = mod.build_crpd_records(img, lbl, split='train', source_name='crpd_raw')[0]
            self.assertEqual(rec['family'], 'normal7')
            self.assertEqual(rec['sub_type'], 'blue')


if __name__ == '__main__':
    unittest.main()
