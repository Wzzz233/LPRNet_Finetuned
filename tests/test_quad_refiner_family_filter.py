import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SCRIPT = ROOT / 'scripts' / 'build_quad_refiner_dataset.py'
spec = importlib.util.spec_from_file_location('build_quad_refiner_dataset', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class QuadRefinerFamilyFilterTests(unittest.TestCase):
    def test_load_family_whitelist(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'families.txt'
            p.write_text('normal7\n\ngreen8\n', encoding='utf-8')
            self.assertEqual(mod.load_family_whitelist(p), {'normal7', 'green8'})

    def test_maybe_filter_family_keeps_allowed(self):
        rec = {'family': 'green8', 'sample_id': 'x'}
        self.assertEqual(mod.maybe_filter_family(rec, {'green8'}), rec)

    def test_maybe_filter_family_drops_disallowed(self):
        rec = {'family': 'special', 'sample_id': 'x'}
        self.assertIsNone(mod.maybe_filter_family(rec, {'normal7', 'green8'}))


if __name__ == '__main__':
    unittest.main()
