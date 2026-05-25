import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SCRIPT = ROOT / 'src' / 'utils' / 'prepare_ccpd_splits.py'
spec = importlib.util.spec_from_file_location('prepare_ccpd_splits', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PrepareCCPDSplitsTests(unittest.TestCase):
    def test_decode_ccpd_plate(self):
        rel = 'ccpd_base/025-95_113-154&383_386&473-386&473_154&465_154&383_386&391-0_0_22_27_27_33_16-68-53.jpg'
        self.assertEqual(mod.decode_ccpd_plate(rel), '皖AY339S')

    def test_generate_split_writes_label_lines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'CCPD2019'
            img_rel = 'ccpd_base/025-95_113-154&383_386&473-386&473_154&465_154&383_386&391-0_0_22_27_27_33_16-68-53.jpg'
            img_path = root / img_rel
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(b'fake')
            split_file = root / 'splits' / 'train.txt'
            split_file.parent.mkdir(parents=True, exist_ok=True)
            split_file.write_text(img_rel + '\n', encoding='utf-8')
            output_file = root / 'prepared_labels' / 'train_labels.txt'
            valid, skipped = mod.generate_split(root, split_file, output_file)
            self.assertEqual(valid, 1)
            self.assertEqual(skipped, 0)
            text = output_file.read_text(encoding='utf-8').strip()
            self.assertEqual(text, img_rel + ' 皖AY339S')


if __name__ == '__main__':
    unittest.main()
