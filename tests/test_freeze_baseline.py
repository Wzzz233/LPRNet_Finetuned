import json
import tempfile
import unittest
from pathlib import Path

from freeze_baseline import read_json


class FreezeBaselineTests(unittest.TestCase):
    def test_read_json_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text('{"x": 1}\n', encoding='utf-8')
            data = read_json(str(p))
            self.assertEqual(data["x"], 1)


if __name__ == "__main__":
    unittest.main()
