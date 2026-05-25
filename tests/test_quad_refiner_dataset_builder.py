import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_quad_refiner_dataset.py"

spec = importlib.util.spec_from_file_location("build_quad_refiner_dataset", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class QuadRefinerDatasetBuilderTests(unittest.TestCase):
    def test_maybe_override_keeps_record_without_coarse_when_not_strict(self):
        rec = {"sample_id": "a", "coarse_quad": [[1, 1], [2, 1], [2, 2], [1, 2]]}
        out = mod.maybe_override(dict(rec), coarse_map={}, strict_coarse=False)
        self.assertIsNotNone(out)
        self.assertEqual(out["sample_id"], "a")

    def test_maybe_override_drops_record_without_coarse_when_strict(self):
        rec = {"sample_id": "a", "coarse_quad": [[1, 1], [2, 1], [2, 2], [1, 2]]}
        out = mod.maybe_override(dict(rec), coarse_map={"b": [[0, 0], [1, 0], [1, 1], [0, 1]]}, strict_coarse=True)
        self.assertIsNone(out)

    def test_maybe_override_replaces_coarse_when_match_exists(self):
        rec = {"sample_id": "a", "coarse_quad": [[1, 1], [2, 1], [2, 2], [1, 2]]}
        coarse = [[10, 10], [20, 10], [20, 20], [10, 20]]
        out = mod.maybe_override(dict(rec), coarse_map={"a": coarse}, strict_coarse=True)
        self.assertIsNotNone(out)
        self.assertEqual(out["coarse_quad"], coarse)


if __name__ == "__main__":
    unittest.main()
