import unittest

from train_LPRNet import get_parser


class TrainManifestModeParserTests(unittest.TestCase):
    def test_parser_accepts_manifest_mode(self):
        import sys
        old = sys.argv
        try:
            sys.argv = [
                "train_LPRNet.py",
                "--data_mode", "manifest",
                "--train_manifest", "/tmp/train.csv",
                "--test_manifest", "/tmp/test.csv",
            ]
            args = get_parser()
            self.assertEqual(args.data_mode, "manifest")
            self.assertEqual(args.train_manifest, "/tmp/train.csv")
            self.assertEqual(args.test_manifest, "/tmp/test.csv")
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
