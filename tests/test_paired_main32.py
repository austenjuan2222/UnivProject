import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("paired_main32", Path(__file__).parents[1] / "scripts" / "paired_main32.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


class PairedTest(unittest.TestCase):
    def test_same_sign_and_larger_than_spread_is_significant(self):
        d, spread, sig = pm.paired({"s1": 2.3, "s2": 2.31, "s3": 2.32}, {"s1": 2.2, "s2": 2.2, "s3": 2.21})
        self.assertAlmostEqual(spread, 0.02)
        self.assertTrue(sig)

    def test_one_seed_smaller_than_spread_is_not_significant(self):
        _, _, sig = pm.paired({"s1": 2.21, "s2": 2.30, "s3": 2.30}, {"s1": 2.2, "s2": 2.2, "s3": 2.2})
        self.assertFalse(sig)

    def test_mixed_sign_is_not_significant(self):
        _, _, sig = pm.paired({"s1": 2.5, "s2": 1.9, "s3": 2.5}, {"s1": 2.2, "s2": 2.2, "s3": 2.2})
        self.assertFalse(sig)


if __name__ == "__main__":
    unittest.main()
