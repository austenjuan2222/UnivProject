import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from seed_stats import pair_across, pair_within, paired_summary  # noqa: E402


class SeedStatsTest(unittest.TestCase):
    def test_within_pairs_by_seed(self):
        d = {"nodp_seed1": 3.0, "nodp_seed2": 3.2, "dp_sigma1.1_seed1": 2.9, "dp_sigma1.1_seed2": 3.0,
             "dp_sigma1.1_seed3": 9.9}
        self.assertEqual(pair_within(d, "nodp", "dp_sigma1.1"), [("seed1", 3.0, 2.9), ("seed2", 3.2, 3.0)])

    def test_across_pairs_same_names(self):
        self.assertEqual(pair_across({"x": 1.0, "y": 2.0}, {"y": 1.5}), [("y", 2.0, 1.5)])

    def test_summary_sign_ci_and_effect(self):
        s = paired_summary([("1", 3.0, 2.9), ("2", 3.1, 2.95), ("3", 3.2, 3.1), ("4", 3.0, 2.92)])
        self.assertLess(s["mean_diff"], 0)
        self.assertLess(s["ci_low"], s["mean_diff"])
        self.assertLess(s["ci_high"], 0)
        self.assertLess(s["dz"], 0)
        self.assertEqual(s["n_b_better"], 4)
        self.assertAlmostEqual(s["p_wilcoxon"], 0.125)  # n=4 精確雙尾最小值

    def test_needs_two_pairs(self):
        with self.assertRaises(ValueError):
            paired_summary([("1", 1.0, 2.0)])


if __name__ == "__main__":
    unittest.main()
