import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("paired_main32", Path(__file__).parents[1] / "scripts" / "paired_main32.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


class PairedTest(unittest.TestCase):
    def test_main32_significance_uses_paired_t(self):
        # Codex 第 9 回合：差值 +0.013／+0.008／+0.009 都小於 seed 間差距（舊規則判不顯著），配對 t 檢定 p ≈ 0.023
        a = {"s1": 2.183, "s2": 2.168, "s3": 2.198}
        b = {"s1": 2.170, "s2": 2.160, "s3": 2.189}
        d, s = pm.paired(a, b)
        self.assertEqual([round(x, 3) for x in d], [0.013, 0.008, 0.009])
        self.assertLess(s["p_t"], 0.05)
        self.assertGreater(s["ci_low"], 0)

    def test_mixed_sign_is_not_significant(self):
        _, s = pm.paired({"s1": 2.5, "s2": 1.9, "s3": 2.5}, {"s1": 2.2, "s2": 2.2, "s3": 2.2})
        self.assertGreater(s["p_t"], 0.05)

    def test_verdict_text(self):
        self.assertEqual(pm.verdict({"p_t": 0.02}), "是")
        self.assertEqual(pm.verdict({"p_t": 0.2}), "否")
        self.assertEqual(pm.verdict({"p_t": float("nan")}), "–")


if __name__ == "__main__":
    unittest.main()
