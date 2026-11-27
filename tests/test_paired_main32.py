import importlib.util
import math
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

    def test_holm_adjusts_for_number_of_tests(self):
        # 主表做 30 幾個檢定：Holm 校正後最小的 p 乘上檢定數，之後保持單調、最多 1，NaN 不算進檢定數
        nan = float("nan")
        adj = pm.holm([0.01, 0.04, nan, 0.03, 0.5])
        self.assertAlmostEqual(adj[0], 0.04)
        self.assertAlmostEqual(adj[3], 0.09)
        self.assertAlmostEqual(adj[1], 0.09)  # 0.04×2=0.08 < 前一個 0.09，取單調
        self.assertAlmostEqual(adj[4], 0.5)
        self.assertTrue(math.isnan(adj[2]))

    def test_verdict_text(self):
        self.assertEqual(pm.verdict(0.001, 0.02), "是")
        self.assertEqual(pm.verdict(0.02, 0.3), "否（未校正 p < 0.05）")  # 多重比較校正後不算
        self.assertEqual(pm.verdict(0.2, 1.0), "否")
        self.assertEqual(pm.verdict(float("nan"), float("nan")), "–")


if __name__ == "__main__":
    unittest.main()
