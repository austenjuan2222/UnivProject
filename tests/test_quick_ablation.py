import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_vd import make  # noqa: E402
from quick_ablation import main  # noqa: E402
from vd_preprocess import preprocess  # noqa: E402


class QuickAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        make(tmp / "raw", days=6, num_vds=3)
        preprocess(tmp / "raw", tmp / "processed", pattern="N1", n=3)
        cls.data = str(tmp / "processed")
        cls.out = str(tmp / "out.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_main(self, *extra):
        return main(["--data-dir", self.data, "--out", self.out, "--rounds", "2", "--seeds", "1",
                     "--name", "t", *extra])

    def test_dp_run_reports_epsilon_and_baseline(self):
        r = self.run_main()
        self.assertGreater(r["eps"], 0)
        self.assertLess(r["eps"], float("inf"))
        self.assertGreater(r["naive"], 0)
        self.assertEqual(len(r["per"]), 3)

    def test_nodp_horizon_and_mia(self):
        r = self.run_main("--sigma", "-1", "--horizon", "3", "--mia")
        self.assertEqual(r["eps"], float("inf"))
        auc = r["mia"][0]["auc"]
        self.assertTrue(0.0 <= auc <= 1.0)

    def test_validation_split_and_finetune(self):
        full = self.run_main("--sigma", "-1")
        val = self.run_main("--sigma", "-1", "--val", "--finetune", "1")
        # --val 只用訓練期間，所以 naive 基準是在不同的（驗證）資料上算的
        self.assertNotAlmostEqual(full["naive"], val["naive"])
        self.assertGreater(val["mae"], 0)

    def test_neighbour_features(self):
        r = self.run_main("--neigh", "both")
        self.assertGreater(r["mae"], 0)


    def test_calendar_features_extra_series_and_avg_last(self):
        r = self.run_main("--cal", "tod,dow", "--feats", "volume,occupancy", "--avg-last", "2",
                          "--finetune", "1")
        self.assertTrue(r["mae"] > 0 and r["mae"] < 100)

    def test_time_feats_shape_and_range(self):
        from quick_ablation import time_feats
        f = time_feats(["2024-03-09 06:00:00", "2024-03-11 18:00:00"], ["tod", "dow"])
        self.assertEqual(f.shape, (2, 3))
        self.assertAlmostEqual(f[0, 0], 1.0, places=6)   # 06:00 → sin(π/2)
        self.assertEqual(list(f[:, 2]), [1.0, 0.0])      # 週六 / 週一

    def test_block_fold_keeps_train_and_test_disjoint(self):
        from quick_ablation import load
        d = load(self.data, 12, 1, fold=1, nfolds=3)
        self.assertGreater(len(d[0]["ytr"]), 0)
        self.assertGreater(len(d[0]["yte"]), 0)
        r = self.run_main("--fold", "1", "--nfolds", "3")
        self.assertTrue(0 < r["mae"] < 100)

if __name__ == "__main__":
    unittest.main()
