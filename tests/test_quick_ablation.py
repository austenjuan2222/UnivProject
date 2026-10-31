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

    def test_neighbour_features(self):
        r = self.run_main("--neigh", "both")
        self.assertGreater(r["mae"], 0)


if __name__ == "__main__":
    unittest.main()
