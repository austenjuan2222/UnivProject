import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_results import collect_runs, main  # noqa: E402


def _run(d, name, use_dp, sigma, mae, eps):
    r = Path(d, name)
    r.mkdir()
    (r / "config.json").write_text(json.dumps({"run_config": {"use-dp": use_dp, "noise-multiplier": sigma}}))
    with (r / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_loss", "eval_loss", "mae_kmh", "rmse_kmh", "epsilon_max", "epsilon_mean"])
        for i, (m, e) in enumerate(zip(mae, eps), 1):
            w.writerow([i, 1, 1, m, m + 1, e, e])


class PlotResultsTest(unittest.TestCase):
    def test_collect_orders_baseline_first_then_sigma_desc(self):
        with tempfile.TemporaryDirectory() as d:
            _run(d, "dp_sigma0.8", True, 0.8, [5, 4], [1, 2])
            _run(d, "nodp", False, 1.1, [4, 3], ["", ""])
            _run(d, "dp_sigma1.1", True, 1.1, [6, 5], [0.5, 1])
            runs = collect_runs(Path(d))
            self.assertEqual([r["label"] for r in runs], ["No DP (baseline)", "DP σ=1.1", "DP σ=0.8"])

    def test_writes_png_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            _run(d, "nodp", False, 1.1, [4, 3], ["", ""])
            _run(d, "dp_sigma1.1", True, 1.1, [6, 5], [0.5, 1])
            main([d])
            self.assertTrue(Path(d, "comparison.png").stat().st_size > 1000)
            with Path(d, "summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["final_mae_kmh"], "3.0")
            self.assertEqual(rows[1]["final_epsilon"], "1.0")
            self.assertEqual(rows[0]["final_epsilon"], "")


if __name__ == "__main__":
    unittest.main()
