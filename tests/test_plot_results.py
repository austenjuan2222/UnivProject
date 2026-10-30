import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_results import collect_runs, group_runs, main  # noqa: E402


def _run(d, name, use_dp, sigma, mae, eps, naive=None):
    r = Path(d, name)
    r.mkdir()
    (r / "config.json").write_text(json.dumps({"run_config": {"use-dp": use_dp, "noise-multiplier": sigma}}))
    with (r / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_loss", "eval_loss", "mae_kmh", "rmse_kmh", "epsilon_max",
                    "epsilon_mean", "naive_mae_kmh"])
        for i, (m, e) in enumerate(zip(mae, eps), 1):
            w.writerow([i, 1, 1, m, m + 1, e, e, "" if naive is None else naive])


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
            _run(d, "nodp", False, 1.1, [4, 3], ["", ""], naive=4.5)
            _run(d, "dp_sigma1.1", True, 1.1, [6, 5], [0.5, 1])
            main([d])
            self.assertTrue(Path(d, "comparison.png").stat().st_size > 1000)
            with Path(d, "summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["final_mae_kmh"], "3.0")
            self.assertEqual(rows[1]["final_epsilon"], "1.0")
            self.assertEqual(rows[0]["final_epsilon"], "")
            self.assertEqual(rows[0]["naive_mae_kmh"], "4.5")
            self.assertEqual(rows[1]["naive_mae_kmh"], "")
            # 相對 naive 基準改善多少（%）：(4.5-3)/4.5
            self.assertAlmostEqual(float(rows[0]["improve_vs_naive_pct"]), 100 / 3, places=4)
            self.assertEqual(rows[0]["final_local_mae_kmh"], "")      # 舊 metrics.csv 沒有此欄

    def test_client_level_dp_run_gets_its_own_label(self):
        with tempfile.TemporaryDirectory() as d:
            _run(d, "cldp_sigma2.0", False, 1.1, [4, 3], ["", ""])
            cfg = Path(d, "cldp_sigma2.0", "config.json")
            cfg.write_text(json.dumps({"run_config": {"use-dp": False, "client-dp-noise": 2.0},
                                       "client_level_dp": {"epsilon": 12.3}}))
            r = collect_runs(Path(d))[0]
            self.assertEqual(r["label"], "Client-level DP σ=2")
            self.assertAlmostEqual(r["client_eps"], 12.3)
            _run(d, "nodp", False, 1.1, [4, 3], ["", ""])
            self.assertEqual([x["name"] for x in collect_runs(Path(d))], ["nodp", "cldp_sigma2.0"])
            main([d])
            with Path(d, "summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[1]["final_epsilon"], "12.3")

    def test_seed_runs_are_grouped_with_mean_and_std(self):
        with tempfile.TemporaryDirectory() as d:
            _run(d, "nodp_seed1", False, 1.1, [4, 3], ["", ""])
            _run(d, "nodp_seed2", False, 1.1, [4, 5], ["", ""])
            _run(d, "dp_sigma1.1_seed1", True, 1.1, [6, 5], [0.5, 1])
            groups = group_runs(collect_runs(Path(d)))
            self.assertEqual([g["label"] for g in groups], ["No DP (baseline)", "DP σ=1.1"])
            self.assertEqual(groups[0]["n"], 2)
            self.assertEqual(groups[0]["mae"], [4.0, 4.0])
            main([d])
            with Path(d, "summary_by_config.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["n_seeds"], "2")
            self.assertAlmostEqual(float(rows[0]["final_mae_mean"]), 4.0)
            self.assertAlmostEqual(float(rows[0]["final_mae_std"]), 2 ** 0.5)  # [3, 5] 的樣本標準差
            self.assertEqual(rows[1]["final_mae_std"], "")                 # 只有 1 個 seed


if __name__ == "__main__":
    unittest.main()
