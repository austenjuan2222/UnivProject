import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from results_log import RunLogger, aggregate_eval, aggregate_train, make_run_name


def _rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class AggregateTest(unittest.TestCase):
    def test_train_weighted_loss_and_epsilon_stats(self):
        out = aggregate_train([
            {"num-examples": 100, "train-loss": 1.0, "epsilon": 2.0},
            {"num-examples": 300, "train-loss": 3.0, "epsilon": 4.0},
        ])
        self.assertAlmostEqual(out["train_loss"], 2.5)
        self.assertEqual(out["epsilon_max"], 4.0)
        self.assertEqual(out["epsilon_mean"], 3.0)

    def test_train_pools_local_model_mae_by_test_size(self):
        out = aggregate_train([
            {"num-examples": 100, "train-loss": 1.0, "local-mae-kmh": 2.0, "num-test": 1},
            {"num-examples": 100, "train-loss": 1.0, "local-mae-kmh": 6.0, "num-test": 3},
        ])
        self.assertAlmostEqual(out["local_mae_kmh"], 5.0)
        self.assertNotIn("local_mae_kmh", aggregate_train([{"num-examples": 1, "train-loss": 1.0}]))

    def test_train_without_dp_has_no_epsilon(self):
        out = aggregate_train([{"num-examples": 1, "train-loss": 1.0}])
        self.assertNotIn("epsilon_max", out)

    def test_eval_pools_naive_baseline_when_present(self):
        out = aggregate_eval([
            {"num-examples": 1, "eval-loss": 0.1, "mae-kmh": 2.0, "mse-kmh": 4.0, "naive-mae-kmh": 3.0},
            {"num-examples": 3, "eval-loss": 0.3, "mae-kmh": 6.0, "mse-kmh": 36.0, "naive-mae-kmh": 7.0},
        ])
        self.assertAlmostEqual(out["naive_mae_kmh"], 6.0)
        self.assertNotIn("naive_mae_kmh", aggregate_eval([
            {"num-examples": 1, "eval-loss": 0.1, "mae-kmh": 2.0, "mse-kmh": 4.0}]))

    def test_eval_rmse_is_sqrt_of_pooled_mse(self):
        out = aggregate_eval([
            {"num-examples": 1, "eval-loss": 0.1, "mae-kmh": 2.0, "mse-kmh": 4.0},
            {"num-examples": 3, "eval-loss": 0.3, "mae-kmh": 6.0, "mse-kmh": 36.0},
        ])
        self.assertAlmostEqual(out["mae_kmh"], 5.0)
        self.assertAlmostEqual(out["rmse_kmh"], math.sqrt(28.0))
        self.assertAlmostEqual(out["eval_loss"], 0.25)


class RunLoggerTest(unittest.TestCase):
    def test_writes_round_metrics_clients_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            log = RunLogger(Path(d) / "run1", config={"use-dp": True, "noise-multiplier": 1.1},
                            vdid_map={0: "A", 1: "B"})
            replies = [
                {"round": 1, "partition-id": 0, "num-examples": 10, "train-loss": 1.0, "epsilon": 0.5},
                {"round": 1, "partition-id": 1, "num-examples": 10, "train-loss": 3.0, "epsilon": 0.7},
            ]
            log.on_train(replies)
            log.on_eval([
                {"round": 1, "partition-id": 0, "num-examples": 5, "eval-loss": 0.2, "mae-kmh": 3.0, "mse-kmh": 9.0},
                {"round": 1, "partition-id": 1, "num-examples": 5, "eval-loss": 0.4, "mae-kmh": 5.0, "mse-kmh": 25.0},
            ])
            m = _rows(Path(d, "run1", "metrics.csv"))
            self.assertEqual(len(m), 1)
            self.assertEqual(m[0]["round"], "1")
            self.assertAlmostEqual(float(m[0]["train_loss"]), 2.0)
            self.assertAlmostEqual(float(m[0]["mae_kmh"]), 4.0)
            self.assertAlmostEqual(float(m[0]["epsilon_max"]), 0.7)

            c = _rows(Path(d, "run1", "clients.csv"))
            self.assertEqual(len(c), 2)
            self.assertEqual(c[1]["vdid"], "B")
            self.assertAlmostEqual(float(c[1]["epsilon"]), 0.7)
            self.assertAlmostEqual(float(c[1]["mae_kmh"]), 5.0)

            cfg = json.loads(Path(d, "run1", "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["run_config"]["noise-multiplier"], 1.1)
            self.assertIn("started_at", cfg)

    def test_partial_replies_warn_and_are_counted(self):
        with tempfile.TemporaryDirectory() as d:
            log = RunLogger(Path(d), config={}, expected_clients=3)
            with self.assertWarnsRegex(UserWarning, "2/3"):
                log.on_train([
                    {"round": 1, "partition-id": 0, "num-examples": 1, "train-loss": 1.0,
                     "epsilon": 0.5, "dp-steps": 4, "sample-rate": 0.25},
                    {"round": 1, "partition-id": 1, "num-examples": 1, "train-loss": 1.0,
                     "epsilon": 0.5, "dp-steps": 4, "sample-rate": 0.25},
                ])
            self.assertEqual(_rows(Path(d, "metrics.csv"))[0]["n_train_clients"], "2")
            c = _rows(Path(d, "clients.csv"))[0]
            self.assertEqual((c["dp_steps"], c["sample_rate"]), ("4", "0.25"))

    def test_make_run_name(self):
        self.assertTrue(make_run_name({"use-dp": True, "noise-multiplier": 1.1}).startswith("dp_sigma1.1_"))
        self.assertTrue(make_run_name({"use-dp": False}).startswith("nodp_"))
        self.assertEqual(make_run_name({"use-dp": False, "run-name": "x"}), "x")


if __name__ == "__main__":
    unittest.main()
