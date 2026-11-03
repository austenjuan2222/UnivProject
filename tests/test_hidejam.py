"""針對性攻擊「隱藏壅塞」：被入侵 VD 把塞車時段的標籤改高，以及偵測它的壅塞時段指標。"""
import unittest

import numpy as np

from results_log import aggregate_eval
from security import hide_congestion, poison_update


class HideJamTest(unittest.TestCase):
    def test_only_congested_labels_are_raised(self):
        mean, std = 80.0, 20.0
        y = (np.array([40.0, 59.0, 60.0, 100.0], dtype=np.float32) - mean) / std
        out = hide_congestion(y, mean, std)
        np.testing.assert_allclose(out * std + mean, [70.0, 89.0, 60.0, 100.0], atol=1e-4)
        self.assertEqual(out.dtype, np.float32)

    def test_hidejam_update_is_boosted_by_scale(self):
        g, local = {"w": np.zeros(3, np.float32)}, {"w": np.ones(3, np.float32)}
        out = poison_update(g, local, "hidejam", 3.0, np.random.default_rng(0))
        np.testing.assert_allclose(out["w"], 3.0)

    def test_eval_pools_jam_metrics_by_jam_count(self):
        reps = [{"num-examples": 10, "eval-loss": 1, "mae-kmh": 2, "mse-kmh": 4,
                 "num-jam": 1, "jam-mae-kmh": 10.0, "jam-bias-kmh": 10.0},
                {"num-examples": 10, "eval-loss": 1, "mae-kmh": 2, "mse-kmh": 4,
                 "num-jam": 3, "jam-mae-kmh": 2.0, "jam-bias-kmh": -2.0}]
        out = aggregate_eval(reps)
        self.assertAlmostEqual(out["jam_mae_kmh"], 4.0)
        self.assertAlmostEqual(out["jam_bias_kmh"], 1.0)


class JamMetricTest(unittest.TestCase):
    def test_evaluate_reports_error_on_congested_samples(self):
        import torch

        from task import evaluate

        class Const(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x))  # 永遠預測平均速度 80

        mean, std = 80.0, 20.0
        y = ((np.array([40.0, 100.0]) - mean) / std).astype(np.float32)
        X = np.zeros((2, 3), np.float32)
        r = evaluate(Const(), X, y, mean, std)
        self.assertEqual(r["n_jam"], 1)
        self.assertAlmostEqual(r["jam_mae_kmh"], 40.0, places=4)
        self.assertAlmostEqual(r["jam_bias_kmh"], 40.0, places=4)  # 正值：把塞車預測成順暢


if __name__ == "__main__":
    unittest.main()
