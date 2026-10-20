import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import build_runs, check_smoke_outputs  # noqa: E402


class BuildRunsTest(unittest.TestCase):
    def test_baseline_then_each_sigma_with_shared_settings(self):
        runs = build_runs(n_clients=8, rounds=20, data_dir="C:/d", batch_dir="C:/r/b",
                          sigmas=["1.1", "0.8"], skip_nodp=False, extra="local-epochs=2")
        self.assertEqual([r[0] for r in runs], ["nodp", "dp_sigma1.1", "dp_sigma0.8"])
        for _, cfg in runs:
            self.assertIn("num-clients=8", cfg)
            self.assertIn("num-server-rounds=20", cfg)
            self.assertIn("data-dir='C:/d'", cfg)
            self.assertIn("local-epochs=2", cfg)
        self.assertIn("use-dp=false", runs[0][1])
        self.assertIn("use-dp=true noise-multiplier=0.8", runs[2][1])

    def test_windows_backslashes_become_forward_slashes(self):
        runs = build_runs(3, 2, r"C:\Users\x\data", r"C:\Users\x\res", ["1.1"], True, "")
        self.assertEqual(len(runs), 1)
        self.assertIn("data-dir='C:/Users/x/data'", runs[0][1])


class SmokeCheckTest(unittest.TestCase):
    def _run(self, b, name, eps, n_clients=3):
        d = Path(b, name)
        d.mkdir(parents=True)
        rows = ["round,n_train_clients,n_eval_clients,train_loss,eval_loss,mae_kmh,rmse_kmh,epsilon_max,epsilon_mean"]
        for i, e in enumerate(eps, 1):
            rows.append(f"{i},{n_clients},{n_clients},1,1,5,6,{e},{e}")
        (d / "metrics.csv").write_text("\n".join(rows) + "\n")
        crow = ["round,partition,vdid"] + [f"{r},{p},V{p}" for r in range(1, len(eps) + 1) for p in range(n_clients)]
        (d / "clients.csv").write_text("\n".join(crow) + "\n")
        (d / "final_model.pt").write_bytes(b"x")

    def test_passes_on_good_outputs(self):
        with tempfile.TemporaryDirectory() as b:
            self._run(b, "nodp", ["", ""])
            self._run(b, "dp_sigma1.1", [0.5, 0.8])
            Path(b, "comparison.png").write_bytes(b"x")
            Path(b, "summary.csv").write_text("x")
            eps = check_smoke_outputs(Path(b), rounds=2, n_clients=3, sigma="1.1")
            self.assertEqual(eps, [0.5, 0.8])

    def test_fails_when_epsilon_not_increasing(self):
        with tempfile.TemporaryDirectory() as b:
            self._run(b, "nodp", ["", ""])
            self._run(b, "dp_sigma1.1", [0.8, 0.8])
            Path(b, "comparison.png").write_bytes(b"x")
            Path(b, "summary.csv").write_text("x")
            with self.assertRaisesRegex(AssertionError, "遞增"):
                check_smoke_outputs(Path(b), rounds=2, n_clients=3, sigma="1.1")


if __name__ == "__main__":
    unittest.main()
