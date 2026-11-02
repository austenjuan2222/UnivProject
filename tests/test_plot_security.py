import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_security import collect  # noqa: E402

FIELDS = ["run", "final_mae_kmh", "finetuned_mae_kmh", "naive_mae_kmh"]


def _write(root, batch, rows):
    d = Path(root) / batch / "20261102-000000"
    d.mkdir(parents=True)
    with (d / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


class CollectTest(unittest.TestCase):
    def test_reads_dp_runs_skips_missing_batches_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, "s32_secure", [
                {"run": "dp_sigma1.1_seed1", "final_mae_kmh": 2.3, "finetuned_mae_kmh": 2.1, "naive_mae_kmh": 2.43},
                {"run": "dp_sigma1.1_seed2", "final_mae_kmh": "inf", "finetuned_mae_kmh": 2.2, "naive_mae_kmh": 2.43},
                {"run": "nodp_seed1", "final_mae_kmh": 9.0, "finetuned_mae_kmh": 9.0, "naive_mae_kmh": 2.43},
            ])
            data = collect(Path(root))
        self.assertEqual([d["label"] for d in data], ["HMAC + model signature on"])
        self.assertEqual(data[0]["n"], 2)
        self.assertEqual(data[0]["g"], [2.3])
        self.assertEqual(data[0]["ft"], [2.1, 2.2])


if __name__ == "__main__":
    unittest.main()
