import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("export_bundle", Path(__file__).parents[1] / "scripts" / "export_bundle.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
from task import TrafficModel  # noqa: E402


class ExportBundleTest(unittest.TestCase):
    def test_bundle_matches_torch_and_live_stats(self):
        with tempfile.TemporaryDirectory() as d:
            run, data = Path(d) / "run", Path(d) / "data"
            run.mkdir(), data.mkdir()
            rc = {"lookback": 12, "hidden-size": 8, "residual": True, "calendar": "tod", "lag-day": False, "horizon": 1, "train-frac": 0.8}
            (run / "config.json").write_text(json.dumps({"run_config": rc}), encoding="utf-8")
            torch.manual_seed(0)
            torch.save(TrafficModel(12, 8, True, 2).state_dict(), run / "final_model.pt")
            vd = "VD-N1-N-10.060-M-RS"
            t = pd.date_range("2024-03-01", periods=500, freq="5min")
            pd.DataFrame({"time": t, "speed": 80 + np.arange(500) % 9}).to_csv(data / "a.csv", index=False)
            (data / "manifest.json").write_text(json.dumps({"clients": [{"vdid": vd, "file": "a.csv"}]}), encoding="utf-8")
            b, model = eb.build(run, data, "abc")
            self.assertLess(eb.self_check(b, model), 1e-5)
            self.assertEqual(b["vds"], [{"vdid": vd, "km": 10.06, "mean": eb.fl.vd_stats(data, 0.8)[vd][0], "std": eb.fl.vd_stats(data, 0.8)[vd][1]}])
            self.assertEqual(b["state_dict"]["net.0.weight"]["shape"], [8, 14])


if __name__ == "__main__":
    unittest.main()
