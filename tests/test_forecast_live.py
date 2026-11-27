import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("forecast_live", Path(__file__).parents[1] / "scripts" / "forecast_live.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)
from task import TrafficModel  # noqa: E402

CFG = {"lookback": 12, "calendar": "tod"}


def naive_model():
    # residual 模型的 MLP 全為 0：輸出 = 最後一筆，等於 naive
    m = TrafficModel(12, 8, True, 2)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    return m.eval()


class ForecastTest(unittest.TestCase):
    def test_recursive_steps_and_target_times(self):
        s = pd.Series(np.linspace(80, 91, 12), index=pd.date_range("2024-03-29 07:00", periods=12, freq="5min"))
        out = fl.forecast(naive_model(), CFG, s, mean=85.0, std=10.0)
        self.assertEqual([t.strftime("%H:%M") for t, _ in out], ["08:00", "08:05", "08:10"])
        for _, kmh in out:
            self.assertAlmostEqual(kmh, 91.0, places=4)  # 還原回 km/h

    def test_missing_input_gives_no_forecast(self):
        s = pd.Series([80.0] * 12, index=pd.date_range("2024-03-29 07:00", periods=12, freq="5min"))
        s.iloc[5] = np.nan
        self.assertEqual(fl.forecast(naive_model(), CFG, s, 85.0, 10.0), [])
        self.assertEqual(fl.forecast(naive_model(), CFG, s.iloc[:6], 85.0, 10.0), [])

    def test_minute_rows_to_complete_five_minute_buckets(self):
        rows = [{"vdid": "V", "time": f"2024-03-29T07:{m:02d}:00+08:00", "speed": 60.0 + m, "volume": 1,
                 "occupancy": 5, "status": 0} for m in range(0, 13)]
        s = fl.five_minute_series(rows, pd.Timestamp("2024-03-29 07:13"))["V"]
        # 07:10 那一桶還沒滿（end 在桶內）不用；07:00、07:05 是 5 分鐘平均
        self.assertEqual([t.strftime("%H:%M") for t in s.index], ["07:00", "07:05"])
        np.testing.assert_allclose(s.to_numpy(), [62.0, 67.0])



class PredictAllTest(unittest.TestCase):
    def test_backtest_attaches_actuals_and_web_serves_json(self):
        import datetime as dt
        import json
        import threading
        import urllib.request
        rows = [{"vdid": "V", "time": f"{(dt.datetime(2024, 3, 29, 6, 0) + dt.timedelta(minutes=m)):%Y-%m-%dT%H:%M}:00+08:00",
                 "speed": 90.0, "volume": 1, "occupancy": 5, "status": 0} for m in range(0, 140)]
        orig = fl.fetch_minutes
        fl.fetch_minutes = lambda end, keep, after=0, fetch=None: [r for r in rows if keep(r["vdid"])]
        try:
            r = fl.predict_all(naive_model(), {**CFG}, {"V": (85.0, 10.0)}, ["V"], dt.datetime(2024, 3, 29, 7, 13))
        finally:
            fl.fetch_minutes = orig
        self.assertTrue(r["backtest"])
        v = r["vds"][0]
        self.assertEqual((v["last_time"], v["last_kmh"]), ("07:05", 90.0))
        self.assertEqual([(q["time"], q["kmh"], q["actual"]) for q in v["preds"]],
                         [("07:10", 90.0, 90.0), ("07:15", 90.0, 90.0), ("07:20", 90.0, 90.0)])

        spec2 = importlib.util.spec_from_file_location("forecast_web", Path(__file__).parents[1] / "scripts" / "forecast_web.py")
        fw = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fw)
        fc = fw.Forecaster.__new__(fw.Forecaster)
        fc.lock, fc.latest = threading.Lock(), r
        srv = fw.ThreadingHTTPServer(("127.0.0.1", 0), fw.make_handler(fc))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            self.assertEqual(json.loads(urllib.request.urlopen(base + "/api/forecast").read())["vds"][0]["vdid"], "V")
            self.assertIn("textContent", urllib.request.urlopen(base + "/").read().decode("utf-8"))  # 不用 innerHTML 塞資料
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(base + "/../etc/passwd")
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
