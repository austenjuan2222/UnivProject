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


if __name__ == "__main__":
    unittest.main()
