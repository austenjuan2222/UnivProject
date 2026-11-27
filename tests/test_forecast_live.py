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
        fl.fetch_minutes = lambda end, keep, minutes=0, after=0, fetch=None: [r for r in rows if keep(r["vdid"])]
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


def _rows(start, minutes, speed=90.0, skip=()):
    import datetime as dt
    t0 = dt.datetime.strptime(start, "%Y-%m-%d %H:%M")
    out = []
    for m in range(minutes):
        t = t0 + dt.timedelta(minutes=m)
        if f"{t:%H:%M}" in skip:
            continue
        sp = speed(t) if callable(speed) else speed
        out.append({"vdid": "V", "time": f"{t:%Y-%m-%dT%H:%M}:00+08:00", "speed": sp, "volume": 1, "occupancy": 5, "status": 0})
    return out


def _predict(rows, end):
    import datetime as dt
    orig = fl.fetch_minutes
    fl.fetch_minutes = lambda end, keep, minutes=0, after=0, fetch=None: [r for r in rows if keep(r["vdid"])]
    try:
        return fl.predict_all(naive_model(), {**CFG}, {"V": (85.0, 10.0)}, ["V"], dt.datetime.strptime(end, "%Y-%m-%d %H:%M"))
    finally:
        fl.fetch_minutes = orig


class Round11Test(unittest.TestCase):
    def test_backtest_future_rows_do_not_change_forecast_inputs(self):
        # 07:05 那一桶整個缺；即時版本在 07:13 時沒有 07:10 之後的資料，不能用未來的值把它補起來
        gap = {f"07:0{m}" for m in range(5, 10)}
        a = _predict(_rows("2024-03-29 06:00", 140, 90.0, gap), "2024-03-29 07:13")
        b = _predict(_rows("2024-03-29 06:00", 140, lambda t: 50.0 if t.hour == 7 and t.minute >= 13 else 90.0, gap),
                     "2024-03-29 07:13")
        self.assertEqual(a["vds"][0]["preds"], [])  # 輸入最後一桶缺值：不預測
        self.assertEqual([q["kmh"] for q in a["vds"][0]["preds"]], [q["kmh"] for q in b["vds"][0]["preds"]])

    def test_stale_tail_is_not_reported_as_fresh_forecast(self):
        r = _predict(_rows("2024-03-29 06:00", 46), "2024-03-29 07:13")  # 資料停在 06:45
        self.assertTrue(r["stale"])
        self.assertEqual(r["data_until"], "06:45")
        self.assertEqual(r["vds"][0]["preds"], [])

    def test_source_lag_is_not_stale_and_partial_bucket_is_dropped(self):
        r = _predict(_rows("2024-03-29 06:00", 68), "2024-03-29 07:13")  # 來源晚約 6 分鐘：最後一分鐘 07:07
        self.assertFalse(r["stale"])
        v = r["vds"][0]
        self.assertEqual(v["last_time"], "07:00")  # 07:05 那一桶只有 3 分鐘，還沒收齊
        self.assertEqual(r["data_until"], "07:05")

    def test_forecast_labels_match_target_timestamps(self):
        r = _predict(_rows("2024-03-29 23:00", 74), "2024-03-30 00:13")  # 跨午夜
        self.assertEqual([(q["at"], q["ahead_min"]) for q in r["vds"][0]["preds"]],
                         [("2024-03-30 00:10", -3), ("2024-03-30 00:15", 2), ("2024-03-30 00:20", 7)])


class Round12Test(unittest.TestCase):
    def test_live_rejects_non_unit_horizon(self):
        import json
        import tempfile
        for h in (3, 6):
            with tempfile.TemporaryDirectory() as d:
                rc = {"lookback": 12, "hidden-size": 8, "residual": True, "calendar": "tod", "lag-day": False, "horizon": h}
                (Path(d) / "config.json").write_text(json.dumps({"run_config": rc}), encoding="utf-8")
                torch.save(TrafficModel(12, 8, True, 2).state_dict(), Path(d) / "final_model.pt")
                with self.assertRaises(SystemExit):
                    fl.load_model(Path(d))

    def test_vd_stats_matches_training_with_missing_timestamps(self):
        import json
        import tempfile
        from data import load_client_arrays
        t = pd.date_range("2024-03-01", periods=288 * 5, freq="5min")
        speed = np.where(t < pd.Timestamp("2024-03-03"), 60.0, 100.0) + np.arange(len(t)) % 7
        df = pd.DataFrame({"time": t, "speed": speed})
        df = df[(df.time < "2024-03-02") | (df.time >= "2024-03-03")]  # 中間缺一天
        with tempfile.TemporaryDirectory() as d:
            df.to_csv(Path(d) / "V.csv", index=False)
            (Path(d) / "manifest.json").write_text(json.dumps({"clients": [{"vdid": "V", "file": "V.csv"}]}), encoding="utf-8")
            tr = load_client_arrays(d, 0, 1, 12, 0.8, 1, "", False)
            self.assertEqual(fl.vd_stats(Path(d), 0.8)["V"], (tr["mean"], tr["std"]))


class Round13Test(unittest.TestCase):
    def test_predict_all_fetch_window_scales_with_lookback(self):
        import datetime as dt
        m = TrafficModel(24, 8, True, 2)
        for p in m.parameters():
            torch.nn.init.zeros_(p)
        rows, asked = _rows("2024-03-29 04:00", 200), []

        def fake(end, keep, minutes=0, after=0, fetch=None):
            asked.append(minutes)
            t0 = pd.Timestamp(end) - pd.Timedelta(minutes=minutes)
            return [r for r in rows if keep(r["vdid"]) and pd.Timestamp(r["time"][:16]) >= t0]
        orig, fl.fetch_minutes = fl.fetch_minutes, fake
        try:
            r = fl.predict_all(m.eval(), {"lookback": 24, "calendar": "tod"}, {"V": (85.0, 10.0)}, ["V"], dt.datetime(2024, 3, 29, 7, 13))
        finally:
            fl.fetch_minutes = orig
        self.assertGreaterEqual(asked[0], 24 * 5 + 10)
        self.assertEqual(len(r["vds"][0]["preds"]), 3)


def _web():
    spec2 = importlib.util.spec_from_file_location("forecast_web", Path(__file__).parents[1] / "scripts" / "forecast_web.py")
    fw = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fw)
    return fw


class ConnectionLimitTest(unittest.TestCase):
    def _serve(self, fw, **kw):
        import threading
        fc = fw.Forecaster.__new__(fw.Forecaster)
        fc.lock, fc.latest = threading.Lock(), {"vds": []}
        srv = fw.make_server("127.0.0.1", 0, fc, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def test_incomplete_headers_connection_times_out(self):
        import socket
        import time
        port = self._serve(_web(), timeout=0.5)
        c = socket.create_connection(("127.0.0.1", port))
        c.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n")  # 標頭沒送完
        c.settimeout(5)
        t0 = time.monotonic()
        self.assertEqual(c.recv(100), b"")  # 伺服器在逾時後關閉，不會一直佔著執行緒
        self.assertLess(time.monotonic() - t0, 3)
        c.close()

    def test_concurrent_connections_are_bounded(self):
        import socket
        import urllib.request
        port = self._serve(_web(), timeout=2, max_conn=2)
        idle = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]
        extra = socket.create_connection(("127.0.0.1", port))
        extra.settimeout(1)
        self.assertEqual(extra.recv(100), b"")  # 超過上限的連線直接關掉，沒有開新執行緒
        extra.close()
        for c in idle:
            c.close()
        import time
        time.sleep(0.3)
        self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/forecast", timeout=5).status, 200)


class RateLimitTest(unittest.TestCase):
    def test_rate_limit_per_client_and_window(self):
        spec2 = importlib.util.spec_from_file_location("forecast_web", Path(__file__).parents[1] / "scripts" / "forecast_web.py")
        fw = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fw)
        rl = fw.RateLimit(limit=3, window=60)
        self.assertEqual([rl.allow("a", now=0) for _ in range(4)], [True, True, True, False])
        self.assertTrue(rl.allow("b", now=1))  # 別的來源不受影響
        self.assertTrue(rl.allow("a", now=61))  # 下一個時間窗重新計算


if __name__ == "__main__":
    unittest.main()
