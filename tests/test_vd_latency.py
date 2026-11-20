"""即時 VDLive 延遲量測：從一份 XML 和抓取時間算出資料落後多久。"""
import datetime as dt
import unittest

from scripts.measure_vd_latency import latency

XML = """<?xml version='1.0' encoding='utf-8'?>
<VDLiveList xmlns="http://traffic.transportdata.tw/standard/traffic/schema/">
  <UpdateTime>2026-11-20T22:30:00+08:00</UpdateTime>
  <VDLives>
    <VDLive><VDID>VD-A</VDID><DataCollectTime>2026-11-20T22:29:00+08:00</DataCollectTime><Status>0</Status></VDLive>
    <VDLive><VDID>VD-B</VDID><DataCollectTime>2026-11-20T22:27:00+08:00</DataCollectTime><Status>0</Status></VDLive>
    <VDLive><VDID>VD-C</VDID><DataCollectTime>2026-11-20T22:29:00+08:00</DataCollectTime><Status>0</Status></VDLive>
  </VDLives>
</VDLiveList>""".encode()

TZ = dt.timezone(dt.timedelta(hours=8))


class LatencyTest(unittest.TestCase):
    def test_lags_in_seconds(self):
        now = dt.datetime(2026, 11, 20, 22, 32, 15, tzinfo=TZ)
        r = latency(XML, now, keep={"VD-A", "VD-B"})
        self.assertEqual(r["update_lag_s"], 135)          # 22:32:15 − 22:30:00
        self.assertEqual(r["n_vd"], 2)                    # 只算指定的 VD
        self.assertEqual(r["collect_lag_s"], [195, 315])  # VD-A 落後 3 分 15 秒、VD-B 5 分 15 秒
        # 5 分鐘一格、以起點標記：22:25 那格要等 22:29 那筆才完整，結束在 22:30，抓取時已經過了 135 秒
        self.assertEqual(r["bin5_lag_s"], [135, 435])     # VD-B 最新完整的是 22:20 那格（結束 22:25）

    def test_missing_or_bad_time_is_skipped(self):
        bad = XML.replace(b"2026-11-20T22:27:00+08:00", b"")
        now = dt.datetime(2026, 11, 20, 22, 32, 15, tzinfo=TZ)
        r = latency(bad, now)
        self.assertEqual(r["n_vd"], 2)


if __name__ == "__main__":
    unittest.main()
