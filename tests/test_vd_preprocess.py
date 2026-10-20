import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vd_preprocess import (
    aggregate_5min,
    aggregate_files,
    fill_short_gaps,
    preprocess,
    safe_name,
    select_vds,
)


def _minute_rows(vdid, start, n, speed=80.0, volume=10):
    t0 = pd.Timestamp(start)
    return pd.DataFrame({
        "vdid": vdid,
        "time": [(t0 + pd.Timedelta(minutes=i)).isoformat() for i in range(n)],
        "speed": speed,
        "volume": volume,
        "occupancy": 5.0,
        "status": 0,
    })


class Aggregate5MinTest(unittest.TestCase):
    def test_volume_weighted_speed_and_summed_volume(self):
        df = _minute_rows("A", "2024-03-04T08:00:00+08:00", 5)
        df.loc[0, ["speed", "volume"]] = [100.0, 30]
        df.loc[1:4, "speed"] = 50.0
        df.loc[1:4, "volume"] = 10
        out = aggregate_5min(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.index[0], pd.Timestamp("2024-03-04 08:00:00"))
        self.assertAlmostEqual(out["speed"].iloc[0], (100 * 30 + 50 * 40) / 70)
        self.assertEqual(out["volume"].iloc[0], 70)

    def test_time_converted_to_taipei_local(self):
        df = _minute_rows("A", "2024-03-04T00:00:00+00:00", 5)
        out = aggregate_5min(df)
        self.assertEqual(out.index[0], pd.Timestamp("2024-03-04 08:00:00"))

    def test_bucket_with_too_few_valid_minutes_is_nan(self):
        df = _minute_rows("A", "2024-03-04T08:00:00+08:00", 10)
        df.loc[0:2, "speed"] = np.nan  # 第一格只剩 2 分鐘有效
        out = aggregate_5min(df, min_valid_minutes=3)
        self.assertTrue(np.isnan(out["speed"].iloc[0]))
        self.assertAlmostEqual(out["speed"].iloc[1], 80.0)


class AggregateFilesTest(unittest.TestCase):
    def test_file_by_file_equals_all_at_once_even_across_midnight(self):
        rng = np.random.default_rng(0)
        a = _minute_rows("VD-N1-A", "2024-03-04T23:50:00+08:00", 20)
        b = _minute_rows("VD-N1-B", "2024-03-04T23:50:00+08:00", 20)
        df = pd.concat([a, b], ignore_index=True)
        df["speed"] = rng.uniform(40, 100, len(df)).round(1)
        df["volume"] = rng.integers(0, 30, len(df))
        df.loc[[3, 25], "speed"] = np.nan
        with tempfile.TemporaryDirectory() as d:
            # 檔案切在 23:58（像真實資料 VDLive_0000 含前一天 23:58 的紀錄），桶子 23:55 橫跨兩檔
            cut = pd.to_datetime(df["time"], format="ISO8601") < pd.Timestamp("2024-03-04T23:58:00+08:00")
            df[cut].to_csv(Path(d, "vd_minute_20240304.csv"), index=False)
            df[~cut].to_csv(Path(d, "vd_minute_20240305.csv"), index=False)
            per_vd = aggregate_files(sorted(Path(d).glob("vd_minute_*.csv")), pattern="N1")
        for vdid, g in df.groupby("vdid"):
            expected = aggregate_5min(g)
            pd.testing.assert_frame_equal(per_vd[vdid], expected, check_freq=False, check_names=False)

    def test_pattern_filters_vds(self):
        with tempfile.TemporaryDirectory() as d:
            pd.concat([_minute_rows("VD-N1-A", "2024-03-04T08:00:00+08:00", 10),
                       _minute_rows("VD-N3-B", "2024-03-04T08:00:00+08:00", 10)]
                      ).to_csv(Path(d, "vd_minute_20240304.csv"), index=False)
            per_vd = aggregate_files([Path(d, "vd_minute_20240304.csv")], pattern="N1")
        self.assertEqual(list(per_vd), ["VD-N1-A"])


class FillShortGapsTest(unittest.TestCase):
    def test_fills_short_gap_but_not_long_gap(self):
        s = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, 10.0])
        out = fill_short_gaps(s, max_gap=3)
        np.testing.assert_allclose(out[:5], [1, 2, 3, 4, 5])
        self.assertTrue(out[5:9].isna().all())
        self.assertEqual(out.iloc[9], 10.0)

    def test_leading_and_trailing_gaps_stay_nan(self):
        s = pd.Series([np.nan, 2.0, 3.0, np.nan])
        out = fill_short_gaps(s, max_gap=3)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertTrue(np.isnan(out.iloc[3]))


class SelectVdsTest(unittest.TestCase):
    def test_picks_most_complete_matching_vds(self):
        idx = pd.date_range("2024-03-04", periods=10, freq="5min")
        good = pd.Series(1.0, index=idx)
        half = good.copy(); half.iloc[:5] = np.nan
        series = {"VD-N1-a": half, "VD-N1-b": good, "VD-N3-c": good, "VD-N1-d": good}
        chosen = select_vds(series, pattern="N1", n=2, min_valid_ratio=0.4)
        self.assertEqual([c[0] for c in chosen], ["VD-N1-b", "VD-N1-d"])

    def test_raises_when_not_enough_vds(self):
        idx = pd.date_range("2024-03-04", periods=4, freq="5min")
        bad = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx)
        with self.assertRaisesRegex(ValueError, "只有 0 個"):
            select_vds({"VD-N1-a": bad}, pattern="N1", n=1, min_valid_ratio=0.8)


class PreprocessEndToEndTest(unittest.TestCase):
    def test_writes_client_csvs_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d, "raw"), Path(d, "processed")
            raw.mkdir()
            frames = [
                _minute_rows("VD-N1-S-1.0", "2024-03-04T00:00:00+08:00", 60 * 24),
                _minute_rows("VD-N1-S-2.0", "2024-03-04T00:00:00+08:00", 60 * 24, speed=60.0),
                _minute_rows("VD-N3-S-1.0", "2024-03-04T00:00:00+08:00", 60 * 24),
            ]
            pd.concat(frames).to_csv(raw / "vd_minute_20240304.csv", index=False)

            manifest = preprocess(raw, out, pattern="N1", n=2)

            self.assertEqual(len(manifest["clients"]), 2)
            on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, manifest)
            first = on_disk["clients"][0]
            self.assertEqual(first["partition"], 0)
            df = pd.read_csv(out / first["file"], parse_dates=["time"])
            self.assertEqual(list(df.columns), ["time", "speed", "volume", "occupancy"])
            self.assertEqual(len(df), 288)

    def test_safe_name(self):
        self.assertEqual(safe_name("VD-N1-S-1.000-M-LOOP"), "VD-N1-S-1.000-M-LOOP")
        self.assertEqual(safe_name("a/b c"), "a_b_c")


if __name__ == "__main__":
    unittest.main()
