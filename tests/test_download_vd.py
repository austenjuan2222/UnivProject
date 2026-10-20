import csv
import datetime as dt
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_vd import day_urls, download_day  # noqa: E402
from tests.test_vd_parse import _doc, _lane, _vd  # noqa: E402


class DayUrlsTest(unittest.TestCase):
    def test_one_file_per_minute(self):
        urls = day_urls(dt.date(2024, 3, 4), base="https://h/history/motc20/VD")
        self.assertEqual(len(urls), 1440)
        self.assertEqual(urls[0], "https://h/history/motc20/VD/20240304/VDLive_0000.xml.gz")
        self.assertEqual(urls[-1], "https://h/history/motc20/VD/20240304/VDLive_2359.xml.gz")


class DownloadDayTest(unittest.TestCase):
    def _fake_fetch(self, fail_minutes=()):
        calls = []

        def fetch(url):
            calls.append(url)
            hhmm = url[-11:-7]
            if hhmm in fail_minutes:
                raise OSError("boom")
            t = f"2024-03-04T{hhmm[:2]}:{hhmm[2:]}:00+08:00"
            return gzip.compress(_doc(
                _vd("VD-N1-A", [_lane(0, 90, 2, {"S": 4})], t=t),
                _vd("VD-N3-B", [_lane(0, 70, 2, {"S": 4})], t=t),
            ))
        return fetch, calls

    def test_writes_filtered_rows_and_marks_done(self):
        fetch, calls = self._fake_fetch()
        with tempfile.TemporaryDirectory() as d:
            res = download_day(dt.date(2024, 3, 4), Path(d), fetch=fetch,
                               keep=lambda v: "N1" in v, base="https://h/x", workers=4)
            self.assertEqual(res["failed"], [])
            out = Path(d, "vd_minute_20240304.csv")
            with out.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1440)
            self.assertEqual({r["vdid"] for r in rows}, {"VD-N1-A"})
            self.assertEqual(rows[0]["time"], "2024-03-04T00:00:00+08:00")
            self.assertTrue(Path(d, "vd_minute_20240304.done").exists())

            n_before = len(calls)
            res2 = download_day(dt.date(2024, 3, 4), Path(d), fetch=fetch,
                                keep=None, base="https://h/x")
            self.assertTrue(res2["skipped"])
            self.assertEqual(len(calls), n_before)

    def test_failed_minutes_are_reported_and_day_not_marked_done(self):
        fetch, _ = self._fake_fetch(fail_minutes={"0005", "1200"})
        with tempfile.TemporaryDirectory() as d:
            res = download_day(dt.date(2024, 3, 4), Path(d), fetch=fetch,
                               keep=None, base="https://h/x", retries=1, max_failed=10)
            self.assertEqual(sorted(res["failed"]), ["0005", "1200"])
            self.assertTrue(Path(d, "vd_minute_20240304.csv").exists())
            self.assertTrue(Path(d, "vd_minute_20240304.done").exists())

    def test_too_many_failures_does_not_mark_done(self):
        fetch, _ = self._fake_fetch(fail_minutes={f"00{m:02d}" for m in range(20)})
        with tempfile.TemporaryDirectory() as d:
            res = download_day(dt.date(2024, 3, 4), Path(d), fetch=fetch,
                               keep=None, base="https://h/x", retries=1, max_failed=10)
            self.assertEqual(len(res["failed"]), 20)
            self.assertFalse(Path(d, "vd_minute_20240304.done").exists())


if __name__ == "__main__":
    unittest.main()
