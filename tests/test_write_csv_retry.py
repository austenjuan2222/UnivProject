"""Windows 上目的檔被別的程式（防毒、索引、讀檔）短暫開著時，os.replace 會丟 PermissionError；要重試而不是讓整個實驗中斷。"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import results_log


class WriteCsvRetryTest(unittest.TestCase):
    def test_retries_when_destination_is_briefly_locked(self):
        real = Path.replace
        calls = []

        def flaky(self, target):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError("[WinError 5] 存取被拒")
            return real(self, target)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(Path, "replace", flaky), \
                mock.patch.object(results_log.time, "sleep"):
            p = Path(d) / "m.csv"
            results_log._write_csv(p, ["a"], [{"a": 1}])
            self.assertEqual(p.read_text(encoding="utf-8").splitlines(), ["a", "1"])
        self.assertEqual(len(calls), 3)

    def test_gives_up_after_persistent_lock(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(Path, "replace", side_effect=PermissionError("locked")), \
                mock.patch.object(results_log.time, "sleep"):
            with self.assertRaises(PermissionError):
                results_log._write_csv(Path(d) / "m.csv", ["a"], [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
