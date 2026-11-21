"""int8 模型輸出成 C：檔案裡的數字要和 Python 版一樣；有 C 編譯器時編起來跑，結果要逐位元相同。"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from edge_int8 import _q, predict_int8, quantize
from edge_int8_c import to_c
from tests.test_edge_numpy import setup


def _compiler():
    """CC 環境變數、系統上的 gcc／clang／cc，或 pip 的 ziglang（python -m ziglang cc）。"""
    if os.environ.get("CC"):
        return os.environ["CC"].split()
    for c in ("gcc", "clang", "cc"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "cc"]
    except ImportError:
        return None


class EdgeInt8CTest(unittest.TestCase):
    def setUp(self):
        _, p, X, _ = setup(n=300)
        self.q, self.X = quantize(p, X[:150]), X[150:]

    def test_arrays_in_c_match_quantized_model(self):
        src = to_c(self.q)
        self.assertNotIn("float", src.split("*/")[-1])  # 註解之外沒有浮點
        for i, L in enumerate(self.q["layers"]):
            for key in ("w", "b", "m", "s"):
                m = re.search(rf"{key.upper()}{i}\[(\d+)\] = \{{([^}}]*)\}}", src)
                got = np.array([int(v) for v in m.group(2).split(",")])
                self.assertEqual(int(m.group(1)), L[key].size)
                np.testing.assert_array_equal(got, L[key].ravel())

    def test_rejects_shifts_c_cannot_do(self):
        # 位移 <= 0 或 > 62 在 C 裡是未定義行為，倍數超出 int32 會被截斷：輸出前就要擋
        L = self.q["layers"][0]
        for key, bad in (("s", 0), ("s", 63), ("m", 2 ** 31)):
            old = L[key].copy()
            L[key] = L[key].astype(np.int64)
            L[key][0] = bad
            with self.assertRaises(ValueError):
                to_c(self.q)
            L[key] = old

    def test_compiled_c_matches_python_bit_for_bit(self):
        cc = _compiler()
        if cc is None:
            self.skipTest("這台電腦沒有 C 編譯器")
        with tempfile.TemporaryDirectory() as d:
            c, exe = os.path.join(d, "vd.c"), os.path.join(d, "vd.exe")
            open(c, "w", encoding="utf-8").write(to_c(self.q))
            subprocess.run(cc + ["-O2", "-DVD_MAIN", "-o", exe, c], check=True, capture_output=True)
            xi = _q(self.X, self.q["s_x"]).astype(np.int8)
            out = subprocess.run([exe], input="\n".join(" ".join(map(str, r)) for r in xi) + "\n",
                                 capture_output=True, text=True, check=True).stdout.split()
        got = np.array([int(v) for v in out]) * (self.q["s_x"] / 256)
        np.testing.assert_array_equal(got, predict_int8(self.q, self.X))


if __name__ == "__main__":
    unittest.main()
