"""compress.py：上傳前把模型更新量化成 8／4 位元（講義：LPWAN 頻寬小、Edge 降低流量）。"""
import unittest

import numpy as np

from compress import quantize_update


def _g():
    return {"w": np.zeros((50, 40), dtype=np.float32), "b": np.zeros(40, dtype=np.float32)}


class QuantizeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.local = {k: v + rng.normal(0, 0.1, v.shape).astype(np.float32) for k, v in _g().items()}

    def test_error_is_bounded_by_one_step(self):
        out, _ = quantize_update(_g(), self.local, 8, np.random.default_rng(1))
        for k in out:
            step = np.abs(self.local[k]).max() / 127
            self.assertLessEqual(np.abs(out[k] - self.local[k]).max(), step * 1.0001)

    def test_stochastic_rounding_is_unbiased(self):
        g = {"w": np.zeros(20000, dtype=np.float32)}
        loc = {"w": np.full(20000, 0.3, dtype=np.float32)}
        loc["w"][0] = 1.0  # 決定量化尺度
        out, _ = quantize_update(g, loc, 2, np.random.default_rng(2))
        self.assertAlmostEqual(float(out["w"][1:].mean()), 0.3, delta=0.01)

    def test_bytes_shrink_with_bits(self):
        _, b8 = quantize_update(_g(), self.local, 8, np.random.default_rng(0))
        _, b4 = quantize_update(_g(), self.local, 4, np.random.default_rng(0))
        self.assertEqual(b8, 2000 + 40 + 4 * 2)      # 每個值 1 byte＋每個張量一個 float32 尺度
        self.assertEqual(b4, 1000 + 20 + 4 * 2)      # 每個值半個 byte

    def test_32_bits_means_no_compression(self):
        out, b = quantize_update(_g(), self.local, 32, np.random.default_rng(0))
        np.testing.assert_array_equal(out["w"], self.local["w"])
        self.assertEqual(b, 4 * (50 * 40 + 40))

    def test_zero_update_stays_zero(self):
        out, _ = quantize_update(_g(), _g(), 8, np.random.default_rng(0))
        self.assertTrue(all(np.all(v == 0) for v in out.values()))


if __name__ == "__main__":
    unittest.main()
