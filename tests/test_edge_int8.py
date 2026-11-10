"""int8 整數推論（講義 Edge 資源：沒有浮點運算單元的微控制器也要能預測車速）。"""
import unittest

import numpy as np

from edge_int8 import footprint, multiplier, predict_int8, quantize, requant
from edge_numpy import predict
from tests.test_edge_numpy import setup


class EdgeInt8Test(unittest.TestCase):
    def test_close_to_float_prediction(self):
        _, p, X, _ = setup(n=500)
        q = quantize(p, X[:250])
        err = np.abs(predict_int8(q, X[250:]) - predict(p, X[250:]))
        self.assertLess(err.mean(), 0.02)  # 標準化尺度；車速標準差約 10 km/h，等於 0.2 km/h

    def test_requant_is_integer_only_and_rounds(self):
        # 乘一個 int32 倍數再右移：C 裡就是 (int64)acc * m >> s，不用浮點
        for scale in (0.0037, 0.25, 0.9):
            m, s = multiplier(scale)
            self.assertIsInstance(m, int)
            acc = np.array([-100_000, -3, 0, 7, 123_456], dtype=np.int32)
            np.testing.assert_allclose(requant(acc, m, s), np.round(acc * scale), atol=1)

    def test_hidden_activations_stay_int8(self):
        _, p, X, _ = setup(n=300)
        q = quantize(p, X)
        h = predict_int8(q, X * 5, return_hidden=True)  # 超出校正範圍也要飽和，不溢位
        for a in h:
            self.assertEqual(a.dtype, np.int8)

    def test_footprint_is_a_quarter_of_float(self):
        _, p, X, _ = setup()
        q = quantize(p, X)
        n_w = sum(v.size for k, v in p.items() if k.endswith("weight"))
        n_b = sum(v.size for k, v in p.items() if k.endswith("bias"))
        self.assertEqual(footprint(q)["weights"], n_w + 4 * n_b)
        self.assertLess(footprint(q)["weights"], 4 * (n_w + n_b) / 3)


if __name__ == "__main__":
    unittest.main()
