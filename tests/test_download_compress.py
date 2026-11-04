"""壓縮下行：伺服器只廣播「新全域模型 − VD 手上那一版」的量化差值（LoRaWAN 下行比上行更受限）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, RecordDict

from guard import GuardedStrategy, arrays_of


class Inner:
    def configure_train(self, rnd, arrays, config, grid):
        return [Message(content=RecordDict({"arrays": arrays}), dst_node_id=1, message_type="train")]

    configure_evaluate = configure_train


def rec(v):
    return ArrayRecord({"w": Array(np.asarray(v, np.float32))})


class DownloadCompressTest(unittest.TestCase):
    def sent(self, g, rnd, v, evaluate=False):
        f = g.configure_evaluate if evaluate else g.configure_train
        return arrays_of(list(f(rnd, rec(v), ConfigRecord({}), None))[0].content["arrays"])["w"]

    def test_first_broadcast_is_exact_then_quantized(self):
        g = GuardedStrategy(Inner(), None, None, download_bits=4)
        rng = np.random.default_rng(0)
        w0 = rng.normal(0, 1, 200)
        np.testing.assert_array_equal(self.sent(g, 1, w0), w0.astype(np.float32))
        w1 = w0 + rng.normal(0, 0.01, 200)
        got = self.sent(g, 2, w1)
        step = np.abs(w1 - w0).max() / 7
        self.assertLessEqual(np.abs(got - w1).max(), step * 1.001)
        self.assertGreater(np.abs(got - w1).max(), 0)  # 真的有量化
        self.assertEqual(g.download_bytes, [800, 100 + 4])

    def test_same_model_is_not_sent_twice(self):
        # 同一輪先評估再下一輪訓練，發的是同一份全域模型：沿用上次廣播的版本，不重新量化
        g = GuardedStrategy(Inner(), None, None, download_bits=4)
        self.sent(g, 1, np.zeros(10))
        a = self.sent(g, 1, np.full(10, 0.3), evaluate=True)
        b = self.sent(g, 2, np.full(10, 0.3))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(g.download_bytes), 2)

    def test_updates_are_screened_against_broadcast_model(self):
        g = GuardedStrategy(Inner(), None, None, download_bits=2)
        self.sent(g, 1, np.zeros(50))
        got = self.sent(g, 2, np.linspace(0, 1, 50))
        np.testing.assert_array_equal(g._global["w"], got)

    def test_32_bits_sends_model_unchanged(self):
        g = GuardedStrategy(Inner(), None, None)
        w = np.linspace(-1, 1, 30)
        self.sent(g, 1, np.zeros(30))
        np.testing.assert_array_equal(self.sent(g, 2, w), w.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
