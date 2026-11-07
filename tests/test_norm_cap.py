"""伺服器端更新大小上限：更新長度超過這輪中位數 k 倍的，縮回 k 倍（對應講義被入侵裝置送放大的惡意更新）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from guard import GuardedStrategy, arrays_of, attach_update_tag


class NormCapTest(unittest.TestCase):
    def make(self, cap):
        self.keys = {p: bytes([p + 1]) * 32 for p in range(4)}
        self.got = []
        outer = self

        class Inner:
            def configure_train(self, rnd, arrays, config, grid):
                return []

            def aggregate_train(self, rnd, replies):
                outer.got.append({int(m.content["metrics"]["partition-id"]): arrays_of(m.content["arrays"])["s"]
                                  for m in replies})
                return None, None

        g = GuardedStrategy(Inner(), None, dict(self.keys), norm_cap=cap)
        g.configure_train(1, ArrayRecord({"s": Array(np.zeros(2, np.float32))}), ConfigRecord(), None)
        return g

    def reply(self, pid, value):
        sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
        c = RecordDict({"arrays": ArrayRecord({"s": Array(np.asarray(value, np.float32))}),
                        "metrics": MetricRecord({"partition-id": pid, "num-examples": 10})})
        attach_update_tag(c, self.keys[pid], 1, pid)
        return Message(content=c, reply_to=sent)

    def test_oversized_update_is_scaled_to_cap(self):
        g = self.make(2.0)
        g.aggregate_train(1, [self.reply(0, [1, 0]), self.reply(1, [0, 1]), self.reply(2, [-1, 0]),
                              self.reply(3, [30, 40])])  # 長度 50，中位數 1 → 縮成長度 2
        np.testing.assert_allclose(self.got[-1][3], [1.2, 1.6], rtol=1e-6)
        np.testing.assert_allclose(self.got[-1][0], [1, 0])

    def test_off_by_default(self):
        g = self.make(0.0)
        g.aggregate_train(1, [self.reply(0, [1, 0]), self.reply(1, [0, 1]), self.reply(3, [30, 40])])
        np.testing.assert_allclose(self.got[-1][3], [30, 40])

    def test_cap_must_be_at_least_one_and_not_with_secure_agg(self):
        with self.assertRaises(ValueError):
            GuardedStrategy(None, None, {}, norm_cap=0.5)
        with self.assertRaises(ValueError):
            GuardedStrategy(None, None, {}, norm_cap=2.0, secure_agg_clients=4)


if __name__ == "__main__":
    unittest.main()
