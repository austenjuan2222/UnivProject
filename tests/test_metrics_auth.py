"""完整性：HMAC 要涵蓋 VD 回報的指標（訓練與評估），不然中間人可以改 train-loss、誤差，讓紀錄與結論失真。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from guard import GuardedStrategy, attach_update_tag


class MetricsAuthTest(unittest.TestCase):
    def setUp(self):
        self.keys = {p: bytes([p + 1]) * 32 for p in range(3)}
        self.train, self.evals = [], []
        outer = self

        class Inner:
            def configure_train(self, rnd, arrays, config, grid):
                return []

            def aggregate_train(self, rnd, replies):
                outer.train.append(sorted(int(m.content["metrics"]["partition-id"]) for m in replies))
                return None, None

            def aggregate_evaluate(self, rnd, replies):
                outer.evals.append(sorted(int(m.content["metrics"]["partition-id"]) for m in replies))

        self.g = GuardedStrategy(Inner(), None, dict(self.keys))
        self.g.configure_train(1, ArrayRecord({"s": Array(np.zeros(2, np.float32))}), ConfigRecord(), None)

    def train_reply(self, pid, after=None):
        c = RecordDict({"arrays": ArrayRecord({"s": Array(np.ones(2, np.float32))}),
                        "metrics": MetricRecord({"partition-id": pid, "num-examples": 10, "train-loss": 0.5})})
        attach_update_tag(c, self.keys[pid], 1, pid)
        if after:
            after(c)
        return Message(content=c, reply_to=Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train"))

    def eval_reply(self, pid, sign=True, after=None):
        c = RecordDict({"metrics": MetricRecord({"partition-id": pid, "num-examples": 5, "round": 1,
                                                 "eval-loss": 0.2, "mae-kmh": 3.0, "mse-kmh": 9.0})})
        if sign:
            attach_update_tag(c, self.keys[pid], 1, pid)
        if after:
            after(c)
        return Message(content=c, reply_to=Message(content=RecordDict(), dst_node_id=100 + pid, message_type="evaluate"))

    def test_tampered_train_metric_is_rejected(self):
        def lie(c):
            c["metrics"]["train-loss"] = 0.01
        self.g.aggregate_train(1, [self.train_reply(0), self.train_reply(1, lie), self.train_reply(2)])
        self.assertEqual(self.train[-1], [0, 2])

    def test_eval_replies_are_authenticated(self):
        def lie(c):
            c["metrics"]["mae-kmh"] = 0.1
        self.g.aggregate_evaluate(1, [self.eval_reply(0), self.eval_reply(1, sign=False), self.eval_reply(2, after=lie)])
        self.assertEqual(self.evals[-1], [0])

    def test_eval_tag_cannot_be_reused_for_another_round(self):
        self.g.aggregate_evaluate(2, [self.eval_reply(0)])  # 第 1 輪簽的，第 2 輪拿來用
        self.assertEqual(self.evals[-1], [])


if __name__ == "__main__":
    unittest.main()
