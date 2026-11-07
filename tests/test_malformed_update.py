"""可用性：持有合法金鑰的 VD 送 NaN／形狀不對的更新，不能讓全域模型變 NaN 或讓聚合當掉（沒開串通偵測時也要擋）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from guard import GuardedStrategy, attach_update_tag


class MalformedUpdateTest(unittest.TestCase):
    def setUp(self):
        self.keys = {p: bytes([p + 1]) * 32 for p in range(3)}
        self.seen, self.stats = [], {}
        outer = self

        class Inner:
            def configure_train(self, rnd, arrays, config, grid):
                return []

            def aggregate_train(self, rnd, replies):
                outer.seen.append(sorted(int(m.content["metrics"]["partition-id"]) for m in replies))
                return None, None

        self.g = GuardedStrategy(Inner(), None, dict(self.keys), on_round=lambda r, s: self.stats.update(s))
        self.g.configure_train(1, ArrayRecord({"s": Array(np.zeros(2, np.float32))}), ConfigRecord(), None)

    def reply(self, pid, value, n=10, dtype=np.float32, after_tag=None):
        sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
        c = RecordDict({"arrays": ArrayRecord({"s": Array(np.asarray(value, dtype))}),
                        "metrics": MetricRecord({"partition-id": pid, "num-examples": n})})
        attach_update_tag(c, self.keys[pid], 1, pid)
        if after_tag:
            after_tag(c)
        return Message(content=c, reply_to=sent)

    def test_nan_update_with_valid_hmac_is_rejected(self):
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [np.nan, 1]), self.reply(2, [1, 1])])
        self.assertEqual(self.seen[-1], [0, 2])
        self.assertEqual(self.stats["n_rejected"], 1)

    def test_wrong_shape_update_is_rejected(self):
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1, 1]), self.reply(2, [np.inf, 0])])
        self.assertEqual(self.seen[-1], [0])
        self.assertEqual(self.stats["n_rejected"], 2)

    def test_bad_num_examples_with_valid_hmac_is_rejected(self):
        # FedAvg 用 num-examples 加權：NaN／inf 讓全域模型變 NaN，總和為 0 會除以零，非整數也不合理
        for bad in (float("nan"), float("inf"), 0, -20, 1.5, 2**62):
            self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], n=bad), self.reply(2, [1, 1])])
            self.assertEqual(self.seen[-1], [0, 2], bad)

    def test_inflated_num_examples_rejected_even_in_small_round(self):
        # 只有 2 份更新時沒有中位數可比，仍要有絕對上限
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], n=10**12)])
        self.assertEqual(self.seen[-1], [0])

    def test_malformed_audit_keeps_vd_id(self):
        audits = []
        self.g.on_audit = audits.extend
        def to_str(c):
            c["arrays"] = ArrayRecord({"s": Array(np.array(["a", "b"]))})
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], after_tag=to_str)])
        self.assertEqual([(a["pid"], a["accepted"]) for a in audits], [(0, True), (1, False)])

    def test_inflated_num_examples_is_rejected(self):
        # 自報 10 億筆資料等於一台 VD 取代整個模型
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], n=10**9), self.reply(2, [1, 1])])
        self.assertEqual(self.seen[-1], [0, 2])

    def test_num_examples_is_covered_by_hmac(self):
        # 中間人改 num-examples、參數不動：HMAC 要驗不過
        def bump(c):
            c["metrics"]["num-examples"] = 30
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], after_tag=bump), self.reply(2, [1, 1])])
        self.assertEqual(self.seen[-1], [0, 2])

    def test_float64_overflowing_float32_is_rejected(self):
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1e300, 1], dtype=np.float64), self.reply(2, [1, 1])])
        self.assertEqual(self.seen[-1], [0, 2])

    def test_non_float_dtype_does_not_crash(self):
        # 沒有合法金鑰也能送：字串或 datetime 陣列不能讓伺服器在驗證時當掉
        def to_str(c):
            c["arrays"] = ArrayRecord({"s": Array(np.array(["a", "b"]))})
        def to_complex(c):
            c["arrays"] = ArrayRecord({"s": Array(np.array([1 + 1j, 1], np.complex64))})
        self.g.aggregate_train(1, [self.reply(0, [1, 1]), self.reply(1, [1, 1], after_tag=to_str),
                                   self.reply(2, [1, 1], after_tag=to_complex)])
        self.assertEqual(self.seen[-1], [0])
        self.assertEqual(self.stats["n_rejected"], 2)

    def test_eval_reply_with_bad_num_examples_is_dropped(self):
        got = []
        self.g.inner.aggregate_evaluate = lambda rnd, replies: got.append(len(replies))
        def ev(n):
            c = RecordDict({"metrics": MetricRecord({"partition-id": 0, "num-examples": n, "eval-loss": float("nan")})})
            return Message(content=c, reply_to=Message(content=RecordDict(), dst_node_id=100, message_type="evaluate"))
        self.g.aggregate_evaluate(1, [ev(10), ev(float("nan")), ev(0)])
        self.assertEqual(got, [1])  # 誤差是 NaN 的照樣留（模型發散要記錄），只擋加權壞掉的


if __name__ == "__main__":
    unittest.main()
