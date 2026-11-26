"""robust_agg.py：Krum／Multi-Krum／Bulyan（距離式穩健聚合，概念取自 FLYOLO 的策略比較）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from robust_agg import AGGREGATORS, make_strategy


def _msg(v, pid):
    c = RecordDict({"arrays": ArrayRecord({"w": Array(np.full(3, v, dtype=np.float32))}),
                    "metrics": MetricRecord({"num-examples": 10, "partition-id": pid})})
    return Message(content=c, dst_node_id=pid + 1, message_type="train")


def _replies(n_honest, n_bad, bad=100.0):
    return [_msg(1.0 + 0.01 * i, i) for i in range(n_honest)] + \
           [_msg(bad, n_honest + j) for j in range(n_bad)]


def _strategy(name, f, n, seen=None, **kw):
    seen = [] if seen is None else seen
    fn = lambda rs, _k: (seen.append(len(rs)), MetricRecord({}))[1]
    return make_strategy(name, f, n, dict(fraction_train=1.0, min_train_nodes=2,
                                          min_available_nodes=2, train_metrics_aggr_fn=fn), **kw)


class RobustAggTest(unittest.TestCase):
    def test_registry_keeps_old_names(self):
        for name in ("fedavg", "median", "trimmed", "krum", "multikrum", "bulyan"):
            self.assertIn(name, AGGREGATORS)

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            _strategy("foo", 0, 8)

    def test_krum_ignores_outliers(self):
        arrays, _ = _strategy("krum", 2, 10).aggregate_train(1, _replies(8, 2))
        self.assertLess(float(arrays["w"].numpy().max()), 1.1)

    def test_multikrum_averages_honest_updates(self):
        arrays, _ = _strategy("multikrum", 2, 10).aggregate_train(1, _replies(8, 2))
        w = arrays["w"].numpy()
        self.assertLess(float(w.max()), 1.1)
        self.assertGreater(float(w.min()), 1.0)  # 平均多份，不是只挑一份

    def test_multikrum_dropout_does_not_select_every_reply(self):
        # 設定 n=10、f=2（預設挑 8 份）；兩台誠實 VD 斷線只剩 6 誠實＋2 惡意，不能 8 份全挑
        arrays, _ = _strategy("multikrum", 2, 10).aggregate_train(1, _replies(6, 2))
        self.assertLess(float(arrays["w"].numpy().max()), 1.1)

    def test_krum_skips_round_below_byzantine_threshold(self):
        # 斷線或被拒後有效回覆少於 2f+3（Bulyan 4f+3）：這輪跳過，不在不符假設的集合上聚合
        for name, f, n, honest in (("krum", 2, 10, 4), ("multikrum", 2, 10, 4), ("bulyan", 2, 12, 8)):
            with self.subTest(name=name):
                arrays, metrics = _strategy(name, f, n).aggregate_train(1, _replies(honest, 2))
                self.assertIsNone(arrays)
                self.assertIsNone(metrics)

    def test_bulyan_ignores_outliers(self):
        arrays, _ = _strategy("bulyan", 2, 12).aggregate_train(1, _replies(10, 2))
        self.assertLess(float(arrays["w"].numpy().max()), 1.1)

    def test_bulyan_needs_enough_clients(self):
        # Bulyan 要求 n ≥ 4f + 3；設定時就擋，不要跑到一半整輪被跳過
        with self.assertRaises(ValueError):
            _strategy("bulyan", 3, 14)
        _strategy("bulyan", 3, 15)

    def test_krum_needs_enough_clients(self):
        with self.assertRaises(ValueError):
            _strategy("krum", 4, 10)  # 需要 n > 2f + 2

    def test_metrics_cover_every_reply_not_only_selected(self):
        # 紀錄程式要看到所有 VD 的訓練指標（ε、loss），不能只看到被 Krum 選中的那一份
        seen = []
        _strategy("krum", 2, 10, seen).aggregate_train(1, _replies(8, 2))
        self.assertEqual(seen, [10])

    def test_negative_f_rejected(self):
        with self.assertRaises(ValueError):
            _strategy("krum", -1, 10)


if __name__ == "__main__":
    unittest.main()
