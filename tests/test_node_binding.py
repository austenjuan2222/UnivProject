"""VD 編號要綁定送出它的節點；撤銷的 VD 金鑰在應用層也要失效（對應 RFID 講義的複製標籤與撤銷）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from guard import GuardedStrategy, attach_update_tag


class NodeBindingTest(unittest.TestCase):
    def setUp(self):
        self.keys = {p: bytes([p + 1]) * 32 for p in range(4)}
        self.seen = []
        outer = self

        class Inner:
            def aggregate_train(self, rnd, replies):
                outer.seen.append(sorted(int(m.content["metrics"]["partition-id"]) for m in replies))
                return None, None

        self.stats = {}
        self.g = GuardedStrategy(Inner(), None, dict(self.keys), on_round=lambda r, s: self.stats.update(s))

    def reply(self, node, pid, rnd, key_of=None):
        sent = Message(content=RecordDict(), dst_node_id=node, message_type="train")
        c = RecordDict({"arrays": ArrayRecord({"s": Array(np.array([1.0], np.float32))}),
                        "metrics": MetricRecord({"partition-id": pid, "num-examples": 10})})
        attach_update_tag(c, self.keys[pid if key_of is None else key_of], rnd, pid)
        return Message(content=c, reply_to=sent)

    def test_stolen_key_cannot_be_used_from_another_node(self):
        self.g.aggregate_train(1, [self.reply(100 + p, p, 1) for p in range(4)])
        self.assertEqual(self.seen[-1], [0, 1, 2, 3])
        # 第 2 輪：節點 101（原本是 1 號）偷了 0 號的金鑰，改用 0 號的身分送；0 號自己這輪斷線
        self.g.aggregate_train(2, [self.reply(101, 0, 2)] + [self.reply(100 + p, p, 2) for p in (2, 3)])
        self.assertEqual(self.seen[-1], [2, 3])
        self.assertEqual(self.stats["n_rejected"], 1)

    def test_revoked_vd_is_rejected_even_with_valid_hmac(self):
        self.g.aggregate_train(1, [self.reply(100 + p, p, 1) for p in range(4)])
        self.g.revoke(0)
        self.g.aggregate_train(2, [self.reply(100 + p, p, 2) for p in range(4)])
        self.assertEqual(self.seen[-1], [1, 2, 3])

    def test_impostor_cannot_get_bound_vd_rejected_by_claiming_its_id(self):
        self.g.aggregate_train(1, [self.reply(100, 0, 1), self.reply(101, 1, 1), self.reply(105, 3, 1)])
        # 已綁的 0 號照常送；101（已綁 1 號）和 105 也送 0 號，不能讓真正的 0 號被當成「同輪搶編號」而拒絕
        self.g.aggregate_train(2, [self.reply(100, 0, 2), self.reply(101, 0, 2), self.reply(105, 0, 2)])
        self.assertEqual(self.seen[-1], [0])
        # 還沒綁的 2 號：已綁別的編號的節點來搶，也不能擋掉真正的 2 號
        self.g.aggregate_train(3, [self.reply(102, 2, 3), self.reply(101, 2, 3)])
        self.assertEqual(self.seen[-1], [2])

    def test_one_node_counts_once_per_round(self):
        self.g.aggregate_train(1, [self.reply(100, 0, 1), self.reply(100, 0, 1), self.reply(101, 1, 1)])
        self.assertEqual(self.seen[-1], [0, 1])

    def test_revoke_without_verification_is_an_error(self):
        g = GuardedStrategy(None, None, None)
        with self.assertRaises(ValueError):
            g.revoke(0)

    def test_revoked_node_cannot_take_another_id(self):
        self.g.aggregate_train(1, [self.reply(100, 0, 1)])
        self.g.revoke(0)
        # 被撤銷的節點 100 拿偷來的 3 號金鑰（3 號還沒出現過）也不行
        self.g.aggregate_train(2, [self.reply(100, 3, 2), self.reply(102, 2, 2)])
        self.assertEqual(self.seen[-1], [2])

    def test_first_contact_binds_node(self):
        # 第一次看到的節點與編號就綁定（TOFU）；之後同一節點換編號也拒絕
        self.g.aggregate_train(1, [self.reply(100, 0, 1)])
        self.g.aggregate_train(2, [self.reply(100, 1, 2), self.reply(102, 2, 2)])
        self.assertEqual(self.seen[-1], [2])
        self.assertEqual(self.stats["n_rejected"], 1)

    def test_rejected_claim_does_not_bind_the_victim_id(self):
        # 節點 101 已綁 1 號；它冒用還沒出現過的 0 號被拒絕後，不能順便把 0 號綁到自己，鎖住真正的 0 號
        self.g.aggregate_train(1, [self.reply(101, 1, 1)])
        self.g.aggregate_train(2, [self.reply(101, 0, 2, key_of=0)])
        self.g.aggregate_train(3, [self.reply(100, 0, 3), self.reply(101, 1, 3)])
        self.assertEqual(self.seen[-1], [0, 1])

    def test_two_nodes_claiming_same_unbound_id_in_one_round_are_both_rejected(self):
        self.g.aggregate_train(1, [self.reply(100, 0, 1), self.reply(105, 0, 1), self.reply(102, 2, 1)])
        self.assertEqual(self.seen[-1], [2])
        self.g.aggregate_train(2, [self.reply(100, 0, 2), self.reply(102, 2, 2)])
        self.assertEqual(self.seen[-1], [0, 2])  # 沒有被搶先綁走：下一輪單獨出現時才綁

if __name__ == "__main__":
    unittest.main()
