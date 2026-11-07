"""guard.GuardedStrategy._recover：只接受來自正確 VD、內容完整的斷線復原回覆。"""
import unittest

from flwr.app import ConfigRecord, Message, MetricRecord, RecordDict

from guard import GuardedStrategy

SEED = "ab" * 32


def _kept(pid):
    sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
    return Message(content=RecordDict({"metrics": MetricRecord({"partition-id": pid})}), reply_to=sent)


class FakeGrid:
    def __init__(self, answer):
        self.answer = answer

    def send_and_receive(self, messages, timeout=None):
        return [Message(content=RecordDict({"secagg": ConfigRecord(self.answer(m.metadata.dst_node_id - 100))}),
                        reply_to=m) for m in messages]


class RecoverTest(unittest.TestCase):
    def _run(self, answer):
        g = GuardedStrategy(object(), None, None, secure_agg_clients=3)
        g._grid = FakeGrid(answer)
        return g._recover(1, [_kept(0), _kept(1)])  # VD 2 斷線

    def test_honest_replies_are_collected(self):
        self.assertEqual(self._run(lambda p: {"pid": p, "dropped": [2], "seeds": [SEED]}),
                         {0: {2: SEED}, 1: {2: SEED}})

    def test_bad_replies_abort_recovery(self):
        for bad in [lambda p: {"pid": 1 - p, "dropped": [2], "seeds": [SEED]},  # 冒充另一台
                    lambda p: {"pid": p, "dropped": [], "seeds": []},  # 少交
                    lambda p: {"pid": p, "dropped": [2], "seeds": ["zz"]},  # 種子格式錯
                    lambda p: {"pid": p}]:  # 缺欄位
            with self.subTest():
                self.assertIsNone(self._run(bad))


class SignedQueryTest(unittest.TestCase):
    def test_server_signs_the_recovery_request(self):
        import tempfile
        from security import load_server_public_key, load_server_secrets, provision_keys, verify_recover
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            priv, _ = load_server_secrets(d)
            sent = []
            class Grid(FakeGrid):
                def send_and_receive(self, messages, timeout=None):
                    sent.extend(messages)
                    return super().send_and_receive(messages, timeout)
            g = GuardedStrategy(object(), priv, None, secure_agg_clients=3)
            g._grid = Grid(lambda p: {"pid": p, "dropped": [2], "seeds": [SEED]})
            g._grid.run = type("Run", (), {"run_id": 7})()
            g._recover(1, [_kept(0), _kept(1)])
            q = sent[0].content["secagg"]
            self.assertTrue(verify_recover(load_server_public_key(d), 1, [2], str(q["sig"]), run=7))


class SealedSeedsTest(unittest.TestCase):
    def test_server_opens_sealed_seeds(self):
        import tempfile
        from secagg import load_agreement_pubs, load_agreement_keys, load_server_agreement_key, provision_agreement_keys, seal_seeds
        with tempfile.TemporaryDirectory() as d:
            provision_agreement_keys(d, 3)
            pubs = load_agreement_pubs(d)
            privs = {p: load_agreement_keys(d, p)[0] for p in (0, 1)}
            g = GuardedStrategy(object(), None, None, secure_agg_clients=3,
                                secagg_keys=(load_server_agreement_key(d), pubs))
            g._grid = FakeGrid(lambda p: {"pid": p, "dropped": [2],
                                          "seeds": list(seal_seeds({2: SEED}, p, privs[p], pubs["server"], 1).values())})
            self.assertEqual(g._recover(1, [_kept(0), _kept(1)]), {0: {2: SEED}, 1: {2: SEED}})
            # 中間人改一個位元：不能默默解出錯的種子（會讓全域模型壞掉），整輪復原放棄
            def flip(p):
                x = seal_seeds({2: SEED}, p, privs[p], pubs["server"], 1)[2]
                return {"pid": p, "dropped": [2], "seeds": [("0" if x[0] != "0" else "1") + x[1:]]}
            g._grid = FakeGrid(flip)
            self.assertIsNone(g._recover(1, [_kept(0), _kept(1)]))

    def test_self_mask_seeds_are_collected_every_round(self):
        import tempfile
        from secagg import load_agreement_pubs, load_agreement_keys, load_server_agreement_key, provision_agreement_keys, seal_seeds
        with tempfile.TemporaryDirectory() as d:
            provision_agreement_keys(d, 2)
            pubs = load_agreement_pubs(d)
            privs = {p: load_agreement_keys(d, p)[0] for p in (0, 1)}
            g = GuardedStrategy(object(), None, None, secure_agg_clients=2, secagg_self_mask=True,
                                secagg_keys=(load_server_agreement_key(d), pubs))
            S = "cd" * 32
            g._grid = FakeGrid(lambda p: {"pid": p, "dropped": [], "seeds": [],
                                          "self": seal_seeds({p: S}, p, privs[p], pubs["server"], 1)[p]})
            self.assertEqual(g._recover(1, [_kept(0), _kept(1)]), {0: {0: S}, 1: {1: S}})  # 沒人斷線也要收
            g._grid = FakeGrid(lambda p: {"pid": p, "dropped": [], "seeds": []})  # 少了自我遮罩種子
            self.assertIsNone(g._recover(1, [_kept(0), _kept(1)]))


class SelfMaskRoundTest(unittest.TestCase):
    """雙重遮罩時，被拒絕的回覆當成斷線處理（它的自我遮罩種子不會交出，更新仍解不開），整輪不作廢。"""
    def _reply(self, pid, arrays, flag=1, n=1, num_vd=3, dtype=None, bits=32, frac=12):
        import numpy as np
        from flwr.app import Array, ArrayRecord
        sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
        return Message(content=RecordDict({"arrays": ArrayRecord({"w": Array(np.asarray(arrays, dtype=dtype or np.uint32))}),
                                           "metrics": MetricRecord({"partition-id": pid, "num-examples": n,
                                                                    "secagg-self-mask": flag,
                                                                    "secagg-num-vd": num_vd,
                                                                    "secagg-bits": bits,
                                                                    "secagg-frac-bits": frac})}), reply_to=sent)

    def _run(self, replies, self_mask=True):
        class Inner:
            def aggregate_train(self, rnd, kept):
                return None, MetricRecord({})
        stats, calls = {}, []
        g = GuardedStrategy(Inner(), None, None, secure_agg_clients=3, secagg_self_mask=self_mask,
                            on_round=lambda r, st: stats.update(st))
        g._recover = lambda rnd, kept: calls.append(len(kept)) or {0: {}, 1: {}}
        arrays, _ = g.aggregate_train(1, replies)
        return arrays, stats, calls

    def test_rejected_reply_is_treated_as_dropped(self):
        bad = self._reply(2, [2.0], n=0)  # 格式不合被拒
        arrays, stats, calls = self._run([self._reply(0, [1.0]), self._reply(1, [3.0]), bad])
        self.assertEqual(calls, [2])
        self.assertIsNotNone(arrays)
        self.assertEqual(stats["secagg_failed"], 0)

    def test_flag_mismatch_fails_the_round(self):
        # VD 沒加自我遮罩、伺服器卻去扣：加總會默默壞掉，要整輪放棄
        arrays, stats, _ = self._run([self._reply(0, [1.0]), self._reply(1, [3.0], flag=0), self._reply(2, [2.0])])
        self.assertIsNone(arrays)
        self.assertEqual(stats["secagg_failed"], 1)

    def test_num_vd_mismatch_fails_the_round(self):
        # VD 的 num-partitions 和伺服器不同：成對遮罩對不上、定點範圍也可能溢位，要整輪放棄
        arrays, stats, _ = self._run([self._reply(0, [1.0]), self._reply(1, [3.0], num_vd=2), self._reply(2, [2.0])])
        self.assertIsNone(arrays)
        self.assertEqual(stats["secagg_failed"], 1)

    def test_ring_setting_mismatch_fails_the_round(self):
        # 小數位數或環的位元數和伺服器不同，解碼會差 2 的次方倍：整輪放棄
        for kw in ({"frac": 6}, {"bits": 16}):
            arrays, stats, _ = self._run([self._reply(0, [1.0]), self._reply(1, [3.0], **kw), self._reply(2, [2.0])])
            self.assertIsNone(arrays)
            self.assertEqual(stats["secagg_failed"], 1)

    def test_recovered_counts_only_real_dropout(self):
        _, stats, _ = self._run([self._reply(0, [1.0]), self._reply(1, [3.0]), self._reply(2, [2.0])])
        self.assertEqual(stats["secagg_recovered"], 0)


if __name__ == "__main__":
    unittest.main()
