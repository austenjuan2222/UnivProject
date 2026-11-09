"""SecAgg+ 鄰居分享：每台 VD 只和環上 k 個鄰居做成對遮罩、分自我遮罩份和臨時私鑰份（門檻 k/2+1）。"""
import tempfile
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict

from client_app import ADV_KEY, claim_mask_round, pair_keys_from_config, secagg_recover_fn
from guard import GuardedStrategy, secure_average
from secagg import (encrypt_shares, load_agreement_keys, load_agreement_pubs, load_server_agreement_key, mask_update,
                    neighbors, provision_agreement_keys, seed_tag, self_seed, share_threshold)
from security import load_server_secrets, provision_keys

N, K, RND = 10, 6, 3


def _err(m, e):
    from flwr.app import Error
    return Message(Error(code=0, reason=str(e)), reply_to=m)


class GraphTest(unittest.TestCase):
    def test_ring_neighbors(self):
        self.assertEqual(neighbors(0, 8, 4), [1, 2, 6, 7])
        self.assertEqual(neighbors(3, 8, 0), [0, 1, 2, 4, 5, 6, 7])  # k=0：所有其他 VD
        for i in range(8):
            for j in neighbors(i, 8, 4):
                self.assertIn(i, neighbors(j, 8, 4))  # 對稱，成對遮罩才抵銷得掉
        self.assertEqual(share_threshold(8, 4), 3)
        with self.assertRaises(ValueError):
            neighbors(0, 8, 3)  # 環狀圖要偶數
        with self.assertRaises(ValueError):
            neighbors(0, 8, 2)  # k=2 時門檻等於全部持有者，斷一台就整輪失敗

    def test_graph_is_random_per_round(self):
        # 每輪換一張圖：攻擊者無法固定收買某台的鄰居
        a, b = neighbors(0, 32, 8, rnd=1, run=0), neighbors(0, 32, 8, rnd=2, run=0)
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 8)
        for i in range(32):
            for j in neighbors(i, 32, 8, rnd=1, run=0):
                self.assertIn(i, neighbors(j, 32, 8, rnd=1, run=0))

    def test_macs_bind_k(self):
        # 伺服器給某台不同的 k，兩兩 MAC 就對不上
        from secagg import consist_mac, epub_mac
        d = tempfile.mkdtemp()
        provision_agreement_keys(d, 2)
        p0, pubs = load_agreement_keys(d, 0)[0], load_agreement_pubs(d)
        self.assertNotEqual(epub_mac(p0, pubs[1], 0, 1, 3, 0, "ab", k=8), epub_mac(p0, pubs[1], 0, 1, 3, 0, "ab", k=4))
        self.assertNotEqual(consist_mac(p0, pubs[1], 0, 1, 3, 0, [2], k=8), consist_mac(p0, pubs[1], 0, 1, 3, 0, [2], k=4))

    def test_mask_with_neighbors_still_cancels(self):
        # 只和鄰居遮罩，加總時還是全部抵銷（定點數範圍照總 VD 數算）
        from secagg import unmask_sum
        d = tempfile.mkdtemp()
        provision_agreement_keys(d, N)
        pubs = load_agreement_pubs(d)
        upd = {i: {"w": np.full((2, 2), 0.01 * i)} for i in range(N)}
        masked = [mask_update(upd[i], 1, i, neighbors(i, N, K, rnd=RND), load_agreement_keys(d, i)[0], pubs, RND,
                              num_vd=N) for i in range(N)]
        got = unmask_sum(masked, N)
        np.testing.assert_allclose(got["w"], np.full((2, 2), 0.01 * sum(range(N)) / N), atol=1e-3)


class NeighborRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        provision_agreement_keys(d, N)
        provision_keys(d, N)
        self.spriv, _ = load_server_secrets(d)
        self.pubs = load_agreement_pubs(d)
        self.privs = {i: load_agreement_keys(d, i)[0] for i in range(N)}
        self.ctxs = {i: Context(run_id=0, node_id=100 + i, node_config={"partition-id": i, "num-partitions": N},
                                state=RecordDict({ADV_KEY: MetricRecord({"round": RND - 1})}),
                                run_config={"key-dir": d, "secagg-shamir": True, "secagg-neighbors": K})
                     for i in range(N)}
        self.silent = set()
        test = self

        class Grid:
            def send_and_receive(self, messages, timeout=None):
                out = []
                for m in messages:
                    pid = m.metadata.dst_node_id - 100
                    phase = str(m.content["secagg"]["phase"]) if "phase" in m.content["secagg"] else ""
                    if pid in test.silent and phase == "":
                        continue
                    try:
                        out.append(secagg_recover_fn(m, test.ctxs[pid]))
                    except RuntimeError as e:
                        out.append(_err(m, e))
                return out
        self.g = GuardedStrategy(object(), self.spriv, None, secure_agg_clients=N, secagg_self_mask=True,
                                 secagg_keys=(load_server_agreement_key(d), self.pubs), secagg_shamir=True,
                                 secagg_neighbors=K)
        self.g._grid = Grid()

    def tearDown(self):
        self.tmp.cleanup()

    def _train_msgs(self):
        msgs = [Message(content=RecordDict({"config": ConfigRecord({"server-round": RND})}), dst_node_id=100 + i,
                        message_type="train") for i in range(N)]
        return {m.metadata.dst_node_id - 100: m for m in self.g._advertise(RND, msgs)}

    def test_advertise_only_to_neighbors(self):
        q = ConfigRecord({"round": RND, "phase": "advertise"})
        c = secagg_recover_fn(Message(content=RecordDict({"secagg": q}), dst_node_id=100, message_type="query"),
                              self.ctxs[0]).content["secagg"]
        self.assertEqual(sum(1 for x in c["eshares"] if x), K)
        self.assertEqual(sum(1 for x in c["epub-macs"] if x), K)
        self.ctxs[0].state[ADV_KEY] = MetricRecord({"round": RND - 1})  # 上面已直接公告過這輪
        rec = self._train_msgs()[0].content["secagg-eph"]
        self.assertEqual(sum(1 for x in rec["epubs"] if x), K + 1)  # 鄰居＋自己

    def test_vd_refuses_k_below_its_floor(self):
        # VD 自己設的最小鄰居數（node_config，伺服器改不了）
        self.ctxs[0].node_config["secagg-min-neighbors"] = 8
        q = ConfigRecord({"round": RND, "phase": "advertise"})
        with self.assertRaises(RuntimeError):
            secagg_recover_fn(Message(content=RecordDict({"secagg": q}), dst_node_id=100, message_type="query"),
                              self.ctxs[0])

    def test_advertise_rounds_must_be_consecutive(self):
        # 伺服器不能跳輪次挑一張對它有利的鄰居圖：公告回合只接受上一輪＋1
        ctx = self.ctxs[0]

        def adv(r):
            q = ConfigRecord({"round": r, "phase": "advertise"})
            return secagg_recover_fn(Message(content=RecordDict({"secagg": q}), dst_node_id=100, message_type="query"),
                                     ctx)
        adv(RND)
        for r in (RND, RND + 2, RND - 1):  # 重送、跳輪、倒退
            with self.assertRaises(RuntimeError):
                adv(r)
        adv(RND + 1)

    def test_train_dropout_plus_silent_recovery_gives_exact_average(self):
        msgs = self._train_msgs()
        rng = np.random.default_rng(1)
        alive = list(range(9))  # VD 9 訓練斷線
        self.silent = {4}  # VD 4 復原時沉默：每台最多少 2 個持有者，還有 4 個 ≥ 門檻 4
        upd = {i: {"w": rng.normal(0, 0.01, (3, 4))} for i in alive}
        n_ex = {i: 10 + i for i in alive}
        contents, kept = [], []
        for i in alive:
            eph, epubs = pair_keys_from_config(self.privs[i], self.pubs, i, N, RND, 0,
                                               msgs[i].content["secagg-eph"], k=K)
            claim_mask_round(self.ctxs[i], RND)
            masked = mask_update(upd[i], n_ex[i], i, neighbors(i, N, K, rnd=RND), self.privs[i], self.pubs, RND, 0,
                                 self_mask=True, pair_priv=eph, pair_pubs=epubs, num_vd=N)
            b = self_seed(self.privs[i], i, RND, 0)
            self.g._shares[i] = encrypt_shares(b, i, neighbors(i, N, K, rnd=RND), self.privs[i], self.pubs, RND, 0,
                                               t=share_threshold(N, K))
            self.g._self_hash[i] = seed_tag(b)
            contents.append(RecordDict({"arrays": ArrayRecord({k: Array(v) for k, v in masked.items()}),
                                        "metrics": MetricRecord({"num-examples": n_ex[i]})}))
            kept.append(Message(content=RecordDict({"metrics": MetricRecord({"partition-id": i})}),
                                reply_to=msgs[i]))
        rev = self.g._recover(RND, kept)
        self.assertIsNotNone(rev)
        got = secure_average(contents, N, rev)
        want = sum(n_ex[i] * upd[i]["w"] for i in alive) / sum(n_ex.values())
        np.testing.assert_allclose(np.asarray(got["w"].numpy()), want, atol=1e-3)

    def test_too_few_holders_gives_up(self):
        # VD 1 的 6 個持有者裡，1 台斷線、2 台沉默，只剩 3 個 < 門檻 4 → 放棄（不默默算錯）
        h = neighbors(1, N, K, rnd=RND)
        drop = h[0]
        self.silent = set(h[1:3])
        msgs = self._train_msgs()
        alive = [i for i in range(N) if i != drop]
        for i in alive:
            claim_mask_round(self.ctxs[i], RND)
            self.g._shares[i] = encrypt_shares(self_seed(self.privs[i], i, RND, 0), i, neighbors(i, N, K, rnd=RND),
                                               self.privs[i], self.pubs, RND, 0, t=share_threshold(N, K))
            self.g._self_hash[i] = seed_tag(self_seed(self.privs[i], i, RND, 0))
        kept = [Message(content=RecordDict({"metrics": MetricRecord({"partition-id": i})}), reply_to=msgs[i])
                for i in alive]
        self.assertIsNone(self.g._recover(RND, kept))


if __name__ == "__main__":
    unittest.main()
