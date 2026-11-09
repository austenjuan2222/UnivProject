"""SecAgg 每輪臨時金鑰：訓練前公告臨時公鑰、臨時私鑰切份；訓練斷線的 VD 由別人交出的份還原臨時私鑰，
伺服器自己算出它和每台在線 VD 的成對種子。所以同一輪「有人訓練斷線」加上「有人復原時沉默」也救得回來。"""
import tempfile
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from client_app import SECAGG_STATE_KEY, claim_mask_round, pair_keys_from_config, secagg_recover_fn
from guard import GuardedStrategy, secure_average
from secagg import (encrypt_shares, ephemeral_key, load_agreement_keys, load_agreement_pubs,
                    load_server_agreement_key, mask_update, open_share, provision_agreement_keys, seed_tag, self_seed)
from security import load_server_secrets, provision_keys

N, RND = 6, 3


def _err(m, e):
    from flwr.app import Error
    return Message(Error(code=0, reason=str(e)), reply_to=m)


class EphemeralTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        provision_agreement_keys(d, N)
        provision_keys(d, N)
        self.spriv, _ = load_server_secrets(d)
        self.pubs = load_agreement_pubs(d)
        self.privs = {i: load_agreement_keys(d, i)[0] for i in range(N)}
        self.ctxs = {}
        from flwr.app import Context
        for i in range(N):
            self.ctxs[i] = Context(run_id=0, node_id=100 + i, node_config={"partition-id": i, "num-partitions": N},
                                   state=RecordDict(), run_config={"key-dir": d, "secagg-shamir": True})
        self.silent = set()
        test = self

        class Grid:
            def send_and_receive(self, messages, timeout=None):
                out = []
                for m in messages:
                    pid = m.metadata.dst_node_id - 100
                    phase = str(m.content["secagg"]["phase"]) if "phase" in m.content["secagg"] else ""
                    if pid in test.silent and phase == "":
                        continue  # 只在復原時沉默
                    try:
                        out.append(secagg_recover_fn(m, test.ctxs[pid]))
                    except RuntimeError as e:
                        out.append(_err(m, e))
                return out
        self.g = GuardedStrategy(object(), self.spriv, None, secure_agg_clients=N, secagg_self_mask=True,
                                 secagg_keys=(load_server_agreement_key(d), self.pubs), secagg_shamir=True)
        self.g._grid = Grid()

    def tearDown(self):
        self.tmp.cleanup()

    def _train_msgs(self):
        msgs = [Message(content=RecordDict({"config": ConfigRecord({"server-round": RND})}), dst_node_id=100 + i,
                        message_type="train") for i in range(N)]
        return {m.metadata.dst_node_id - 100: m for m in self.g._advertise(RND, msgs)}

    def test_train_dropout_plus_silent_recovery_gives_exact_average(self):
        msgs = self._train_msgs()
        rng = np.random.default_rng(0)
        alive = [0, 1, 2, 3, 4]  # VD 5 訓練時斷線
        self.silent = {4}  # VD 4 上傳了，但復原時沉默
        upd = {i: {"w": rng.normal(0, 0.01, (3, 4))} for i in alive}
        n_ex = {i: 10 + i for i in alive}
        contents, kept = [], []
        for i in alive:
            eph, epubs = pair_keys_from_config(self.privs[i], self.pubs, i, N, RND, 0, msgs[i].content["secagg-eph"])
            claim_mask_round(self.ctxs[i], RND)
            masked = mask_update(upd[i], n_ex[i], i, range(N), self.privs[i], self.pubs, RND, 0, self_mask=True,
                                 pair_priv=eph, pair_pubs=epubs)
            b = self_seed(self.privs[i], i, RND, 0)
            self.g._shares[i] = encrypt_shares(b, i, range(N), self.privs[i], self.pubs, RND, 0)
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

    def test_forged_ephemeral_key_is_refused(self):
        rec = self._train_msgs()[0].content["secagg-eph"]
        epubs = [str(x) for x in rec["epubs"]]
        epubs[1] = ephemeral_key(self.privs[2], 2, RND, 0).public_key().public_bytes_raw().hex()  # 伺服器調包
        bad = ConfigRecord({"epubs": epubs, "macs": list(rec["macs"])})
        with self.assertRaises(RuntimeError):
            pair_keys_from_config(self.privs[0], self.pubs, 0, N, RND, 0, bad)

    def test_self_seed_share_cannot_pass_as_ephemeral_share(self):
        # 伺服器把 VD 5 的自我遮罩種子份冒充成臨時私鑰的份：AAD 不同，解不開（不然兩種都拿到就能解開遲到的更新）
        b_ct = encrypt_shares(self_seed(self.privs[5], 5, RND, 0), 5, range(N), self.privs[5], self.pubs, RND, 0)
        self.assertIsNone(open_share(b_ct[0], 5, 0, self.privs[0], self.pubs, RND, 0, kind="eshare"))

    def test_eshare_owners_must_be_exactly_the_dropped_list(self):
        self._train_msgs()
        self.g._shares = {i: encrypt_shares(self_seed(self.privs[i], i, RND, 0), i, range(N), self.privs[i],
                                            self.pubs, RND, 0) for i in range(5)}
        for i in range(N):
            claim_mask_round(self.ctxs[i], RND)
        macs = self.g._consist(RND, [5], {i: 100 + i for i in range(5)})
        q = self.g._recover_query(RND, [5], [0, 1, 2, 3, 4], macs)
        c = dict(q(0)["secagg"])
        c["eshare-owners"], c["eshare-cts"] = [4, 5], [self.g._eshares[4][0], self.g._eshares[5][0]]
        with self.assertRaisesRegex(RuntimeError, "臨時私鑰"):
            secagg_recover_fn(Message(content=RecordDict({"secagg": ConfigRecord(c)}), dst_node_id=100,
                                      message_type="query"), self.ctxs[0])

    def test_low_order_ephemeral_key_is_refused_cleanly(self):
        # 稽核 LOW：惡意 VD 公告全零公鑰（低階點），別台算共享金鑰會丟 ValueError；要變成「拒絕這輪」而不是當掉
        from secagg import epub_mac
        rec = self._train_msgs()[0].content["secagg-eph"]
        epubs, macs = [str(x) for x in rec["epubs"]], [str(x) for x in rec["macs"]]
        epubs[1] = "00" * 32
        macs[1] = epub_mac(self.privs[1], self.pubs[0], 1, 0, RND, 0, epubs[1])
        with self.assertRaises(RuntimeError):
            pair_keys_from_config(self.privs[0], self.pubs, 0, N, RND, 0, ConfigRecord({"epubs": epubs, "macs": macs}))

    def test_advertise_refused_without_self_mask(self):
        # 稽核 LOW：沒有自我遮罩時交出臨時私鑰的份，伺服器謊稱它斷線就能解開它的更新
        self.ctxs[0].run_config["secagg-self-mask"] = False
        q = ConfigRecord({"round": RND, "phase": "advertise"})
        with self.assertRaises(RuntimeError):
            secagg_recover_fn(Message(content=RecordDict({"secagg": q}), dst_node_id=100, message_type="query"),
                              self.ctxs[0])

    def test_missing_ephemeral_shares_gives_up_instead_of_crashing(self):
        msgs = self._train_msgs()
        del self.g._eshares[5]
        self.g._shares = {i: {j: "00" for j in range(N)} for i in range(5)}
        kept = [Message(content=RecordDict({"metrics": MetricRecord({"partition-id": i})}), reply_to=msgs[i])
                for i in range(5)]
        self.assertIsNone(self.g._recover(RND, kept))

    def test_advertise_payload_is_binary(self):
        # LPWAN：公告回合的公鑰、MAC、份都用二進位傳（十六進位會多一倍）
        q = ConfigRecord({"round": RND, "phase": "advertise"})
        c = secagg_recover_fn(Message(content=RecordDict({"secagg": q}), dst_node_id=100, message_type="query"),
                              self.ctxs[0]).content["secagg"]
        up = [c["epub"], *c["epub-macs"], *c["eshares"]]
        self.assertTrue(all(isinstance(x, bytes) for x in up))
        self.assertLessEqual(sum(len(x) for x in up), 32 + 32 * (N - 1) + 82 * N)
        rec = self._train_msgs()[1].content["secagg-eph"]
        self.assertTrue(all(isinstance(x, bytes) for x in [*rec["epubs"], *rec["macs"]]))

    def test_non_binary_advertise_reply_is_ignored(self):
        # 惡意 VD 回十六進位以外的字串：伺服器忽略它（這輪不附公鑰），不能在 configure_train 裡當掉
        inner = self.g._grid

        class Bad:
            def send_and_receive(self, messages, timeout=None):
                out = inner.send_and_receive(messages, timeout)
                c = dict(out[0].content["secagg"])
                c["epub-macs"] = ["zz"] * N
                return [Message(content=RecordDict({"secagg": ConfigRecord(c)}), reply_to=messages[0])] + out[1:]
        self.g._grid = Bad()
        msgs = self._train_msgs()
        self.assertNotIn("secagg-eph", msgs[1].content.keys())


if __name__ == "__main__":
    unittest.main()
