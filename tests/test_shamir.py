"""Shamir 秘密分享自我遮罩種子（SecAgg 的做法）：VD 自己不直接交出種子，伺服器要湊滿門檻份數才能還原。"""
import tempfile
import unittest

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict

from client_app import SECAGG_STATE_KEY, secagg_recover_fn
from guard import GuardedStrategy
from client_app import claim_mask_round
from secagg import (consist_mac, consistency_threshold, seed_tag, encrypt_shares, load_agreement_keys, load_agreement_pubs, load_server_agreement_key,
                    open_share, provision_agreement_keys, self_seed, shamir_combine, shamir_split, shamir_threshold)
from security import load_server_secrets, provision_keys, sign_recover


class ShamirMathTest(unittest.TestCase):
    def test_any_threshold_subset_recovers_and_fewer_do_not(self):
        secret = bytes(range(32))
        shares = shamir_split(secret, [0, 1, 2, 3, 4], 3)
        self.assertEqual(shamir_combine({j: shares[j] for j in (0, 2, 4)}), secret)
        self.assertEqual(shamir_combine({j: shares[j] for j in (1, 3, 4)}), secret)

    def test_too_few_shares_can_be_detected(self):
        # 份數不足通常內插出 ≥ 2^256 的數：回傳 None，不要默默截成一顆錯的種子
        shares = shamir_split(bytes(range(32)), [0, 1, 2, 3, 4], 3)
        self.assertIsNone(shamir_combine({j: shares[j] for j in (0, 1)}))
        self.assertIsNone(shamir_combine({}))

    def test_seed_tag_survives_float_metrics(self):
        # 指標會經過 float：超過 2^53 會失真，伺服器每輪都會誤判（第一版就是這樣整輪作廢）
        for i in range(50):
            t = seed_tag(bytes([i]) * 32)
            self.assertEqual(int(float(t)), t)

    def test_threshold_is_majority(self):
        self.assertEqual(shamir_threshold(32), 17)
        self.assertEqual(shamir_threshold(3), 2)


class ShareEncryptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_round_gives_same_ciphertexts(self):
        # 係數由種子導出：重算不會用同一個 nonce 加密不同的明文
        p0, pubs = load_agreement_keys(self.tmp.name, 0)
        a = encrypt_shares(b"" * 32, 0, [0, 1, 2], p0, pubs, rnd=4, run=9)
        self.assertEqual(a, encrypt_shares(b"" * 32, 0, [0, 1, 2], p0, pubs, rnd=4, run=9))

    def test_only_the_holder_opens_its_share(self):
        p0, pubs = load_agreement_keys(self.tmp.name, 0)
        cts = encrypt_shares(b"\x07" * 32, 0, [0, 1, 2], p0, pubs, rnd=4, run=9)
        p1, _ = load_agreement_keys(self.tmp.name, 1)
        p2, _ = load_agreement_keys(self.tmp.name, 2)
        self.assertIsNotNone(open_share(cts[1], 0, 1, p1, pubs, 4, 9))
        self.assertIsNone(open_share(cts[1], 0, 2, p2, pubs, 4, 9))  # 別人的份
        self.assertIsNone(open_share(cts[1], 0, 1, p1, pubs, 5, 9))  # 別輪
        got = {j: open_share(cts[j], 0, j, load_agreement_keys(self.tmp.name, j)[0], pubs, 4, 9) for j in (1, 2)}
        self.assertEqual(shamir_combine(got), b"\x07" * 32)


def _err(m, e):
    from flwr.app import Error
    return Message(Error(code=0, reason=str(e)), reply_to=m)


def _ctx(d, pid, n):
    ctx = Context(run_id=1, node_id=pid, node_config={"partition-id": pid, "num-partitions": n},
                  state=RecordDict(), run_config={"key-dir": d, "secagg-shamir": True})
    ctx.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": 3, "revealed-round": -1})
    return ctx


class VdShareReplyTest(unittest.TestCase):
    """VD 只交出「名單上在線」那些 VD 的份，自己的種子不直接交出。"""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 4)
        provision_keys(self.tmp.name, 4)
        self.priv, _ = load_server_secrets(self.tmp.name)
        self.pubs = load_agreement_pubs(self.tmp.name)
        self.cts = {i: encrypt_shares(self_seed(load_agreement_keys(self.tmp.name, i)[0], i, 3, 1), i, range(4),
                                      load_agreement_keys(self.tmp.name, i)[0], self.pubs, 3, 1) for i in range(4)}

    def tearDown(self):
        self.tmp.cleanup()

    def _q(self, c):
        return Message(content=RecordDict({"secagg": ConfigRecord(c)}), dst_node_id=1, message_type="query")

    def _ask(self, pid, dropped, owners, attesters=None, forge=False, ctx=None):
        base = {"round": 3, "dropped": dropped, "sig": sign_recover(self.priv, 3, dropped, run=1)}
        if ctx is None:
            ctx = _ctx(self.tmp.name, pid, 4)
            secagg_recover_fn(self._q({**base, "phase": "consist"}), ctx)  # 先替自己看到的名單做證明
        if attesters is None:
            attesters = [i for i in range(4) if i != pid and i not in dropped]
        macs = [consist_mac(load_agreement_keys(self.tmp.name, i)[0], self.pubs[pid], i, pid, 3, 1,
                            [] if forge else dropped) for i in attesters]
        q = {**base, "share-owners": list(owners), "share-cts": [self.cts[i][pid] for i in owners],
             "attest-from": list(attesters), "attest-macs": macs}
        return secagg_recover_fn(self._q(q), ctx)

    def test_gives_shares_for_alive_owners_and_no_plain_self_seed(self):
        out = self._ask(0, [3], [0, 1, 2]).content["secagg"]
        self.assertNotIn("self", out)
        self.assertEqual(sorted(int(i) for i in out["share-owners"]), [0, 1, 2])

    def test_consist_phase_returns_macs_for_every_other_vd(self):
        ctx = _ctx(self.tmp.name, 0, 4)
        out = secagg_recover_fn(self._q({"round": 3, "dropped": [3], "phase": "consist",
                                         "sig": sign_recover(self.priv, 3, [3], run=1)}), ctx).content["secagg"]
        macs = list(out["macs"])
        self.assertEqual(macs[1], consist_mac(load_agreement_keys(self.tmp.name, 0)[0], self.pubs[1], 0, 1, 3, 1, [3]))
        with self.assertRaises(RuntimeError):  # 每輪只替一份名單做證明
            secagg_recover_fn(self._q({"round": 3, "dropped": [], "phase": "consist",
                                       "sig": sign_recover(self.priv, 3, [], run=1)}), ctx)

    def test_needs_enough_matching_attestations(self):
        # 門檻 2n/3+1 = 3（含自己）：只有 1 份別人的證明、或證明的是另一份名單，都不交
        self.assertEqual(consistency_threshold(4), 3)
        with self.assertRaises(RuntimeError):
            self._ask(0, [3], [0, 1, 2], attesters=[1])
        with self.assertRaises(RuntimeError):
            self._ask(0, [3], [0, 1, 2], forge=True)

    def test_recovery_without_attesting_is_refused(self):
        q = {"round": 3, "dropped": [3], "sig": sign_recover(self.priv, 3, [3], run=1),
             "share-owners": [0, 1, 2], "share-cts": [self.cts[i][0] for i in (0, 1, 2)]}
        with self.assertRaises(RuntimeError):
            secagg_recover_fn(self._q(q), _ctx(self.tmp.name, 0, 4))

    def test_malformed_request_does_not_use_up_the_answer(self):
        # 稽核 LOW：份的擁有者不對就拒絕，但不能因此耗掉這輪唯一一次回答
        ctx = _ctx(self.tmp.name, 0, 4)
        secagg_recover_fn(self._q({"round": 3, "dropped": [3], "phase": "consist",
                                   "sig": sign_recover(self.priv, 3, [3], run=1)}), ctx)
        with self.assertRaises(RuntimeError):
            self._ask(0, [3], [0, 1], ctx=ctx)
        out = self._ask(0, [3], [0, 1, 2], ctx=ctx).content["secagg"]
        self.assertEqual(sorted(int(i) for i in out["share-owners"]), [0, 1, 2])

    def test_owners_must_be_exactly_the_alive_list(self):
        with self.assertRaises(RuntimeError):
            self._ask(0, [3], [0, 1])

    def test_refuses_share_of_a_dropped_vd(self):
        # 惡意伺服器說 3 斷線（拿成對種子），又要 3 的自我遮罩份：拒絕
        with self.assertRaises(RuntimeError):
            self._ask(0, [3], [1, 3])


class ServerReconstructTest(unittest.TestCase):
    """復原時沉默的在線 VD：用其他 VD 交出的份還原它的自我遮罩種子，這輪不作廢。"""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 3)
        provision_keys(self.tmp.name, 3)
        self.spriv, _ = load_server_secrets(self.tmp.name)
        self.pubs = load_agreement_pubs(self.tmp.name)
        self.privs = {i: load_agreement_keys(self.tmp.name, i)[0] for i in range(3)}
        self.seeds = {i: self_seed(self.privs[i], i, 3, 0) for i in range(3)}
        self.cts = {i: encrypt_shares(self.seeds[i], i, range(3), self.privs[i], self.pubs, 3, 0) for i in range(3)}

    def tearDown(self):
        self.tmp.cleanup()

    def _guard(self, silent, alive):
        tmp, cts = self.tmp.name, self.cts

        ctxs = {}
        for pid in range(3):
            ctxs[pid] = _ctx(tmp, pid, 3)
            ctxs[pid].run_id = 0

        class Grid:
            def send_and_receive(self, messages, timeout=None):
                out = []
                for m in messages:
                    pid = m.metadata.dst_node_id - 100
                    if pid in silent and "phase" not in m.content["secagg"]:
                        continue  # 復原時才沉默（一致性回合有回）
                    try:
                        out.append(secagg_recover_fn(m, ctxs[pid]))
                    except RuntimeError as e:
                        out.append(Message(m.content, reply_to=m) if False else _err(m, e))
                return out
        g = GuardedStrategy(object(), self.spriv, None, secure_agg_clients=3, secagg_self_mask=True,
                            secagg_keys=(load_server_agreement_key(tmp), self.pubs), secagg_shamir=True)
        g._grid = Grid()
        g._shares = {i: {j: cts[i][j] for j in range(3)} for i in alive}
        g._self_hash = {i: seed_tag(self.seeds[i]) for i in alive}
        return g

    def _kept(self, pid):
        sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
        return Message(content=RecordDict({"metrics": MetricRecord({"partition-id": pid})}), reply_to=sent)

    def test_silent_vd_self_seed_rebuilt_from_shares(self):
        g = self._guard(silent={2}, alive=[0, 1, 2])
        rev = g._recover(3, [self._kept(i) for i in range(3)])
        self.assertIsNotNone(rev)
        self.assertEqual(rev[2], {2: self.seeds[2].hex()})
        self.assertEqual(rev[0][0], self.seeds[0].hex())

    def test_wrong_share_is_caught_by_seed_hash(self):
        g = self._guard(silent={2}, alive=[0, 1, 2])
        g._self_hash[2] ^= 1  # 等同有人交了錯的份：還原出的種子和上傳時承諾的雜湊對不上
        self.assertIsNone(g._recover(3, [self._kept(i) for i in range(3)]))

    def test_too_few_responders_gives_up(self):
        g = self._guard(silent={1, 2}, alive=[0, 1, 2])
        self.assertIsNone(g._recover(3, [self._kept(i) for i in range(3)]))



class MaskRoundTest(unittest.TestCase):
    """同一輪不能重新遮罩：否則惡意伺服器重送訓練請求就能重設「每輪只回答一次復原」。"""
    def test_second_train_for_same_or_older_round_is_refused(self):
        ctx = Context(run_id=1, node_id=1, node_config={}, state=RecordDict(), run_config={})
        claim_mask_round(ctx, 3)
        ctx.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": 3, "revealed-round": 3})
        for rnd in (3, 2):
            with self.subTest(rnd=rnd), self.assertRaises(RuntimeError):
                claim_mask_round(ctx, rnd)
        claim_mask_round(ctx, 4)
        self.assertEqual(int(ctx.state[SECAGG_STATE_KEY]["revealed-round"]), -1)



class InconsistentListsTest(unittest.TestCase):
    """惡意伺服器對不同 VD 送不同斷線名單（想同時拿到 X 的份和 X 的所有成對種子）：湊不滿相同名單的證明，沒人交出東西。"""
    def test_split_lists_get_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            n = 6  # n=3 的版本（{0:[1],1:[],2:[0]}）已在一致性回合就被在線門檻擋掉，見 LivenessTest
            provision_agreement_keys(d, n)
            provision_keys(d, n)
            spriv, _ = load_server_secrets(d)
            lists = {p: [] for p in range(n)}
            lists[0], lists[1] = [1], [0]  # 對 0 說 1 斷線、對 1 說 0 斷線，對其他人說沒人斷線
            ctxs = {p: _ctx(d, p, n) for p in range(n)}
            macs = {}
            for p, dl in lists.items():
                q = {"round": 3, "dropped": dl, "sig": sign_recover(spriv, 3, dl, run=1), "phase": "consist"}
                out = secagg_recover_fn(Message(content=RecordDict({"secagg": ConfigRecord(q)}), dst_node_id=1,
                                                message_type="query"), ctxs[p]).content["secagg"]
                macs[p] = list(out["macs"])
            for p, dl in lists.items():
                others = [i for i in range(n) if i != p and i not in dl]
                q = {"round": 3, "dropped": dl, "sig": sign_recover(spriv, 3, dl, run=1),
                     "share-owners": [i for i in range(n) if i not in dl], "share-cts": ["00"] * (n - len(dl)),
                     "attest-from": others, "attest-macs": [macs[i][p] for i in others]}
                with self.subTest(vd=p), self.assertRaisesRegex(RuntimeError, "一致的證明"):
                    secagg_recover_fn(Message(content=RecordDict({"secagg": ConfigRecord(q)}), dst_node_id=1,
                                              message_type="query"), ctxs[p])



class LivenessTest(unittest.TestCase):
    """稽核 MEDIUM：在線 VD 少於一致性門檻時，復原一定湊不滿證明，一致性回合就要直接拒絕（n=3 有人斷線就做不到）。"""
    def test_consist_refused_when_alive_below_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            provision_agreement_keys(d, 3)
            provision_keys(d, 3)
            spriv, _ = load_server_secrets(d)
            q = {"round": 3, "dropped": [1], "sig": sign_recover(spriv, 3, [1], run=1), "phase": "consist"}
            with self.assertRaises(RuntimeError):
                secagg_recover_fn(Message(content=RecordDict({"secagg": ConfigRecord(q)}), dst_node_id=1,
                                          message_type="query"), _ctx(d, 0, 3))


if __name__ == "__main__":
    unittest.main()
