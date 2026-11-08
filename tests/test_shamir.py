"""Shamir 秘密分享自我遮罩種子（SecAgg 的做法）：VD 自己不直接交出種子，伺服器要湊滿門檻份數才能還原。"""
import tempfile
import unittest

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict

from client_app import SECAGG_STATE_KEY, secagg_recover_fn
from guard import GuardedStrategy
from client_app import claim_mask_round
from secagg import (seed_tag, encrypt_shares, load_agreement_keys, load_agreement_pubs, load_server_agreement_key,
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

    def _ask(self, pid, dropped, owners):
        q = {"round": 3, "dropped": dropped, "sig": sign_recover(self.priv, 3, dropped, run=1),
             "share-owners": list(owners), "share-cts": [self.cts[i][pid] for i in owners]}
        msg = Message(content=RecordDict({"secagg": ConfigRecord(q)}), dst_node_id=1, message_type="query")
        return secagg_recover_fn(msg, _ctx(self.tmp.name, pid, 4))

    def test_gives_shares_for_alive_owners_and_no_plain_self_seed(self):
        out = self._ask(0, [3], [0, 1, 2]).content["secagg"]
        self.assertNotIn("self", out)
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

        class Grid:
            def send_and_receive(self, messages, timeout=None):
                out = []
                for m in messages:
                    pid = m.metadata.dst_node_id - 100
                    if pid in silent:
                        continue
                    ctx = _ctx(tmp, pid, 3)
                    ctx.run_id = 0
                    out.append(secagg_recover_fn(m, ctx))
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


if __name__ == "__main__":
    unittest.main()
