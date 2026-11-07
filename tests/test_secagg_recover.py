"""client_app.secagg_recover_fn：VD 只在合理的情況下交出斷線復原用的成對種子。"""
import tempfile
import unittest
from pathlib import Path

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict

import client_app
from client_app import SECAGG_STATE_KEY, secagg_recover_fn
from secagg import provision_agreement_keys
from security import load_server_secrets, provision_keys, sign_recover


class RecoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 6)
        provision_keys(self.tmp.name, 6)
        self.priv, _ = load_server_secrets(self.tmp.name)
        self.ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 0, "num-partitions": 6},
                           state=RecordDict(), run_config={"key-dir": self.tmp.name})
        self.ctx.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": 3, "revealed-round": -1})

    def tearDown(self):
        self.tmp.cleanup()

    def _ask(self, rnd, dropped):
        q = RecordDict({"secagg": ConfigRecord({"round": rnd, "dropped": dropped,
                                                "sig": sign_recover(self.priv, rnd, dropped, run=1)})})
        return secagg_recover_fn(Message(content=q, dst_node_id=1, message_type="query"), self.ctx)

    def test_reveals_once_for_current_round(self):
        out = self._ask(3, [5]).content["secagg"]
        self.assertEqual(list(out["dropped"]), [5])
        with self.assertRaises(RuntimeError):
            self._ask(3, [5])  # 同一輪第二次（換名單）不回答

    def test_rejects_suspicious_requests(self):
        for rnd, dropped in [(2, [5]), (3, [0]), (3, [3, 4, 5]), (3, [9])]:  # 舊輪、自己、太多台、不存在
            with self.subTest(rnd=rnd, dropped=dropped), self.assertRaises(RuntimeError):
                self._ask(rnd, dropped)


class SignedRecoverTest(unittest.TestCase):
    """斷線復原請求一律要有伺服器簽章（不論 sign-models），中間人不能假造「某台斷線」來解開它的遮罩。"""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 6)
        provision_keys(self.tmp.name, 6)
        self.priv, _ = load_server_secrets(self.tmp.name)
        self.ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 0, "num-partitions": 6},
                           state=RecordDict(), run_config={"key-dir": self.tmp.name})
        self.ctx.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": 3, "revealed-round": -1})

    def tearDown(self):
        self.tmp.cleanup()

    def _ask(self, rnd, dropped, sig=None):
        c = {"round": rnd, "dropped": dropped}
        if sig is not None:
            c["sig"] = sig
        q = RecordDict({"secagg": ConfigRecord(c)})
        return secagg_recover_fn(Message(content=q, dst_node_id=1, message_type="query"), self.ctx)

    def test_signed_request_is_answered(self):
        from secagg import load_agreement_keys, reveal_seeds
        out = self._ask(3, [5], sign_recover(self.priv, 3, [5], run=1)).content["secagg"]
        self.assertEqual(list(out["dropped"]), [5])
        p0, pubs = load_agreement_keys(self.tmp.name, 0)
        from secagg import load_server_agreement_key, open_seeds
        plain = reveal_seeds(0, [5], p0, pubs, 3, run=1)
        self.assertNotEqual(list(out["seeds"]), list(plain.values()))  # 不是明文
        self.assertEqual(open_seeds(dict(zip([5], out["seeds"])), 0, load_server_agreement_key(self.tmp.name),
                                    pubs[0], 3, run=1), plain)

    def test_unsigned_or_forged_request_is_refused(self):
        for sig in [None, "", "00" * 64, sign_recover(self.priv, 3, [4], run=1), sign_recover(self.priv, 2, [5], run=1),
                    sign_recover(self.priv, 3, [5], run=2)]:  # 別次執行的合法簽章也不能重送
            with self.subTest(sig=sig), self.assertRaises(RuntimeError):
                self._ask(3, [5], sig)
        # 被拒絕的假請求不能用掉這一輪唯一一次的回答機會
        self.assertEqual(list(self._ask(3, [5], sign_recover(self.priv, 3, [5], run=1)).content["secagg"]["dropped"]), [5])


if __name__ == "__main__":
    unittest.main()
