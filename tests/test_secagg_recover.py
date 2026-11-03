"""client_app.secagg_recover_fn：VD 只在合理的情況下交出斷線復原用的成對種子。"""
import tempfile
import unittest
from pathlib import Path

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict

import client_app
from client_app import SECAGG_STATE_KEY, secagg_recover_fn
from secagg import provision_agreement_keys


class RecoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 6)
        self.ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 0, "num-partitions": 6},
                           state=RecordDict(), run_config={"key-dir": self.tmp.name})
        self.ctx.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": 3, "revealed-round": -1})

    def tearDown(self):
        self.tmp.cleanup()

    def _ask(self, rnd, dropped):
        q = RecordDict({"secagg": ConfigRecord({"round": rnd, "dropped": dropped})})
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


if __name__ == "__main__":
    unittest.main()
