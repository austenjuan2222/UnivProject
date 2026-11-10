"""跨次執行的重送（講義攻擊類型：重送）：伺服器簽的全域模型、VD 簽的更新都要綁這次執行的編號。"""
import os
import types
import unittest

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from guard import AUTH, GuardedStrategy, arrays_of, attach_update_tag, screen_replies
from security import verify_model


class Inner:
    def configure_train(self, rnd, arrays, config, grid):
        return [Message(content=RecordDict({"arrays": arrays}), dst_node_id=1, message_type="train")]

    configure_evaluate = configure_train


def grid(run_id):
    return types.SimpleNamespace(run=types.SimpleNamespace(run_id=run_id))


def rec():
    return ArrayRecord({"w": Array(np.arange(4, dtype=np.float32))})


class RunBindingTest(unittest.TestCase):
    def test_server_signs_with_this_runs_id(self):
        priv = Ed25519PrivateKey.generate()
        g = GuardedStrategy(Inner(), priv, None)
        for f in (g.configure_train, g.configure_evaluate):
            m = list(f(3, rec(), ConfigRecord({}), grid(77)))[0]
            sig, arrays = str(m.content[AUTH]["sig"]), arrays_of(m.content["arrays"])
            self.assertTrue(verify_model(priv.public_key(), arrays, 3, sig, run=77))
            self.assertFalse(verify_model(priv.public_key(), arrays, 3, sig, run=76))

    def test_update_signed_in_another_run_is_rejected(self):
        key = os.urandom(32)

        def reply(run):
            c = RecordDict({"arrays": rec(), "metrics": MetricRecord({"partition-id": 2, "num-examples": 10})})
            attach_update_tag(c, key, 5, 2, run)
            return c
        rng = np.random.default_rng(0)
        keep, _ = screen_replies([reply(41), reply(42)], 5, {2: key}, True, 0.0, rng, run=42)
        self.assertEqual(keep, [False, True])


if __name__ == "__main__":
    unittest.main()
