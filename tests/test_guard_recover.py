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
            g._recover(1, [_kept(0), _kept(1)])
            q = sent[0].content["secagg"]
            self.assertTrue(verify_recover(load_server_public_key(d), 1, [2], str(q["sig"])))


if __name__ == "__main__":
    unittest.main()
