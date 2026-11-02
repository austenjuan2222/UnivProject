"""guard.py：伺服器端檢查 VD 上傳的更新（簽章驗證、模擬中間人、剔除斷線 client）。"""
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from guard import arrays_of, attach_update_tag, screen_replies
from security import vd_key

SECRET = "lab-secret"


def _content(pid, rnd=3, v=0.5, tag=True, secret=SECRET):
    arrays = {"w": np.full((2, 2), v, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    c = RecordDict({"arrays": ArrayRecord({k: Array(a) for k, a in arrays.items()}),
                    "metrics": MetricRecord({"partition-id": pid, "num-examples": 10, "round": rnd})})
    if tag:
        attach_update_tag(c, vd_key(secret, pid), rnd, pid)
    return c


class ScreenRepliesTest(unittest.TestCase):
    def test_valid_updates_are_kept_and_auth_record_removed(self):
        cs = [_content(0), _content(1)]
        keep, stats = screen_replies(cs, 3, SECRET, verify=True, tamper_rate=0.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True, True])
        self.assertEqual(stats["n_rejected"], 0)
        self.assertNotIn("auth", cs[0])  # 內層策略只看到一個 ArrayRecord 與一個 MetricRecord

    def test_forged_and_missing_tags_are_rejected(self):
        cs = [_content(0), _content(1, secret="attacker"), _content(2, tag=False)]
        keep, stats = screen_replies(cs, 3, SECRET, verify=True, tamper_rate=0.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True, False, False])
        self.assertEqual(stats["n_rejected"], 2)

    def test_replay_from_previous_round_is_rejected(self):
        keep, _ = screen_replies([_content(0, rnd=2)], 3, SECRET, verify=True, tamper_rate=0.0,
                                 rng=np.random.default_rng(0))
        self.assertEqual(keep, [False])

    def test_tampered_in_transit_rejected_when_verifying(self):
        cs = [_content(i) for i in range(4)]
        keep, stats = screen_replies(cs, 3, SECRET, verify=True, tamper_rate=1.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(stats["n_tampered"], 4)
        self.assertEqual(keep, [False] * 4)

    def test_tampered_in_transit_accepted_without_verification(self):
        cs = [_content(0)]
        keep, stats = screen_replies(cs, 3, SECRET, verify=False, tamper_rate=1.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True])
        self.assertEqual(stats["n_tampered"], 1)
        self.assertFalse(np.allclose(arrays_of(cs[0]["arrays"])["w"], 0.5))  # 竄改真的進入聚合


if __name__ == "__main__":
    unittest.main()
