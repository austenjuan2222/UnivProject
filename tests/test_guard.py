"""guard.py：伺服器端檢查 VD 上傳的更新（簽章驗證、模擬中間人、剔除斷線 client）。"""
import os
import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, MetricRecord, RecordDict

from guard import arrays_of, attach_update_tag, screen_replies

KEYS = {p: os.urandom(32) for p in range(4)}  # 伺服器端的驗證表：VD 編號 → 金鑰


def _content(pid, rnd=3, v=0.5, tag=True, key=None):
    arrays = {"w": np.full((2, 2), v, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    c = RecordDict({"arrays": ArrayRecord({k: Array(a) for k, a in arrays.items()}),
                    "metrics": MetricRecord({"partition-id": pid, "num-examples": 10, "round": rnd})})
    if tag:
        attach_update_tag(c, key or KEYS[pid], rnd, pid)
    return c


class ScreenRepliesTest(unittest.TestCase):
    def test_valid_updates_are_kept_and_auth_record_removed(self):
        cs = [_content(0), _content(1)]
        keep, stats = screen_replies(cs, 3, KEYS, verify=True, tamper_rate=0.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True, True])
        self.assertEqual(stats["n_rejected"], 0)
        self.assertNotIn("auth", cs[0])  # 內層策略只看到一個 ArrayRecord 與一個 MetricRecord

    def test_forged_and_missing_tags_are_rejected(self):
        cs = [_content(0), _content(1, key=os.urandom(32)), _content(2, tag=False)]
        keep, stats = screen_replies(cs, 3, KEYS, verify=True, tamper_rate=0.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True, False, False])
        self.assertEqual(stats["n_rejected"], 2)

    def test_audit_entries_record_who_sent_what(self):
        cs = [_content(0), _content(1, key=os.urandom(32))]
        _, stats = screen_replies(cs, 3, KEYS, verify=True, tamper_rate=0.0, rng=np.random.default_rng(0))
        audit = stats["audit"]
        self.assertEqual([(a["round"], a["pid"], a["accepted"]) for a in audit], [(3, 0, True), (3, 1, False)])
        self.assertEqual(len(audit[0]["digest"]), 64)  # 更新內容的 SHA-256
        self.assertTrue(audit[0]["tag"])

    def test_unknown_vd_is_rejected(self):
        c = _content(0)
        c["metrics"]["partition-id"] = 9  # 不在驗證表裡的 VD 編號
        keep, _ = screen_replies([c], 3, KEYS, verify=True, tamper_rate=0.0, rng=np.random.default_rng(0))
        self.assertEqual(keep, [False])

    def test_replay_from_previous_round_is_rejected(self):
        keep, _ = screen_replies([_content(0, rnd=2)], 3, KEYS, verify=True, tamper_rate=0.0,
                                 rng=np.random.default_rng(0))
        self.assertEqual(keep, [False])

    def _replay_two_rounds(self, verify):
        last = {}
        screen_replies([_content(0, rnd=2, v=0.1)], 2, KEYS, verify=verify, tamper_rate=0.0,
                       rng=np.random.default_rng(0), replay_rate=1.0, last=last)
        c = _content(0, rnd=3, v=0.5)
        keep, stats = screen_replies([c], 3, KEYS, verify=verify, tamper_rate=0.0,
                                     rng=np.random.default_rng(0), replay_rate=1.0, last=last)
        return c, keep, stats

    def test_replayed_whole_message_rejected_when_verifying(self):
        _, keep, stats = self._replay_two_rounds(verify=True)
        self.assertEqual(stats["n_replayed"], 1)
        self.assertEqual(keep, [False])

    def test_replayed_whole_message_accepted_without_verification(self):
        c, keep, stats = self._replay_two_rounds(verify=False)
        self.assertEqual((keep, stats["n_replayed"]), ([True], 1))
        self.assertTrue(np.allclose(arrays_of(c["arrays"])["w"], 0.1))  # 上一輪的舊更新進入聚合

    def test_tampered_in_transit_rejected_when_verifying(self):
        cs = [_content(i) for i in range(4)]
        keep, stats = screen_replies(cs, 3, KEYS, verify=True, tamper_rate=1.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(stats["n_tampered"], 4)
        self.assertEqual(keep, [False] * 4)

    def test_tampered_in_transit_accepted_without_verification(self):
        cs = [_content(0)]
        keep, stats = screen_replies(cs, 3, KEYS, verify=False, tamper_rate=1.0,
                                     rng=np.random.default_rng(0))
        self.assertEqual(keep, [True])
        self.assertEqual(stats["n_tampered"], 1)
        self.assertFalse(np.allclose(arrays_of(cs[0]["arrays"])["w"], 0.5))  # 竄改真的進入聚合


if __name__ == "__main__":
    unittest.main()


class CollusionFilterTest(unittest.TestCase):
    """跨輪偵測串通：一群 VD 每輪更新方向都特別像，累積起來就被標記。"""

    def _round(self, rng, n=32, k=6, d=400):
        g = rng.normal(size=d).astype(np.float32)
        shared = rng.choice([-1.0, 1.0], size=d)
        ups = {}
        for p in range(n):
            u = 0.3 * g + rng.normal(size=d)  # 誠實 VD：有共同趨勢，加上各自的雜訊
            ups[p] = u + np.abs(u) * shared if p >= n - k else u
        return ups

    def test_colluders_are_flagged_after_a_few_rounds(self):
        from guard import CollusionFilter
        rng, f = np.random.default_rng(0), CollusionFilter()
        for _ in range(3):
            flagged = f.update(self._round(rng))
        self.assertEqual(sorted(flagged), list(range(26, 32)))

    def test_no_attack_flags_nobody(self):
        from guard import CollusionFilter
        rng, f = np.random.default_rng(1), CollusionFilter()
        for _ in range(5):
            flagged = f.update(self._round(rng, k=0))
        self.assertEqual(flagged, set())

    def test_malformed_or_nonfinite_update_is_rejected_not_raised(self):
        from guard import update_delta
        g = {"w": np.zeros((2, 2), dtype=np.float32)}
        self.assertIsNone(update_delta({"w": np.zeros(3, dtype=np.float32)}, g))
        self.assertIsNone(update_delta({"x": np.zeros((2, 2), dtype=np.float32)}, g))
        self.assertIsNone(update_delta({"w": np.full((2, 2), np.nan, dtype=np.float32)}, g))
        self.assertEqual(update_delta({"w": np.ones((2, 2), dtype=np.float32)}, g).shape, (4,))
