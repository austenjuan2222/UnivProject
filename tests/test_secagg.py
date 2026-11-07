"""secagg.py：成對遮罩的安全聚合（伺服器只看得到加總，看不到單一 VD 的更新）。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from secagg import drop_correction, load_agreement_keys, mask_update, provision_agreement_keys, reveal_seeds, unmask_sum

SHAPES = {"w": (3, 2), "b": (3,)}


def _update(v):
    return {"w": np.full((3, 2), v, dtype=np.float32), "b": np.arange(3, dtype=np.float32) * v}


class SecAggTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        provision_agreement_keys(self.dir, 4)
        self.pids = [0, 1, 2, 3]

    def tearDown(self):
        self.tmp.cleanup()

    def _masked(self, pid, update, n, rnd=1):
        priv, pubs = load_agreement_keys(self.dir, pid)
        return mask_update(update, n, pid, self.pids, priv, pubs, rnd)

    def test_masks_cancel_in_the_weighted_sum(self):
        ups = {p: _update(0.1 * (p + 1)) for p in self.pids}
        ns = {0: 10, 1: 20, 2: 30, 3: 40}
        masked = [self._masked(p, ups[p], ns[p]) for p in self.pids]
        avg = unmask_sum(masked, sum(ns.values()))
        expect = {k: sum(ns[p] * ups[p][k].astype(np.float64) for p in self.pids) / 100 for k in SHAPES}
        for k in SHAPES:
            np.testing.assert_allclose(avg[k], expect[k], rtol=0, atol=1e-6)

    def test_single_masked_update_hides_the_values(self):
        m = self._masked(0, _update(0.5), 10)
        self.assertGreater(np.abs(m["w"] - 10 * 0.5).min(), 1.0)  # 每個值都被大幅遮住

    def test_masks_change_every_round(self):
        a = self._masked(1, _update(0.5), 10, rnd=1)
        b = self._masked(1, _update(0.5), 10, rnd=2)
        self.assertFalse(np.allclose(a["w"], b["w"]))

    def test_missing_vd_leaves_masks_uncancelled(self):
        masked = [self._masked(p, _update(0.2), 10) for p in self.pids[:3]]  # VD 3 斷線
        avg = unmask_sum(masked, 30)
        self.assertGreater(np.abs(avg["w"] - 0.2).max(), 0.5)  # 基本版不支援斷線（需 SecAgg+ 的秘密分享）

    def test_survivors_reveal_seeds_so_dropped_vd_masks_cancel(self):
        ups = {p: _update(0.1 * (p + 1)) for p in self.pids}
        alive = self.pids[:3]  # VD 3 遮罩後斷線
        masked = [self._masked(p, ups[p], 10) for p in alive]
        revealed = {}
        for p in alive:
            priv, pubs = load_agreement_keys(self.dir, p)
            revealed[p] = reveal_seeds(p, [3], priv, pubs, 1)
        avg = unmask_sum(masked, 30, drop_correction(revealed, masked[0]))
        for k in SHAPES:
            expect = sum(ups[p][k].astype(np.float64) for p in alive) / 3
            np.testing.assert_allclose(avg[k], expect, rtol=0, atol=1e-6)

    def test_revealed_seeds_only_cover_dropped_pairs(self):
        priv, pubs = load_agreement_keys(self.dir, 0)
        self.assertEqual(set(reveal_seeds(0, [2, 3], priv, pubs, 1)), {2, 3})
        self.assertEqual(reveal_seeds(0, [0], priv, pubs, 1), {})  # 不會交出自己

    def test_pair_seed_is_symmetric(self):
        from secagg import pair_seed
        p0, pubs = load_agreement_keys(self.dir, 0)
        p1, _ = load_agreement_keys(self.dir, 1)
        self.assertEqual(pair_seed(p0, pubs[1], 0, 1, 5), pair_seed(p1, pubs[0], 1, 0, 5))

    def test_pair_seed_depends_on_run(self):
        # 金鑰跨次執行保留：種子不綁執行編號的話，上一次執行交出的種子能解開這次同一輪的遮罩
        from secagg import pair_seed
        p0, pubs = load_agreement_keys(self.dir, 0)
        self.assertNotEqual(pair_seed(p0, pubs[1], 0, 1, 5, run=1), pair_seed(p0, pubs[1], 0, 1, 5, run=2))

    def test_sealed_seeds_open_only_on_server(self):
        # 復原用的種子加密給伺服器：沒有 TLS 時中間人攔下更新、再讀回覆也拿不到種子
        from secagg import load_server_agreement_key, seal_seeds
        p0, pubs = load_agreement_keys(self.dir, 0)
        spriv = load_server_agreement_key(self.dir)
        plain = reveal_seeds(0, [3], p0, pubs, 2, run=9)
        sealed = seal_seeds(plain, 0, p0, pubs["server"], 2, run=9)
        self.assertNotEqual(sealed, plain)
        self.assertEqual(seal_seeds(sealed, 0, spriv, pubs[0], 2, run=9), plain)
        self.assertNotEqual(seal_seeds(sealed, 0, spriv, pubs[0], 3, run=9), plain)  # 別輪的金鑰流不同


class SecureAverageTest(unittest.TestCase):
    """伺服器端：把遮罩後的 RecordDict 加總成全域模型；少了任何一個 VD 就放棄這一輪。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_agreement_keys(self.tmp.name, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def _contents(self, pids):
        from flwr.app import Array, ArrayRecord, MetricRecord, RecordDict
        out = []
        for p in pids:
            priv, pubs = load_agreement_keys(self.tmp.name, p)
            m = mask_update(_update(0.3), 5, p, [0, 1, 2], priv, pubs, 1)
            out.append(RecordDict({"arrays": ArrayRecord({k: Array(v) for k, v in m.items()}),
                                   "metrics": MetricRecord({"num-examples": 5, "partition-id": p})}))
        return out

    def test_full_participation_gives_float32_average_in_model_key_order(self):
        from guard import arrays_of, secure_average
        rec = secure_average(self._contents([0, 1, 2]), expected=3)
        self.assertEqual(list(rec.keys()), ["w", "b"])
        a = arrays_of(rec)
        self.assertEqual(a["w"].dtype, np.float32)
        np.testing.assert_allclose(a["w"], 0.3, atol=1e-5)

    def test_missing_vd_aborts_the_round(self):
        from guard import secure_average
        self.assertIsNone(secure_average(self._contents([0, 1]), expected=3))


if __name__ == "__main__":
    unittest.main()
