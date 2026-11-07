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
        m = unmask_sum([self._masked(0, _update(0.5), 10)], 10)  # 只有一份時「解碼」出來的值
        self.assertGreater(np.abs(m["w"] - 0.5).min(), 1.0)  # 每個值都被大幅遮住

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

    def _self_masked(self, pid, update, n, rnd=1):
        priv, pubs = load_agreement_keys(self.dir, pid)
        return mask_update(update, n, pid, self.pids, priv, pubs, rnd, self_mask=True)

    def test_self_masks_removed_with_revealed_self_seeds(self):
        from secagg import self_seed
        ups = {p: _update(0.1 * (p + 1)) for p in self.pids}
        alive = self.pids[:3]  # VD 3 斷線
        masked = [self._self_masked(p, ups[p], 10) for p in alive]
        revealed = {}
        for p in alive:
            priv, pubs = load_agreement_keys(self.dir, p)
            revealed[p] = {**reveal_seeds(p, [3], priv, pubs, 1), p: self_seed(priv, p, 1).hex()}
        avg = unmask_sum(masked, 30, drop_correction(revealed, masked[0]))
        for k in SHAPES:
            np.testing.assert_allclose(avg[k], sum(ups[p][k].astype(np.float64) for p in alive) / 3, rtol=0, atol=1e-6)

    def test_late_update_stays_hidden_after_pair_seeds_revealed(self):
        # 雙重遮罩：伺服器宣布 VD 3 斷線、拿到所有和它配對的種子後，VD 3 的更新才遲到送達，仍然解不開
        late = self._self_masked(3, _update(0.5), 10)
        revealed = {}
        for p in self.pids[:3]:
            priv, pubs = load_agreement_keys(self.dir, p)
            revealed[p] = reveal_seeds(p, [3], priv, pubs, 1)
        pair_part = drop_correction(revealed, late)  # 在線 VD 為 3 加的遮罩；3 自己加的是相反號
        guess = unmask_sum([{k: late[k] + pair_part[k] for k in late}], 10)  # 扣掉 3 的成對遮罩
        self.assertGreater(np.abs(guess["w"] - 0.5).min(), 1.0)

    def test_fixed_point_upload_is_uint32_and_cancels_exactly(self):
        # 有限體（mod 2^32）上做整數遮罩：上傳和 float32 一樣大（float64 版本是兩倍），抵銷沒有浮點誤差
        ups = {p: _update(0.1 * (p + 1)) for p in self.pids}
        masked = [self._masked(p, ups[p], 10) for p in self.pids]
        self.assertTrue(all(m[k].dtype == np.uint32 for m in masked for k in m))
        self.assertEqual(sum(v.nbytes for v in masked[0].values()), sum(v.nbytes for v in ups[0].values()))
        avg = unmask_sum(masked, 40)
        for k in SHAPES:
            np.testing.assert_allclose(avg[k], sum(ups[p][k].astype(np.float64) for p in self.pids) / 4,
                                       rtol=0, atol=2 ** -12)

    def test_out_of_range_values_are_clipped_and_counted(self):
        from secagg import count_clipped
        big = {"w": np.full((3, 2), 1e6, dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}
        self.assertEqual(count_clipped(big, 10, 4), 6)
        self.assertEqual(count_clipped(_update(0.3), 10, 4), 0)
        m = [self._masked(p, big if p == 0 else _update(0.0), 10) for p in self.pids]
        avg = unmask_sum(m, 40)
        self.assertTrue(np.all(np.isfinite(avg["w"])) and np.all(avg["w"] > 0))  # 飽和，不會溢位成負數

    def test_16bit_ring_halves_upload_and_cancels_with_dropout(self):
        # LPWAN：Z_{2^16}，上傳是 uint16（float32 的一半）；小數位數可設，斷線復原照樣成立
        ups = {p: _update(0.1 * (p + 1)) for p in self.pids}
        masked = []
        for p in (0, 1, 2):
            priv, pubs = load_agreement_keys(self.dir, p)
            masked.append(mask_update(ups[p], 10, p, self.pids, priv, pubs, 1, bits=16, frac_bits=6, self_mask=True))
        self.assertTrue(all(v.dtype == np.uint16 for m in masked for v in m.values()))
        from secagg import self_seed
        revealed = {}
        for p in (0, 1, 2):
            priv, pubs = load_agreement_keys(self.dir, p)
            revealed[p] = {**reveal_seeds(p, [3], priv, pubs, 1), p: self_seed(priv, p, 1).hex()}
        avg = unmask_sum(masked, 30, drop_correction(revealed, masked[0]), frac_bits=6)
        expect = {k: sum(10 * ups[p][k].astype(np.float64) for p in (0, 1, 2)) / 30 for k in SHAPES}
        for k in SHAPES:
            np.testing.assert_allclose(avg[k], expect[k], atol=3 * 2 ** -6 / 30 + 1e-9)

    def test_16bit_clip_limit_uses_16bit_range(self):
        from secagg import count_clipped
        up = {"w": np.array([0.5, 2.0], dtype=np.float32)}  # n=1、4 台、小數 12 位：上限 32767//4/4096 ≈ 2.0
        self.assertEqual(count_clipped(up, 1, 4, bits=16, frac_bits=12), 1)
        self.assertEqual(count_clipped(up, 1, 4), 0)

    def test_mixed_ring_widths_rejected(self):
        with self.assertRaises(ValueError):
            unmask_sum([{"w": np.zeros(2, np.uint32)}, {"w": np.zeros(2, np.uint16)}], 1.0)

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
        # 復原用的種子加密給伺服器：沒有 TLS 時中間人攔下更新、再讀回覆也拿不到種子；改過的密文要被發現
        from secagg import load_agreement_pubs, load_server_agreement_key, open_seeds, seal_seeds
        p0, pubs = load_agreement_keys(self.dir, 0)
        spub = load_agreement_pubs(self.dir)["server"]
        spriv = load_server_agreement_key(self.dir)
        plain = reveal_seeds(0, [3], p0, pubs, 2, run=9)
        sealed = seal_seeds(plain, 0, p0, spub, 2, run=9)
        self.assertNotEqual(sealed, plain)
        self.assertEqual(open_seeds(sealed, 0, spriv, pubs[0], 2, run=9), plain)
        self.assertIsNone(open_seeds(sealed, 0, spriv, pubs[0], 3, run=9))  # 別輪
        bad = {3: ("0" if sealed[3][0] != "0" else "1") + sealed[3][1:]}
        self.assertIsNone(open_seeds(bad, 0, spriv, pubs[0], 2, run=9))  # 被改過
        self.assertIsNone(open_seeds({3: "zz"}, 0, spriv, pubs[0], 2, run=9))  # 格式錯


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

    def test_base_is_added_back(self):
        # VD 上傳的是相對於全域模型的差值，伺服器還原後加回全域模型
        from guard import arrays_of, secure_average
        base = {"w": np.full((3, 2), 2.0, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
        a = arrays_of(secure_average(self._contents([0, 1, 2]), expected=3, base=base))
        np.testing.assert_allclose(a["w"], 2.3, atol=1e-3)

    def test_missing_vd_aborts_the_round(self):
        from guard import secure_average
        self.assertIsNone(secure_average(self._contents([0, 1]), expected=3))



class IntegerOnlyTest(unittest.TestCase):
    def test_unmask_sum_rejects_non_uint32(self):
        import numpy as np
        from secagg import unmask_sum
        with self.assertRaises(ValueError):
            unmask_sum([{"w": np.zeros(2, np.uint32)}, {"w": np.zeros(2, np.float32)}], 1.0)

    def test_tamper_keeps_uint32_and_moves_by_lsb_scale(self):
        import numpy as np
        from security import tamper
        a = {"w": np.zeros(1000, np.uint32)}
        out = tamper(a, magnitude=1.0, rng=np.random.default_rng(0))["w"]
        self.assertEqual(out.dtype, np.uint32)
        # magnitude 是實際數值的單位：定點數要放大 2^12 倍才算數
        self.assertGreater(np.abs(out.view(np.int32)).mean(), 1000)

    def test_mask_is_shake256_stream(self):
        import hashlib, numpy as np
        from secagg import _mask
        seed = bytes(range(32))
        want = np.frombuffer(hashlib.shake_256(seed).digest(4 * 6), "<u4").reshape(2, 3)
        np.testing.assert_array_equal(_mask(seed, (2, 3)), want)


if __name__ == "__main__":
    unittest.main()
