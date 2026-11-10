"""security.py：模型更新的完整性（HMAC）、全域模型簽章（Ed25519）、攻擊模擬。"""
import unittest

import numpy as np

import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from security import (drops_out, is_attacker, load_server_public_key, load_server_secrets, load_vd_key,
                      load_vd_signing_key, load_vd_verify_keys, poison_update, provision_keys, sign_model,
                      sign_update, tamper, verify_model, verify_update)


def _arrays(v=0.5):
    return {"net.0.weight": np.full((3, 2), v, dtype=np.float32),
            "net.0.bias": np.arange(3, dtype=np.float32)}


class KeyProvisioningTest(unittest.TestCase):
    """每個 VD 只拿到自己的金鑰與伺服器公鑰；伺服器的私鑰與驗證表放在另一個資料夾。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "keys"
        provision_keys(self.dir, 4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_vd_gets_a_distinct_random_key(self):
        keys = [load_vd_key(self.dir, p) for p in range(4)]
        self.assertEqual(len(set(keys)), 4)
        self.assertTrue(all(len(k) == 32 for k in keys))

    def test_vd_files_hold_no_server_secret(self):
        vd_files = sorted(f.name for f in self.dir.iterdir() if f.is_file())
        self.assertEqual(vd_files, ["server.pub", "vd-0.ed25519", "vd-0.key", "vd-1.ed25519", "vd-1.key",
                                    "vd-2.ed25519", "vd-2.key", "vd-3.ed25519", "vd-3.key", "vd_sign_pub.json"])
        priv, table = load_server_secrets(self.dir)
        self.assertEqual(set(table), {0, 1, 2, 3})
        self.assertEqual(table[2], load_vd_key(self.dir, 2))

    def test_server_signature_verifies_with_published_public_key(self):
        priv, _ = load_server_secrets(self.dir)
        sig = sign_model(priv, _arrays(), rnd=1)
        self.assertTrue(verify_model(load_server_public_key(self.dir), _arrays(), 1, sig))

    def test_provisioning_twice_keeps_existing_keys(self):
        before = load_vd_key(self.dir, 1)
        provision_keys(self.dir, 4)
        self.assertEqual(load_vd_key(self.dir, 1), before)

    def test_more_vds_adds_keys_without_changing_old_ones(self):
        before = load_vd_key(self.dir, 0)
        provision_keys(self.dir, 6)
        self.assertEqual(load_vd_key(self.dir, 0), before)
        self.assertEqual(set(load_server_secrets(self.dir)[1]), set(range(6)))

    @unittest.skipIf(os.name == "nt", "POSIX 權限")
    def test_secret_files_are_owner_only(self):
        self.assertEqual((self.dir / "vd-0.key").stat().st_mode & 0o077, 0)
        self.assertEqual((self.dir / "server" / "ed25519.key").stat().st_mode & 0o077, 0)


class UpdateSignatureTest(unittest.TestCase):
    """不可否認性：VD 用自己的 Ed25519 私鑰簽更新，伺服器只有公鑰，偽造不了 VD 的簽章。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        provision_keys(self.tmp.name, 3)
        self.pubs = load_vd_verify_keys(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vd_signature_verifies_with_public_key_only(self):
        tag = sign_update(load_vd_signing_key(self.tmp.name, 1), _arrays(), rnd=2, pid=1)
        self.assertTrue(verify_update(self.pubs[1], _arrays(), 2, 1, tag))

    def test_signature_by_another_vd_is_rejected(self):
        tag = sign_update(load_vd_signing_key(self.tmp.name, 0), _arrays(), rnd=2, pid=1)
        self.assertFalse(verify_update(self.pubs[1], _arrays(), 2, 1, tag))

    def test_server_side_table_holds_only_public_keys(self):
        self.assertTrue(all(isinstance(k, Ed25519PublicKey) for k in self.pubs.values()))

    def test_modified_update_or_round_is_rejected(self):
        tag = sign_update(load_vd_signing_key(self.tmp.name, 2), _arrays(), rnd=2, pid=2)
        self.assertFalse(verify_update(self.pubs[2], _arrays(0.6), 2, 2, tag))
        self.assertFalse(verify_update(self.pubs[2], _arrays(), 3, 2, tag))


class UpdateMacTest(unittest.TestCase):
    def setUp(self):
        self.key = os.urandom(32)

    def test_signed_update_verifies(self):
        tag = sign_update(self.key, _arrays(), rnd=5, pid=3)
        self.assertTrue(verify_update(self.key, _arrays(), 5, 3, tag))

    def test_one_changed_value_is_rejected(self):
        tag = sign_update(self.key, _arrays(), rnd=5, pid=3)
        bad = _arrays()
        bad["net.0.bias"][1] += 1e-3
        self.assertFalse(verify_update(self.key, bad, 5, 3, tag))

    def test_replayed_old_round_is_rejected(self):
        tag = sign_update(self.key, _arrays(), rnd=4, pid=3)
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, tag))

    def test_replayed_from_previous_run_is_rejected(self):
        # 講義攻擊類型（重送）：上一次執行同一輪的更新原封不動送進這次執行，金鑰沒換時以前驗得過
        tag = sign_update(self.key, _arrays(), rnd=5, pid=3, run=11)
        self.assertTrue(verify_update(self.key, _arrays(), 5, 3, tag, run=11))
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, tag, run=12))

    def test_other_vd_key_cannot_forge(self):
        tag = sign_update(os.urandom(32), _arrays(), rnd=5, pid=3)
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, tag))

    def test_garbage_tag_is_rejected_not_raised(self):
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, "zz-not-hex"))


class ModelSignatureTest(unittest.TestCase):
    def setUp(self):
        self.priv = Ed25519PrivateKey.generate()
        self.pub = self.priv.public_key()

    def test_global_model_signature_verifies(self):
        sig = sign_model(self.priv, _arrays(), rnd=2)
        self.assertTrue(verify_model(self.pub, _arrays(), 2, sig))

    def test_fake_model_update_is_rejected(self):
        sig = sign_model(self.priv, _arrays(), rnd=2)
        self.assertFalse(verify_model(self.pub, _arrays(0.6), 2, sig))
        self.assertFalse(verify_model(self.pub, _arrays(), 3, sig))

    def test_model_from_previous_run_is_rejected(self):
        # 上一次執行第 2 輪的全域模型（舊的，或當時被下毒的）不能拿來騙這次執行第 2 輪的 VD
        sig = sign_model(self.priv, _arrays(), rnd=2, run=11)
        self.assertTrue(verify_model(self.pub, _arrays(), 2, sig, run=11))
        self.assertFalse(verify_model(self.pub, _arrays(), 2, sig, run=12))

    def test_signature_from_other_server_is_rejected(self):
        sig = sign_model(Ed25519PrivateKey.generate(), _arrays(), rnd=2)
        self.assertFalse(verify_model(self.pub, _arrays(), 2, sig))


class AttackSimulationTest(unittest.TestCase):
    def test_flip_attack_reverses_and_scales_update(self):
        g, local = _arrays(0.5), _arrays(0.7)
        out = poison_update(g, local, "flip", scale=10.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out["net.0.weight"], 0.5 - 10 * 0.2, rtol=1e-5)
        np.testing.assert_allclose(out["net.0.bias"], g["net.0.bias"])  # 更新量 0 → 不變

    def test_noise_attack_changes_every_array(self):
        g = _arrays()
        out = poison_update(g, _arrays(), "noise", scale=5.0, rng=np.random.default_rng(0))
        for k in g:
            self.assertGreater(np.abs(out[k] - g[k]).max(), 0.1)

    def test_collude_attackers_shift_the_same_way_within_honest_scale(self):
        g = _arrays(0.5)
        a = poison_update(g, _arrays(0.7), "collude", scale=1.0, rng=np.random.default_rng([1, 3]))
        b = poison_update(g, _arrays(0.6), "collude", scale=1.0, rng=np.random.default_rng([1, 3]))
        k = "net.0.weight"
        np.testing.assert_array_equal(np.sign(a[k] - 0.7 + 1e-9), np.sign(b[k] - 0.6 + 1e-9))
        self.assertLessEqual(np.abs(a[k] - g[k]).max(), 2 * 0.2 + 1e-5)  # 不超過自己更新量的 (1+scale) 倍

    def test_unknown_attack_raises(self):
        with self.assertRaises(ValueError):
            poison_update(_arrays(), _arrays(), "nope", 1.0, np.random.default_rng(0))

    def test_tamper_modifies_without_touching_input(self):
        a = _arrays()
        out = tamper(a, magnitude=1.0, rng=np.random.default_rng(0))
        self.assertFalse(np.allclose(out["net.0.weight"], a["net.0.weight"]))
        np.testing.assert_allclose(a["net.0.weight"], 0.5)


class ScenarioTest(unittest.TestCase):
    def test_attackers_are_the_last_k_vds(self):
        self.assertEqual([p for p in range(8) if is_attacker(p, 8, 2)], [6, 7])
        self.assertFalse(any(is_attacker(p, 8, 0) for p in range(8)))
        # 指定編號時改用指定的 VD（例如挑幾乎沒塞車的 VD 當攻擊者）
        self.assertEqual([p for p in range(8) if is_attacker(p, 8, 2, ids=(1, 3))], [1, 3])

    def test_dropout_is_reproducible_and_near_rate(self):
        a = [drops_out(42, p, r, 0.3) for p in range(50) for r in range(1, 21)]
        self.assertEqual(a, [drops_out(42, p, r, 0.3) for p in range(50) for r in range(1, 21)])
        self.assertAlmostEqual(sum(a) / len(a), 0.3, delta=0.05)
        self.assertFalse(any(drops_out(42, p, 1, 0.0) for p in range(50)))




class AdaptiveShrinkTest(unittest.TestCase):
    def test_picks_largest_step_that_stays_within_tolerance(self):
        from security import adaptive_shrink

        g, bad = {"w": np.zeros(2, np.float32)}, {"w": np.full(2, 4.0, np.float32)}
        score = lambda a: 1.0 + float(a["w"][0])  # 分數隨偏移線性變差；原點 1.0
        out, alpha = adaptive_shrink(g, bad, score, tol=1.5)  # 容許到 2.5 → α=0.25（偏移 1）可、0.5（偏移 2）不可
        self.assertEqual(alpha, 0.25)
        np.testing.assert_allclose(out["w"], 1.0)

    def test_falls_back_to_global_when_nothing_passes(self):
        from security import adaptive_shrink

        g, bad = {"w": np.zeros(1, np.float32)}, {"w": np.full(1, 100.0, np.float32)}
        out, alpha = adaptive_shrink(g, bad, lambda a: 1.0 + float(a["w"][0]), tol=0.01)
        self.assertEqual(alpha, 0.0)
        np.testing.assert_allclose(out["w"], 0.0)


class KeyRotationTest(unittest.TestCase):
    """金鑰輪替（講義 TLS／金鑰管理：金鑰有使用期限）：不撤銷 VD，只把它的金鑰換新，偷到的舊金鑰從此無效。"""

    def test_rotation_replaces_both_keys_and_old_ones_stop_working(self):
        from security import rotate_vd, sign_update, verify_update
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            old_hmac, others = load_vd_key(d, 1), {p: load_vd_key(d, p) for p in (0, 2)}
            old_pub = load_vd_verify_keys(d)[1].public_bytes_raw()
            arrays = {"w": np.ones(3, np.float32)}
            stolen_tag = sign_update(old_hmac, arrays, 4, 1)
            rotate_vd(d, 1)
            _, table = load_server_secrets(d)
            self.assertEqual(table[1], load_vd_key(d, 1))  # VD 拿到的新金鑰和伺服器表一致
            self.assertNotEqual(table[1], old_hmac)
            self.assertFalse(verify_update(table[1], arrays, 4, 1, stolen_tag))
            pub = load_vd_verify_keys(d)[1]
            self.assertNotEqual(pub.public_bytes_raw(), old_pub)
            self.assertEqual(pub.public_bytes_raw(), load_vd_signing_key(d, 1).public_key().public_bytes_raw())
            self.assertEqual({p: load_vd_key(d, p) for p in (0, 2)}, others)  # 別台不受影響
            provision_keys(d, 3)  # 之後再跑配發不會把舊金鑰換回來
            self.assertEqual(load_server_secrets(d)[1][1], table[1])

    def test_cannot_rotate_a_vd_that_was_never_provisioned(self):
        from security import rotate_vd
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            for pid in (3, -1):  # 打錯編號不能憑空多一個有效身分
                with self.assertRaises(ValueError):
                    rotate_vd(d, pid)
            self.assertEqual(sorted(load_server_secrets(d)[1]), [0, 1, 2])

    def test_rotation_also_replaces_the_secure_aggregation_key(self):
        # 金鑰資料夾外洩時 vd-<編號>.x25519 也一起外洩；只換 HMAC／Ed25519 的話安全聚合身分不變
        from secagg import load_agreement_keys, provision_agreement_keys
        from security import rotate_vd
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            provision_agreement_keys(d, 3)
            old_priv, old_pubs = load_agreement_keys(d, 1)
            rotate_vd(d, 1)
            priv, pubs = load_agreement_keys(d, 1)
            raw = lambda k: k.public_bytes_raw()
            self.assertNotEqual(raw(priv.public_key()), raw(old_priv.public_key()))
            self.assertEqual(raw(pubs[1]), raw(priv.public_key()))
            self.assertEqual(raw(pubs[0]), raw(old_pubs[0]))

    def test_revoked_vd_cannot_be_rotated_back_in(self):
        from security import revoke_vd, rotate_vd
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 2)
            revoke_vd(d, 1)
            with self.assertRaises(ValueError):
                rotate_vd(d, 1)
            self.assertEqual(sorted(load_server_secrets(d)[1]), [0])


class PersistentRevocationTest(unittest.TestCase):
    def test_revoked_vd_stays_revoked_after_server_restart(self):
        from security import revoke_vd
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            revoke_vd(d, 1)
            provision_keys(d, 3)  # 重新執行配發（例如伺服器重開）不能把撤銷的 VD 加回來
            _, table = load_server_secrets(d)
            self.assertEqual(sorted(table), [0, 2])
            self.assertEqual(sorted(load_vd_verify_keys(d)), [0, 2])
            revoke_vd(d, 1)  # 重複撤銷不出錯
            self.assertEqual(sorted(load_server_secrets(d)[1]), [0, 2])

    def test_readmitting_a_revoked_vd_needs_new_keys(self):
        from security import revoke_vd
        with tempfile.TemporaryDirectory() as d:
            provision_keys(d, 3)
            old = load_server_secrets(d)[1][1]
            old_pub = load_vd_verify_keys(d)[1].public_bytes_raw()
            revoke_vd(d, 1)
            (Path(d) / "server" / "revoked.json").unlink()  # 管理者把它從撤銷清單拿掉
            provision_keys(d, 3)
            self.assertNotEqual(load_server_secrets(d)[1][1], old)  # 舊（可能已外洩的）金鑰不能復活
            self.assertNotEqual(load_vd_verify_keys(d)[1].public_bytes_raw(), old_pub)



class IntegerPayloadTest(unittest.TestCase):
    """安全聚合上傳的是整數：HMAC 要涵蓋型別和原始位元組，不能先轉 float32。"""
    def test_recast_to_wider_dtype_breaks_tag(self):
        from security import sign_update, verify_update
        key = b"k" * 32
        a = {"w": np.array([1, 65535], dtype=np.uint16)}
        tag = sign_update(key, a, 1, 0)
        self.assertTrue(verify_update(key, a, 1, 0, tag))
        self.assertFalse(verify_update(key, {"w": a["w"].astype(np.uint32)}, 1, 0, tag))

    def test_low_bits_of_uint32_are_covered(self):
        from security import sign_update, verify_update
        key = b"k" * 32
        a = {"w": np.array([2 ** 31], dtype=np.uint32)}
        tag = sign_update(key, a, 1, 0)
        self.assertFalse(verify_update(key, {"w": np.array([2 ** 31 + 1], dtype=np.uint32)}, 1, 0, tag))


if __name__ == "__main__":
    unittest.main()
