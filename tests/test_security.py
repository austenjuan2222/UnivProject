"""security.py：模型更新的完整性（HMAC）、全域模型簽章（Ed25519）、攻擊模擬。"""
import unittest

import numpy as np

from security import (drops_out, is_attacker, poison_update, server_private_key, server_public_key, sign_model, sign_update,
                      tamper, verify_model, verify_update, vd_key)


def _arrays(v=0.5):
    return {"net.0.weight": np.full((3, 2), v, dtype=np.float32),
            "net.0.bias": np.arange(3, dtype=np.float32)}


class UpdateMacTest(unittest.TestCase):
    def setUp(self):
        self.key = vd_key("lab-secret", 3)

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

    def test_other_vd_key_cannot_forge(self):
        tag = sign_update(vd_key("lab-secret", 4), _arrays(), rnd=5, pid=3)
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, tag))

    def test_keys_differ_per_vd_and_secret(self):
        self.assertNotEqual(vd_key("lab-secret", 1), vd_key("lab-secret", 2))
        self.assertNotEqual(vd_key("lab-secret", 1), vd_key("other", 1))

    def test_garbage_tag_is_rejected_not_raised(self):
        self.assertFalse(verify_update(self.key, _arrays(), 5, 3, "zz-not-hex"))


class ModelSignatureTest(unittest.TestCase):
    def setUp(self):
        self.priv = server_private_key("lab-secret")
        self.pub = server_public_key("lab-secret")

    def test_global_model_signature_verifies(self):
        sig = sign_model(self.priv, _arrays(), rnd=2)
        self.assertTrue(verify_model(self.pub, _arrays(), 2, sig))

    def test_fake_model_update_is_rejected(self):
        sig = sign_model(self.priv, _arrays(), rnd=2)
        self.assertFalse(verify_model(self.pub, _arrays(0.6), 2, sig))
        self.assertFalse(verify_model(self.pub, _arrays(), 3, sig))

    def test_signature_from_other_server_is_rejected(self):
        sig = sign_model(server_private_key("attacker"), _arrays(), rnd=2)
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

    def test_dropout_is_reproducible_and_near_rate(self):
        a = [drops_out(42, p, r, 0.3) for p in range(50) for r in range(1, 21)]
        self.assertEqual(a, [drops_out(42, p, r, 0.3) for p in range(50) for r in range(1, 21)])
        self.assertAlmostEqual(sum(a) / len(a), 0.3, delta=0.05)
        self.assertFalse(any(drops_out(42, p, 1, 0.0) for p in range(50)))


if __name__ == "__main__":
    unittest.main()
