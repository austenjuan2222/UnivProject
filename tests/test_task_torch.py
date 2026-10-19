"""需要 torch（與 opacus）；未安裝時自動略過。"""
import importlib.util
import unittest

import numpy as np

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_OPACUS = HAVE_TORCH and importlib.util.find_spec("opacus") is not None


def _toy(n=512, lookback=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, lookback)).astype(np.float32)
    y = (0.8 * X[:, -1] + 0.1 * X[:, -2]).astype(np.float32)
    return X, y


@unittest.skipUnless(HAVE_TORCH, "需要 torch")
class TaskTorchTest(unittest.TestCase):
    def test_model_output_shape(self):
        import torch
        from task import TrafficModel
        out = TrafficModel(12)(torch.zeros(5, 12))
        self.assertEqual(tuple(out.shape), (5,))

    def test_training_reduces_loss_without_dp(self):
        import torch
        from task import TrafficModel, evaluate, make_loader, set_seed, train
        set_seed(0)
        X, y = _toy()
        m = TrafficModel(12)
        before = evaluate(m, X, y, 0.0, 1.0)["loss"]
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        _, steps = train(m, make_loader(X, y, 64, True), opt, epochs=5)
        self.assertEqual(steps, 5 * 8)
        self.assertLess(evaluate(m, X, y, 0.0, 1.0)["loss"], before)

    def test_evaluate_reports_kmh(self):
        import torch
        from task import TrafficModel, evaluate
        m = TrafficModel(12)
        for p in m.parameters():
            torch.nn.init.zeros_(p)
        X = np.zeros((4, 12), np.float32)
        y = np.ones(4, np.float32)
        r = evaluate(m, X, y, mean=80.0, std=10.0)
        self.assertAlmostEqual(r["mae_kmh"], 10.0, places=5)
        self.assertAlmostEqual(r["mse_kmh"], 100.0, places=4)

    @unittest.skipUnless(HAVE_OPACUS, "需要 opacus")
    def test_dp_training_steps_and_unwrapped_state_dict(self):
        import torch
        from opacus import PrivacyEngine
        from task import TrafficModel, make_loader, train, unwrap
        X, y = _toy()
        m = TrafficModel(12)
        keys = set(m.state_dict())
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        loader = make_loader(X, y, 64, True)
        pe = PrivacyEngine()
        pm, popt, ploader = pe.make_private(module=m, optimizer=opt, data_loader=loader,
                                            noise_multiplier=1.1, max_grad_norm=1.0)
        _, steps = train(pm, ploader, popt, epochs=2)
        self.assertEqual(steps, 2 * len(ploader))
        self.assertEqual(set(unwrap(pm).state_dict()), keys)

    @unittest.skipUnless(HAVE_OPACUS, "需要 opacus")
    def test_our_epsilon_matches_opacus_accountant_within_one_round(self):
        import torch
        from opacus import PrivacyEngine
        from privacy import compute_epsilon
        from task import TrafficModel, make_loader, train
        X, y = _toy()
        m = TrafficModel(12)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        pe = PrivacyEngine(accountant="rdp")
        pm, popt, ploader = pe.make_private(module=m, optimizer=opt,
                                            data_loader=make_loader(X, y, 64, True),
                                            noise_multiplier=1.1, max_grad_norm=1.0)
        _, steps = train(pm, ploader, popt, epochs=1)
        ours = compute_epsilon([(1.1, 1 / len(ploader), steps)], delta=1e-5)
        self.assertAlmostEqual(ours, pe.get_epsilon(1e-5), places=6)


if __name__ == "__main__":
    unittest.main()
