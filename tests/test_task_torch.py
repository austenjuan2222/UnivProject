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

    def test_residual_model_starts_from_last_value(self):
        import torch
        from task import TrafficModel
        m = TrafficModel(12, residual=True)
        for p in m.parameters():
            torch.nn.init.zeros_(p)
        x = torch.randn(4, 12)
        self.assertTrue(torch.allclose(m(x), x[:, -1]))   # MLP 修正量為 0 時 = naive

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

    def test_robust_losses_train_and_unknown_loss_fails(self):
        import torch
        from task import TrafficModel, make_loader, train
        X, y = _toy()
        for name in ("huber", "l1"):
            m = TrafficModel(12)
            opt = torch.optim.SGD(m.parameters(), lr=0.05)
            loss, steps = train(m, make_loader(X, y, 64, True), opt, epochs=1, loss=name)
            self.assertEqual(steps, 8)
            self.assertTrue(loss == loss)   # 不是 NaN
        with self.assertRaises(KeyError):
            train(TrafficModel(12), make_loader(X, y, 64, True),
                  torch.optim.SGD(TrafficModel(12).parameters(), lr=0.1), epochs=1, loss="nope")

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

    def test_evaluate_reports_naive_last_value_baseline(self):
        from task import TrafficModel, evaluate
        X = np.zeros((2, 12), np.float32)
        X[:, -1] = [0.5, -1.0]                     # 最後一筆（標準化）
        y = np.array([1.0, -1.0], np.float32)
        r = evaluate(TrafficModel(12), X, y, mean=80.0, std=10.0)
        # 「下一筆 = 最後一筆」的誤差：|0.5-1|*10=5、|-1-(-1)|*10=0 → 平均 2.5
        self.assertAlmostEqual(r["naive_mae_kmh"], 2.5, places=5)

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
        ours = compute_epsilon([(1.1, ploader.sample_rate, steps)], delta=1e-5)
        # Opacus 內部用 q=1/len(loader)，等於或略大於真實 q → 我們的 ε 不會比它大
        self.assertLessEqual(ours, pe.get_epsilon(1e-5) + 1e-9)
        if abs(1 / len(ploader) - ploader.sample_rate) < 1e-12:
            self.assertAlmostEqual(ours, pe.get_epsilon(1e-5), places=6)


if __name__ == "__main__":
    unittest.main()
