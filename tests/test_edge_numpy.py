"""numpy 版 VD 端 DP-SGD 要和 PyTorch＋Opacus 版算出一樣的東西（才能說換執行環境不影響結果）。"""
import unittest

import numpy as np
import torch

from edge_numpy import dp_step, per_sample_grads, predict
from task import TrafficModel


def setup(seed=0, n=40, lookback=14):
    torch.manual_seed(seed)
    m = TrafficModel(12, 32, residual=True, n_extra=lookback - 12)
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, lookback)).astype(np.float32)
    y = (X[:, -1] + rng.normal(0, 0.3, n)).astype(np.float32)
    p = {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}
    return m, p, X, y


class EdgeNumpyTest(unittest.TestCase):
    def test_prediction_matches_torch(self):
        m, p, X, _ = setup()
        np.testing.assert_allclose(predict(p, X), m(torch.from_numpy(X)).detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_per_sample_grads_match_autograd(self):
        m, p, X, y = setup(n=5)
        gs = per_sample_grads(p, X, y)
        for i in range(5):
            m.zero_grad()
            ((m(torch.from_numpy(X[i:i + 1])) - torch.from_numpy(y[i:i + 1])) ** 2).sum().backward()
            for k, t in m.named_parameters():
                np.testing.assert_allclose(gs[k][i], t.grad.numpy(), rtol=1e-4, atol=1e-6)

    def test_dp_step_matches_opacus_without_noise(self):
        from opacus import GradSampleModule
        from opacus.optimizers import DPOptimizer

        m, p, X, y = setup(n=40)
        gm = GradSampleModule(m)
        opt = DPOptimizer(torch.optim.SGD(gm.parameters(), lr=0.1), noise_multiplier=0.0, max_grad_norm=0.5,
                          expected_batch_size=32)
        torch.nn.functional.mse_loss(gm(torch.from_numpy(X)), torch.from_numpy(y)).backward()
        opt.step()
        got = dp_step(p, X, y, lr=0.1, sigma=0.0, clip=0.5, expected_batch=32, rng=np.random.default_rng(0))
        for k, t in m.state_dict().items():
            np.testing.assert_allclose(got[k], t.numpy(), rtol=1e-4, atol=1e-6)

    def test_noise_has_expected_scale(self):
        _, p, X, y = setup()
        rng = np.random.default_rng(1)
        a = dp_step(p, X[:0], y[:0], lr=1.0, sigma=2.0, clip=0.5, expected_batch=10, rng=rng)
        d = (p["net.2.weight"] - a["net.2.weight"]).ravel()  # 空批次：只有雜訊 / 期望批次
        self.assertAlmostEqual(float(d.std()), 2.0 * 0.5 / 10, delta=0.01)


if __name__ == "__main__":
    unittest.main()
