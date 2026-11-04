"""不用 PyTorch 的 VD 端 DP-SGD（只用 numpy），給記憶體小的路邊設備（講義：Edge 資源限制）。

和 task.TrafficModel（residual、ReLU 的三層 MLP）與 Opacus 的 DP-SGD 同一個演算法：
每筆樣本的 MSE 梯度各自裁剪到 L2 ≤ C，加總後加 N(0, (σC)²) 雜訊，除以期望批次大小，再走一步 SGD；
批次用 Poisson 取樣（每筆以機率 q 入選），每個 epoch 走 round(1/q) 步。參數名稱和 PyTorch 的 state_dict 一樣，可直接互換。
ε 由伺服器（或任何有 Opacus 的地方）用 (σ, q, 步數) 計算，VD 端不需要。
"""
from __future__ import annotations

import numpy as np

W0, B0, W1, B1, W2, B2 = "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"


def _forward(p: dict, X: np.ndarray):
    a1 = X @ p[W0].T + p[B0]
    h1 = np.maximum(a1, 0)
    a2 = h1 @ p[W1].T + p[B1]
    h2 = np.maximum(a2, 0)
    return a1, h1, a2, h2, (h2 @ p[W2].T + p[B2])[:, 0]


def predict(p: dict, X: np.ndarray, residual: bool = True) -> np.ndarray:
    out = _forward(p, X)[-1]
    return out + X[:, -1] if residual else out


def per_sample_grads(p: dict, X: np.ndarray, y: np.ndarray, residual: bool = True) -> dict:
    """每筆樣本 (pred − y)² 對各參數的梯度，形狀 (B, *參數形狀)。"""
    a1, h1, a2, h2, out = _forward(p, X)
    g = 2 * (out + (X[:, -1] if residual else 0) - y)                  # (B,)
    d2 = (g[:, None] * p[W2][0]) * (a2 > 0)                            # (B, H)
    d1 = (d2 @ p[W1]) * (a1 > 0)                                       # (B, H)
    return {W2: g[:, None, None] * h2[:, None, :], B2: g[:, None],
            W1: d2[:, :, None] * h1[:, None, :], B1: d2,
            W0: d1[:, :, None] * X[:, None, :], B0: d1}


def dp_step(p: dict, X: np.ndarray, y: np.ndarray, lr: float, sigma: float, clip: float,
            expected_batch: float, rng: np.random.Generator, residual: bool = True) -> dict:
    gs = per_sample_grads(p, X, y, residual) if len(y) else {k: np.zeros((0,) + v.shape) for k, v in p.items()}
    norms = np.sqrt(sum((v.reshape(len(y), -1) ** 2).sum(1) for v in gs.values())) if len(y) else np.zeros(0)
    factor = np.minimum(1.0, clip / (norms + 1e-6))                     # 和 Opacus 一樣的裁剪係數
    out = {}
    for k, v in p.items():
        s = np.tensordot(factor, gs[k], axes=1) if len(y) else np.zeros_like(v)
        noisy = (s + rng.normal(0.0, sigma * clip, v.shape)) / expected_batch
        out[k] = (v - lr * noisy).astype(np.float32)
    return out


def sampling(n: int, batch_size: int) -> tuple[float, int, int]:
    """和 Opacus DPDataLoader 一樣的（取樣率 q, 每 epoch 步數, 期望批次大小）。"""
    q = 1.0 / max(1, -(-n // batch_size))                              # 1 / DataLoader 的批次數
    return q, int(1 / q), int(n * q)


def dp_train(p: dict, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float, sigma: float,
             clip: float, rng: np.random.Generator, residual: bool = True) -> tuple[dict, float, int]:
    """回傳（新參數, 取樣率 q, 步數）。"""
    q, n_steps, expected = sampling(len(y), batch_size)
    steps = 0
    for _ in range(epochs):
        for _ in range(n_steps):
            m = rng.random(len(y)) < q
            p = dp_step(p, X[m], y[m], lr, sigma, clip, expected, rng, residual)
            steps += 1
    return p, q, steps
