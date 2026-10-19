"""跨輪次累計 DP-SGD 隱私預算 ε。

問題：Flower 每輪都重新建立 ClientApp/PrivacyEngine，Opacus accountant 會歸零，
`privacy_engine.get_epsilon()` 只反映「這一輪」。

做法：每輪結束把（σ, q, 步數）附加到 client 的 `context.state`，
每次用全部歷史重算 RDP 總和再轉成 (ε, δ)。公式與 Opacus `RDPAccountant`
完全相同（同一組 orders、同一個 `compute_rdp` / `get_privacy_spent`）。
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from opacus.accountants.analysis import rdp as rdp_analysis

# 與 opacus.accountants.RDPAccountant.DEFAULT_ALPHAS 相同
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def compute_epsilon(history: Iterable[tuple[float, float, int]], delta: float) -> float:
    """history = [(noise_multiplier σ, sample_rate q, steps), ...]。"""
    total = np.zeros(len(ORDERS))
    any_steps = False
    for sigma, q, steps in history:
        if steps <= 0:
            continue
        any_steps = True
        total = total + np.asarray(
            rdp_analysis.compute_rdp(q=float(q), noise_multiplier=float(sigma),
                                     steps=int(steps), orders=ORDERS)
        )
    if not any_steps:
        return 0.0
    eps, _ = rdp_analysis.get_privacy_spent(orders=ORDERS, rdp=total, delta=delta)
    return float(eps)


def append_history(history: Optional[dict], sigma: float, q: float, steps: int) -> dict:
    """在 {"sigma": [...], "q": [...], "steps": [...]} 附加一筆；設定相同就累加步數。"""
    h = {"sigma": [], "q": [], "steps": []} if history is None else {
        k: list(history[k]) for k in ("sigma", "q", "steps")
    }
    for i, (s, qq) in enumerate(zip(h["sigma"], h["q"])):
        if s == sigma and qq == q:
            h["steps"][i] += int(steps)
            return h
    h["sigma"].append(float(sigma))
    h["q"].append(float(q))
    h["steps"].append(int(steps))
    return h


def history_tuples(history: Optional[dict]) -> list[tuple[float, float, int]]:
    if history is None:
        return []
    return list(zip(history["sigma"], history["q"], history["steps"]))
