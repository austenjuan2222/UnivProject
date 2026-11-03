"""上傳壓縮：把「本地模型 − 全域模型」的更新量量化成 b 位元整數再上傳（講義：LPWAN 頻寬小、Edge 降低流量）。

每個張量一個尺度 s = max|更新量| / (2^(b−1) − 1)，以隨機捨入（期望值不偏）轉成整數；伺服器收到後還原。
這是對 DP 訓練結果的後處理，不影響 ε。b=32 代表不壓縮（原本的 float32）。
"""
from __future__ import annotations

import numpy as np


def quantize_update(global_arrays: dict, local_arrays: dict, bits: int,
                    rng: np.random.Generator) -> tuple[dict, int]:
    """回傳（伺服器還原後的模型參數, 上傳的位元組數）。"""
    if bits >= 32:
        out = {k: np.asarray(v, dtype=np.float32) for k, v in local_arrays.items()}
        return out, int(sum(v.size * 4 for v in out.values()))
    if not 2 <= bits <= 16:
        raise ValueError(f"upload-bits 必須在 2～16 或 32，收到 {bits}")
    levels = 2 ** (bits - 1) - 1
    out, nbytes = {}, 0
    for k, g in global_arrays.items():
        g = np.asarray(g, dtype=np.float32)
        delta = np.asarray(local_arrays[k], dtype=np.float32) - g
        m = float(np.abs(delta).max())
        if m == 0.0:
            q = np.zeros_like(delta)
            scale = 0.0
        else:
            scale = m / levels
            x = delta / scale
            low = np.floor(x)
            q = low + (rng.random(x.shape) < (x - low))  # 隨機捨入
            q = np.clip(q, -levels, levels)
        out[k] = (g + q * scale).astype(np.float32)
        nbytes += -(-delta.size * bits // 8) + 4  # 整數部分（無條件進位到 byte）＋一個 float32 尺度
    return out, int(nbytes)
