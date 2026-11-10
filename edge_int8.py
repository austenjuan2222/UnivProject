"""VD 端 int8 整數推論（講義 Edge 資源：沒有浮點運算單元的微控制器，例如 Cortex-M0）。

和 edge_numpy.predict 同一個 residual MLP，訓練完的模型做訓練後量化：
權重每個輸出通道一個比例、對稱量化成 int8；每層輸入的比例用一段校正資料的最大值決定；
中間只用整數：int8 × int8 累加成 int32，加 int32 偏差，ReLU，再乘一個 int32 倍數右移回 int8（CMSIS-NN 的做法）。
只有最前面把標準化車速轉成 int8、最後把結果轉回車速時用到一次乘法比例，在 MCU 上也可以換成定點數。
這裡的每一步都能逐行照抄成 C（int8_t、int32_t、int64_t）。
"""
from __future__ import annotations

import numpy as np

from edge_numpy import B0, B1, B2, W0, W1, W2

OUT_FRAC = 256  # 輸出比輸入細 256 倍：residual 模型只預測小幅變化，用輸入的格距會太粗


def multiplier(scale: float) -> tuple[int, int]:
    """把實數比例拆成 int32 倍數 m 和右移位數 s：x * scale ≈ (x * m) >> s，m 落在 [2^30, 2^31)。"""
    s = 30 - int(np.floor(np.log2(scale)))
    m = int(round(scale * 2.0 ** s))
    return (m // 2, s - 1) if m == 2 ** 31 else (m, s)  # 進位到 2^31 會超出 int32


def requant(acc, m, s):
    """(acc * m + 2^(s-1)) >> s，用 int64 算（C：(int32_t)(((int64_t)acc * m + (1LL << (s-1))) >> s)）。"""
    acc, m, s = np.asarray(acc, np.int64), np.asarray(m, np.int64), np.asarray(s, np.int64)
    return (acc * m + (np.int64(1) << (s - 1))) >> s


def _q(x, scale, lo=-127, hi=127):
    return np.clip(np.round(x / scale), lo, hi)


def quantize(p: dict, X_calib: np.ndarray) -> dict:
    s_x = float(np.abs(X_calib).max()) / 127
    h1 = np.maximum(X_calib @ p[W0].T + p[B0], 0)
    h2 = np.maximum(h1 @ p[W1].T + p[B1], 0)
    s_in = [s_x, float(h1.max()) / 127 or 1.0, float(h2.max()) / 127 or 1.0]
    s_out = s_x / OUT_FRAC
    q = {"s_x": s_x, "layers": []}
    for li, (w, b) in enumerate(((W0, B0), (W1, B1), (W2, B2))):
        sw = np.maximum(np.abs(p[w]).max(axis=1), 1e-12) / 127
        acc_scale = s_in[li] * sw  # 累加器一格代表的實數
        nxt = s_in[li + 1] if li < 2 else s_out
        ms = [multiplier(float(a / nxt)) for a in acc_scale]
        q["layers"].append({"w": _q(p[w], sw[:, None]).astype(np.int8),
                            "b": np.round(p[b] / acc_scale).astype(np.int32),
                            "m": np.array([m for m, _ in ms], np.int64), "s": np.array([s for _, s in ms], np.int64)})
    return q


def predict_int8(q: dict, X: np.ndarray, residual: bool = True, return_hidden: bool = False):
    a = _q(X, q["s_x"]).astype(np.int8)
    last = a[:, -1].astype(np.int64)
    hidden = []
    for li, L in enumerate(q["layers"]):
        acc = a.astype(np.int32) @ L["w"].T.astype(np.int32) + L["b"]
        if li < 2:
            a = np.clip(requant(np.maximum(acc, 0), L["m"], L["s"]), 0, 127).astype(np.int8)
            hidden.append(a)
        else:
            out = requant(acc, L["m"], L["s"])[:, 0] + (last * OUT_FRAC if residual else 0)
    if return_hidden:
        return hidden
    return out * (q["s_x"] / OUT_FRAC)


def footprint(q: dict) -> dict:
    """bytes：權重 int8＋偏差 int32（放 flash）、每通道倍數與位移、推論時最大的兩層暫存（RAM）。"""
    Ls = q["layers"]
    return {"weights": sum(L["w"].size + 4 * L["b"].size for L in Ls),
            "requant": sum(5 * L["m"].size for L in Ls),  # int32 倍數＋uint8 位移
            "ram": max(L["w"].shape[1] + 4 * L["w"].shape[0] for L in Ls)}
