#!/usr/bin/env python3
"""訓練完的全域模型改用 int8 整數推論，在每台 VD 的測試資料上比 MAE（講義 Edge 資源）。

校正資料只用前 4 台 VD 的訓練段（代表伺服器手上的參考資料；VD 的原始資料不離開 VD），量化後同一份 int8 模型發給所有 VD。

  python scripts/edge_int8_eval.py --data-dir data/processed_28d_32 results/night11/len/*/dp_sigma1.1_seed*/final_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_client_arrays  # noqa: E402
from edge_int8 import footprint, predict_int8, quantize  # noqa: E402
from edge_numpy import predict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-vd", type=int, default=32)
    ap.add_argument("--calib-vd", type=int, default=4)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    cfg = json.loads((Path(a.models[0]).parent / "config.json").read_text(encoding="utf-8"))["run_config"]
    data = [load_client_arrays(a.data_dir, i, a.num_vd, cfg["lookback"], cfg["train-frac"], cfg["horizon"],
                               cfg["calendar"], cfg["lag-day"]) for i in range(a.num_vd)]
    calib = np.concatenate([d["X_train"] for d in data[:a.calib_vd]])
    print("| 模型 | float32 MAE | int8 MAE | 差值 | 塞車時段 float32／int8 | flash（權重） | RAM（暫存） |")
    print("|---|---|---|---|---|---|---|")
    for path in a.models:
        p = {k: v.numpy().astype(np.float32) for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        q = quantize(p, calib)
        ef, ei, jf, ji = [], [], [], []
        for d in data:
            yt = d["y_test"] * d["std"] + d["mean"]
            f = predict(p, d["X_test"]) * d["std"] + d["mean"]
            i = predict_int8(q, d["X_test"]) * d["std"] + d["mean"]
            jam = yt < 60
            ef.append(np.abs(f - yt)), ei.append(np.abs(i - yt)), jf.append(np.abs(f - yt)[jam]), ji.append(np.abs(i - yt)[jam])
        mf, mi = np.concatenate(ef).mean(), np.concatenate(ei).mean()
        fp = footprint(q)
        print(f"| {Path(path).parent.name} | {mf:.4f} | {mi:.4f} | {mi - mf:+.4f} | {np.concatenate(jf).mean():.3f}／{np.concatenate(ji).mean():.3f} "
              f"| {fp['weights'] + fp['requant']} B（float32 {4 * sum(v.size for v in p.values())} B） | {fp['ram']} B |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
