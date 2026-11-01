"""Client 資料：manifest 對應、時間切分、標準化、滑動視窗（只依賴 numpy/pandas）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def make_windows(x: np.ndarray, lookback: int = 12, horizon: int = 1):
    """x[t-lookback:t] → x[t+horizon-1]（horizon=1 為下一筆）；含 NaN 的視窗捨棄。"""
    x = np.asarray(x, dtype=float)
    n = len(x) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback)), np.empty((0,))
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X, y = x[idx], x[lookback + horizon - 1:]
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[ok], y[ok]


def split_series(x: np.ndarray, lookback: int = 12, train_frac: float = 0.8, horizon: int = 1) -> dict:
    """依時間切訓練/測試，兩邊視窗互不跨越切點；標準化參數只用訓練段。"""
    x = np.asarray(x, dtype=float)
    split = int(len(x) * train_frac)
    train, test = x[:split], x[split:]
    mean = float(np.nanmean(train))
    std = float(np.nanstd(train))
    std = std if std > 1e-6 else 1.0
    Xtr, ytr = make_windows((train - mean) / std, lookback, horizon)
    Xte, yte = make_windows((test - mean) / std, lookback, horizon)
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return {"X_train": f32(Xtr), "y_train": f32(ytr), "X_test": f32(Xte), "y_test": f32(yte),
            "mean": mean, "std": std, "split_index": split}


def load_manifest(data_dir) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先執行 scripts/download_vd.py 與 scripts/preprocess_vd.py，"
            "並確認 run config 的 data-dir 是絕對路徑。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def client_info(data_dir, partition_id: int, num_partitions: int) -> dict:
    clients = load_manifest(data_dir)["clients"]
    if num_partitions != len(clients):
        raise ValueError(
            f"模擬的 client 數（num-supernodes={num_partitions}）與 manifest 中的 VD 數"
            f"（{len(clients)}）不一致。請用 scripts/run_experiments.sh 執行，"
            "或修改 Flower 設定檔的 options.num-supernodes。"
        )
    return clients[partition_id]


@lru_cache(maxsize=None)
def load_client_arrays(data_dir, partition_id: int, num_partitions: int,
                       lookback: int = 12, train_frac: float = 0.8, horizon: int = 1) -> dict:
    info = client_info(data_dir, partition_id, num_partitions)
    df = pd.read_csv(Path(data_dir) / info["file"])
    d = split_series(df["speed"].to_numpy(dtype=float), lookback, train_frac, horizon)
    if len(d["y_train"]) == 0 or len(d["y_test"]) == 0:
        raise ValueError(f"{info['vdid']} 有效視窗不足（train={len(d['y_train'])}, test={len(d['y_test'])}）")
    d["vdid"] = info["vdid"]
    return d
