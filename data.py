"""Client 資料：manifest 對應、時間切分、標準化、滑動視窗（只依賴 numpy/pandas）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def calendar_features(times, kinds: str = "tod") -> np.ndarray:
    """時間點的日曆特徵（公開資訊，不是私有量測）。

    tod：一天中的時段，以 sin/cos 編碼（23:55 與 00:00 相鄰）；dow：週末旗標。
    回傳 (len(times), k)；kinds 為空字串時 k = 0。
    """
    kinds = [k for k in kinds.split(",") if k]
    unknown = set(kinds) - {"tod", "dow"}
    if unknown:
        raise ValueError(f"未知的日曆特徵 {sorted(unknown)}；可用 tod、dow")
    t = pd.to_datetime(pd.Series(times))
    cols = []
    if "tod" in kinds:
        m = (t.dt.hour * 60 + t.dt.minute).to_numpy(float) / 1440 * 2 * np.pi
        cols += [np.sin(m), np.cos(m)]
    if "dow" in kinds:
        cols.append((t.dt.dayofweek >= 5).to_numpy(float))
    return np.stack(cols, 1) if cols else np.zeros((len(t), 0))


def n_calendar(kinds: str) -> int:
    kinds = [k for k in kinds.split(",") if k]
    return 2 * ("tod" in kinds) + ("dow" in kinds)


def make_windows(x: np.ndarray, lookback: int = 12, horizon: int = 1, cal: np.ndarray | None = None):
    """x[t-lookback:t] → x[t+horizon-1]（horizon=1 為下一筆）；含 NaN 的視窗捨棄。

    cal（與 x 等長的日曆特徵）不為 None 時，取「目標時間點」那一列放在視窗前面：
    X = [日曆特徵, 車速視窗]，所以 X[:, -1] 仍是最後一筆車速（殘差模型與 naive 基準照用）。
    """
    x = np.asarray(x, dtype=float)
    k = 0 if cal is None else cal.shape[1]
    n = len(x) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback + k)), np.empty((0,))
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X, y = x[idx], x[lookback + horizon - 1:]
    if k:
        X = np.concatenate([cal[lookback + horizon - 1:], X], axis=1)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[ok], y[ok]


STEPS_PER_DAY = 288  # 5 分鐘一筆


def split_series(x: np.ndarray, lookback: int = 12, train_frac: float = 0.8, horizon: int = 1,
                 cal: np.ndarray | None = None, lag_day: bool = False) -> dict:
    """依時間切訓練/測試，兩邊視窗互不跨越切點；標準化參數只用訓練段（日曆特徵不標準化）。

    lag_day=True 時多一個輸入：目標時間點「昨天同一時刻」的車速（同一 VD 的過去值，已標準化）。
    測試期間第一天的昨天值落在訓練期間，屬於預測當下已知的歷史，不是洩漏。
    """
    x = np.asarray(x, dtype=float)
    split = int(len(x) * train_frac)
    train, test = x[:split], x[split:]
    mean = float(np.nanmean(train))
    std = float(np.nanstd(train))
    std = std if std > 1e-6 else 1.0
    if lag_day:
        z = (x - mean) / std
        lag = np.full(len(x), np.nan)
        lag[STEPS_PER_DAY:] = z[:-STEPS_PER_DAY]
        cal = lag[:, None] if cal is None else np.concatenate([cal, lag[:, None]], axis=1)
    ctr, cte = (None, None) if cal is None else (cal[:split], cal[split:])
    Xtr, ytr = make_windows((train - mean) / std, lookback, horizon, ctr)
    Xte, yte = make_windows((test - mean) / std, lookback, horizon, cte)
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
                       lookback: int = 12, train_frac: float = 0.8, horizon: int = 1,
                       calendar: str = "", lag_day: bool = False) -> dict:
    info = client_info(data_dir, partition_id, num_partitions)
    df = pd.read_csv(Path(data_dir) / info["file"])
    cal = calendar_features(df["time"], calendar) if calendar else None
    d = split_series(df["speed"].to_numpy(dtype=float), lookback, train_frac, horizon, cal, lag_day)
    if len(d["y_train"]) == 0 or len(d["y_test"]) == 0:
        raise ValueError(f"{info['vdid']} 有效視窗不足（train={len(d['y_train'])}, test={len(d['y_test'])}）")
    d["vdid"] = info["vdid"]
    return d
