"""雲端版即時預測：和 scripts/forecast_live.py 相同流程，但模型用 numpy 計算（不需 torch）。

模型與標準化參數來自 model_bundle.json（由桌機從訓練結果匯出）。
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import calendar_features
from download_vd import DEFAULT_BASE, _fetch_parse, http_fetch
from vd_preprocess import aggregate_5min, fill_short_gaps

LOOKBACK_MIN = 70
STALE_MIN = 20


class NumpyModel:
    """TrafficModel 的 numpy 版：Linear→ReLU→Linear→ReLU→Linear，residual 時加回最後一筆。"""

    def __init__(self, state: dict, residual: bool):
        g = lambda k: np.asarray(state[k], dtype=np.float64)  # noqa: E731
        self.W = [g(f"net.{i}.weight") for i in (0, 2, 4)]
        self.b = [g(f"net.{i}.bias") for i in (0, 2, 4)]
        self.residual = residual

    def __call__(self, x: np.ndarray) -> float:
        h = np.maximum(self.W[0] @ x + self.b[0], 0)
        h = np.maximum(self.W[1] @ h + self.b[1], 0)
        out = float((self.W[2] @ h + self.b[2])[0])
        return out + float(x[-1]) if self.residual else out


def load_bundle(path: Path):
    b = json.loads(path.read_text(encoding="utf-8"))
    if b.get("format") != "univproject-forecast-bundle/1":
        raise ValueError("不認得的 bundle 格式")
    m = b["model"]
    if int(m["horizon"]) != 1:
        raise ValueError("雲端版只支援 horizon=1 的模型")
    cfg = {"lookback": int(m["lookback"]), "calendar": m["calendar"]}
    state = {k: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"]) for k, v in b["state_dict"].items()}
    stats = {v["vdid"]: (float(v["mean"]), float(v["std"])) for v in b["vds"]}
    return NumpyModel(state, bool(m["residual"])), cfg, stats, b


def data_cutoff(rows, end):
    if not rows:
        return end.floor("5min")
    last = pd.to_datetime(max(r["time"] for r in rows), utc=True, format="ISO8601").tz_convert("Asia/Taipei").tz_localize(None)
    return min(end.floor("5min"), (last + pd.Timedelta(minutes=1)).floor("5min"))


def five_minute_series(rows, end, max_gap=3):
    if not rows:
        return {}
    cut = data_cutoff(rows, end)
    out = {}
    for vdid, g in pd.DataFrame(rows).groupby("vdid"):
        s = aggregate_5min(g)["speed"]
        s = s[s.index < cut]
        if s.empty:
            continue
        s = s.reindex(pd.date_range(s.index.min(), cut - pd.Timedelta(minutes=5), freq="5min"))
        out[vdid] = fill_short_gaps(s, max_gap)
    return out


def forecast(model, cfg, series, mean, std, steps=3):
    lb = cfg["lookback"]
    if len(series) < lb or series.iloc[-lb:].isna().any():
        return []
    z = list((series.iloc[-lb:].to_numpy(float) - mean) / std)
    t, out = series.index[-1], []
    for _ in range(steps):
        t = t + pd.Timedelta(minutes=5)
        cal = calendar_features([t], cfg["calendar"])[0] if cfg["calendar"] else np.zeros(0)
        p = model(np.concatenate([cal, z[-lb:]]))
        z.append(p)
        out.append((t, p * std + mean))
    return out


def fetch_minutes(end, keep, minutes=LOOKBACK_MIN, workers=12):
    times = [end + dt.timedelta(minutes=k) for k in range(-minutes, 0)]
    urls = [f"{DEFAULT_BASE}/{t:%Y%m%d}/VDLive_{t:%H%M}.xml.gz" for t in times]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda u: _fetch_parse(u, lambda x: http_fetch(x, timeout=8.0), keep, retries=2, backoff=0.5), urls))
    return [r for rows, _e in res for r in rows], [e for _r, e in res if e]


def taipei_now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None, second=0, microsecond=0)


def predict_all(model, cfg, stats, vds, end):
    minutes = int(cfg["lookback"]) * 5 + LOOKBACK_MIN - 60
    rows, errs = fetch_minutes(end, keep=set(vds).__contains__, minutes=minutes)
    t_end = pd.Timestamp(end)
    local = pd.to_datetime([r["time"] for r in rows], utc=True, format="ISO8601").tz_convert("Asia/Taipei").tz_localize(None)
    rows_in = [r for r, t in zip(rows, local) if t < t_end]
    series = five_minute_series(rows_in, t_end)
    cut = data_cutoff(rows_in, t_end)
    stale = not rows_in or t_end - cut > pd.Timedelta(minutes=STALE_MIN)
    out = []
    for v in vds:
        s = series.get(v, pd.Series(dtype=float))
        ok = len(s) and not np.isnan(s.iloc[-1])
        preds = [] if stale else [
            {"time": f"{t:%H:%M}", "at": f"{t:%Y-%m-%d %H:%M}", "ahead_min": int((t - t_end) / pd.Timedelta(minutes=1)),
             "kmh": round(kmh, 1), "actual": None}
            for t, kmh in forecast(model, cfg, s, *stats[v])]
        out.append({"vdid": v, "last_time": f"{s.index[-1]:%H:%M}" if ok else None,
                    "last_kmh": round(float(s.iloc[-1]), 1) if ok else None, "preds": preds})
    r = {"end": f"{end:%Y-%m-%d %H:%M}", "minutes": len({x["time"] for x in rows_in}), "backtest": False,
         "data_until": f"{cut:%H:%M}", "data_age_min": int((t_end - cut) / pd.Timedelta(minutes=1)),
         "stale": bool(stale), "vds": out}
    if not rows_in:
        r["error"] = f"抓不到高公局資料（{(errs or ['無回應'])[0][:80]}）"
    return r
