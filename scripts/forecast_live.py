#!/usr/bin/env python3
"""用訓練好的全域模型預測國道 VD 未來 5／10／15 分鐘的車速（給一般使用者用的最小版本）。

資料：高公局歷史 VD 檔（每分鐘一檔，和 download_vd.py 同來源），抓預測時間點前約 70 分鐘，
彙整成 5 分鐘車速（同 preprocess 的流量加權、短缺口內插），取最後 12 筆（1 小時）當輸入。
10、15 分鐘是把前一步的預測接回輸入再預測一次（遞迴），誤差會累積。

範例：
  python scripts/forecast_live.py --run-dir results/<批次>/dp_sigma1.1_seed1 --data-dir ../UnivProject-run/data/processed_28d_32
  python scripts/forecast_live.py ... --vd VD-N1-N-10.060-M-RS --at "2024-03-29 08:00"   # 回測：用過去的時間點，順便看實際值
標準化參數用 data-dir 裡該 VD 前 80% 的資料（和訓練時相同），所以只支援 data-dir 裡有的 VD。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data import calendar_features, n_calendar  # noqa: E402
from vd_preprocess import aggregate_5min, fill_short_gaps  # noqa: E402

LOOKBACK_MIN = 70  # 12 個 5 分鐘＋目前還沒滿的那一桶＋幾分鐘緩衝


def load_model(run_dir: Path):
    import torch
    from task import TrafficModel
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["run_config"]
    model = TrafficModel(c["lookback"], c["hidden-size"], c["residual"], n_calendar(c["calendar"]) + int(c["lag-day"]))
    # weights_only：模型檔只該有張量，不執行檔案裡的任何程式
    model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=True), strict=True)
    if c["lag-day"]:
        raise SystemExit("這個模型需要昨天同時刻的車速（lag-day），即時版本還不支援")
    return model.eval(), c


def vd_stats(data_dir: Path, train_frac: float) -> dict[str, tuple[float, float]]:
    """每台 VD 的標準化參數（只用前 train_frac 的資料，和 data.split_series 相同）。"""
    man = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for c in man["clients"]:
        s = pd.read_csv(data_dir / c["file"])["speed"].to_numpy(float)
        tr = s[: int(len(s) * train_frac)]
        std = float(np.nanstd(tr))
        out[c["vdid"]] = (float(np.nanmean(tr)), std if std > 1e-6 else 1.0)
    return out


def five_minute_series(rows: list[dict], end: pd.Timestamp, max_gap: int = 3) -> dict[str, pd.Series]:
    """每分鐘列 → 每台 VD 到 end（不含）為止的完整 5 分鐘車速；end 所在、還沒滿的那一桶不用。"""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for vdid, g in df.groupby("vdid"):
        s = aggregate_5min(g)["speed"]
        s = s[s.index < end.floor("5min")]
        out[vdid] = fill_short_gaps(s, max_gap)
    return out


def forecast(model, cfg: dict, series: pd.Series, mean: float, std: float, steps: int = 3) -> list[tuple]:
    """回傳 [(目標時間, 預測車速 km/h)]；最後 12 筆有缺值時回傳空清單（不硬猜）。"""
    import torch
    lb = cfg["lookback"]
    if len(series) < lb or series.iloc[-lb:].isna().any():
        return []
    z = list((series.iloc[-lb:].to_numpy(float) - mean) / std)
    t, out = series.index[-1], []
    for _ in range(steps):
        t = t + pd.Timedelta(minutes=5)
        cal = calendar_features([t], cfg["calendar"])[0] if cfg["calendar"] else np.zeros(0)
        x = torch.tensor(np.concatenate([cal, z[-lb:]])[None, :], dtype=torch.float32)
        with torch.no_grad():
            p = float(model(x)[0])
        z.append(p)
        out.append((t, p * std + mean))
    return out


def fetch_minutes(end: dt.datetime, keep, minutes: int = LOOKBACK_MIN, fetch=None) -> list[dict]:
    """抓 end 之前 minutes 分鐘的每分鐘檔；不存在（還沒上傳）的分鐘略過。"""
    from download_vd import DEFAULT_BASE, _fetch_parse, http_fetch
    fetch = fetch or http_fetch
    rows = []
    for k in range(minutes, 0, -1):
        t = end - dt.timedelta(minutes=k)
        url = f"{DEFAULT_BASE}/{t:%Y%m%d}/VDLive_{t:%H%M}.xml.gz"
        r, _err = _fetch_parse(url, fetch, keep, retries=2, backoff=1.0)
        rows += r
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path, help="含 final_model.pt 與 config.json 的訓練結果")
    p.add_argument("--data-dir", required=True, type=Path, help="訓練用的資料夾（取標準化參數與 VD 清單）")
    p.add_argument("--vd", nargs="*", help="只看這些 VD（預設 data-dir 裡全部）")
    p.add_argument("--at", help="預測時間點（台灣時間，YYYY-MM-DD HH:MM）；預設現在")
    a = p.parse_args(argv)

    model, cfg = load_model(a.run_dir)
    stats = vd_stats(a.data_dir, cfg["train-frac"])
    vds = a.vd or sorted(stats)
    unknown = [v for v in vds if v not in stats]
    if unknown:
        raise SystemExit(f"data-dir 裡沒有這些 VD，無法取得標準化參數：{unknown}")
    if a.at:
        end = dt.datetime.strptime(a.at, "%Y-%m-%d %H:%M")
    else:
        end = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None, second=0, microsecond=0)
    wanted = set(vds)
    rows = fetch_minutes(end, keep=wanted.__contains__)
    series = five_minute_series(rows, pd.Timestamp(end))
    print(f"預測時間點 {end:%Y-%m-%d %H:%M}（台灣時間）；抓到 {len({r['time'] for r in rows})} 分鐘的資料")
    print("| VD | 最近 5 分鐘車速 | +5 分 | +10 分 | +15 分 |\n|---|---|---|---|---|")
    for v in vds:
        s = series.get(v, pd.Series(dtype=float))
        f = forecast(model, cfg, s, *stats[v])
        last = f"{s.iloc[-1]:.1f}（{s.index[-1]:%H:%M}）" if len(s) and not np.isnan(s.iloc[-1]) else "–"
        cells = " | ".join(f"{kmh:.1f}" for _, kmh in f) if f else "資料不足 | – | –"
        print(f"| {v} | {last} | {cells} |")


if __name__ == "__main__":
    main()
