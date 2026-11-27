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
from data import calendar_features, n_calendar, read_vd_csv  # noqa: E402
from vd_preprocess import aggregate_5min, fill_short_gaps  # noqa: E402

LOOKBACK_MIN = 70  # 12 個 5 分鐘＋目前還沒滿的那一桶＋幾分鐘緩衝


def load_model(run_dir: Path):
    import torch
    from task import TrafficModel
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["run_config"]
    if int(c.get("horizon", 1)) != 1:
        # 遞迴預測把每步輸出當成下一個 5 分鐘接回輸入，只對單步模型成立
        raise SystemExit(f"這個模型是 horizon={c['horizon']}（一次跳 {5 * int(c['horizon'])} 分鐘），即時版本只支援 horizon=1")
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
        s = read_vd_csv(data_dir / c["file"])["speed"].to_numpy(float)
        tr = s[: int(len(s) * train_frac)]
        std = float(np.nanstd(tr))
        out[c["vdid"]] = (float(np.nanmean(tr)), std if std > 1e-6 else 1.0)
    return out


STALE_MIN = 20  # 資料比預測時間點舊超過這麼多分鐘就不預測（來源平常晚 4～7 分鐘）


def data_cutoff(rows: list[dict], end: pd.Timestamp) -> pd.Timestamp:
    """完整 5 分鐘桶的結束點：不超過 end，也不超過來源最後一分鐘（來源晚上傳時，最後一桶還沒收齊）。"""
    if not rows:
        return end.floor("5min")
    last = pd.to_datetime(max(r["time"] for r in rows), utc=True, format="ISO8601").tz_convert("Asia/Taipei").tz_localize(None)
    return min(end.floor("5min"), (last + pd.Timedelta(minutes=1)).floor("5min"))


def five_minute_series(rows: list[dict], end: pd.Timestamp, max_gap: int = 3) -> dict[str, pd.Series]:
    """每分鐘列 → 每台 VD 到 data_cutoff 為止的完整 5 分鐘車速；沒收齊的最後一桶不用。
    所有 VD 對齊到同一條時間軸，某台尾端沒資料就是 NaN（之後拒絕預測），不會拿舊資料當最新。"""
    if not rows:
        return {}
    cut = data_cutoff(rows, end)
    df = pd.DataFrame(rows)
    out = {}
    for vdid, g in df.groupby("vdid"):
        s = aggregate_5min(g)["speed"]
        s = s[s.index < cut]
        if s.empty:
            continue
        s = s.reindex(pd.date_range(s.index.min(), cut - pd.Timedelta(minutes=5), freq="5min"))
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


def fetch_minutes(end: dt.datetime, keep, minutes: int = LOOKBACK_MIN, fetch=None, after: int = 0,
                  workers: int = 4) -> list[dict]:
    """抓 end 前 minutes 分鐘到 end 後 after 分鐘的每分鐘檔；不存在（還沒上傳）的分鐘略過。workers 不要設太高。"""
    from concurrent.futures import ThreadPoolExecutor
    from download_vd import DEFAULT_BASE, _fetch_parse, http_fetch
    fetch = fetch or http_fetch
    times = [end + dt.timedelta(minutes=k) for k in range(-minutes, after)]
    urls = [f"{DEFAULT_BASE}/{t:%Y%m%d}/VDLive_{t:%H%M}.xml.gz" for t in times]
    with ThreadPoolExecutor(workers) as ex:
        return [r for rows, _err in ex.map(lambda u: _fetch_parse(u, fetch, keep, retries=2, backoff=1.0), urls)
                for r in rows]


def taipei_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None, second=0, microsecond=0)


def predict_all(model, cfg: dict, stats: dict, vds: list[str], end: dt.datetime, fetch=None) -> dict:
    """抓資料並預測 vds 每一台；end 加 25 分鐘已經過去時（回測）一併附上實際值。網頁與指令列共用。
    輸入只用 end 之前的資料（回測多抓的未來資料只拿來對答案，不參與補值），和即時執行時看到的一樣。"""
    after = 25 if end + dt.timedelta(minutes=25) <= taipei_now() else 0
    rows = fetch_minutes(end, keep=set(vds).__contains__, after=after, fetch=fetch)
    t_end = pd.Timestamp(end)
    local = pd.to_datetime([r["time"] for r in rows], utc=True, format="ISO8601").tz_convert("Asia/Taipei").tz_localize(None)
    rows_in = [r for r, t in zip(rows, local) if t < t_end]
    series = five_minute_series(rows_in, t_end)
    cut = data_cutoff(rows_in, t_end)
    stale = not rows_in or t_end - cut > pd.Timedelta(minutes=STALE_MIN)
    actual = five_minute_series(rows, t_end + pd.Timedelta(minutes=after)) if after else {}
    out = []
    for v in vds:
        s = series.get(v, pd.Series(dtype=float))
        ok = len(s) and not np.isnan(s.iloc[-1])
        a = actual.get(v, pd.Series(dtype=float))
        preds = [] if stale else [
            {"time": f"{t:%H:%M}", "at": f"{t:%Y-%m-%d %H:%M}", "ahead_min": int((t - t_end) / pd.Timedelta(minutes=1)),
             "kmh": round(kmh, 1), "actual": None if np.isnan(a.get(t, np.nan)) else round(float(a[t]), 1)}
            for t, kmh in forecast(model, cfg, s, *stats[v])]
        out.append({"vdid": v, "last_time": f"{s.index[-1]:%H:%M}" if ok else None,
                    "last_kmh": round(float(s.iloc[-1]), 1) if ok else None, "preds": preds})
    return {"end": f"{end:%Y-%m-%d %H:%M}", "minutes": len({r["time"] for r in rows_in}), "backtest": bool(after),
            "data_until": f"{cut:%H:%M}", "data_age_min": int((t_end - cut) / pd.Timedelta(minutes=1)),
            "stale": bool(stale), "vds": out}


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
    end = dt.datetime.strptime(a.at, "%Y-%m-%d %H:%M") if a.at else taipei_now()
    r = predict_all(model, cfg, stats, vds, end)
    print(f"預測時間點 {r['end']}（台灣時間）；抓到 {r['minutes']} 分鐘的資料，完整資料到 {r['data_until']}")
    if r["stale"]:
        print(f"資料比預測時間點舊 {r['data_age_min']} 分鐘（超過 {STALE_MIN} 分鐘），不預測")
    print("每格是「該時刻開始的 5 分鐘」平均車速 km/h" + ("；括號內是實際值" if r["backtest"] else ""))
    print("| VD | 最近一段 | 預測 1 | 預測 2 | 預測 3 |\n|---|---|---|---|---|")
    for v in r["vds"]:
        last = f"{v['last_kmh']:.1f}（{v['last_time']}）" if v["last_kmh"] is not None else "–"
        cells = " | ".join(f"{q['time']} {q['kmh']:.1f}" + ("" if q["actual"] is None else f"（{q['actual']:.1f}）")
                           for q in v["preds"]) if v["preds"] else "資料不足 | – | –"
        print(f"| {v['vdid']} | {last} | {cells} |")


if __name__ == "__main__":
    main()
