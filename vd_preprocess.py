"""把每分鐘 VD 資料（download_vd.py 輸出）整理成每個 client 一個 5 分鐘 CSV。

流程：讀 raw CSV → 各 VD 5 分鐘聚合（流量加權車速）→ 對齊共同時間軸
→ 只補短缺口 → 依有效率挑 VD → 寫 <VD>.csv 與 manifest.json。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FREQ = "5min"
LOCAL_TZ = "Asia/Taipei"
OUTPUT_COLUMNS = ["time", "speed", "volume", "occupancy"]


def safe_name(vdid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", vdid)


def _to_local_naive(times: pd.Series) -> pd.Series:
    t = pd.to_datetime(times, utc=True, format="ISO8601")
    return t.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)


def aggregate_5min(df: pd.DataFrame, min_valid_minutes: int = 3) -> pd.DataFrame:
    """單一 VD 的每分鐘資料 → 5 分鐘。車速以流量加權（流量為 0 時取平均）。"""
    d = pd.DataFrame({
        "time": _to_local_naive(df["time"]),
        "speed": pd.to_numeric(df["speed"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0).clip(lower=0),
        "occupancy": pd.to_numeric(df["occupancy"], errors="coerce"),
    })
    d = d.drop_duplicates("time").set_index("time").sort_index()
    valid = d["speed"].notna()
    d["w_speed"] = d["speed"].where(valid, 0.0) * d["volume"].where(valid, 0.0)
    d["w"] = d["volume"].where(valid, 0.0)
    d["n_valid"] = valid.astype(int)

    g = d.resample(FREQ, label="left", closed="left")
    out = pd.DataFrame({
        "w_speed": g["w_speed"].sum(),
        "w": g["w"].sum(),
        "mean_speed": g["speed"].mean(),
        "n_valid": g["n_valid"].sum(),
        "volume": g["volume"].sum(),
        "occupancy": g["occupancy"].mean(),
    })
    speed = np.where(out["w"] > 0, out["w_speed"] / out["w"].where(out["w"] > 0, 1.0),
                     out["mean_speed"])
    speed = pd.Series(speed, index=out.index).where(out["n_valid"] >= min_valid_minutes)
    return pd.DataFrame({
        "speed": speed,
        "volume": out["volume"].where(out["n_valid"] >= min_valid_minutes),
        "occupancy": out["occupancy"].where(out["n_valid"] >= min_valid_minutes),
    })


def fill_short_gaps(s: pd.Series, max_gap: int = 3) -> pd.Series:
    """只內插長度 <= max_gap 且兩端都有值的缺口；長缺口與頭尾缺口保留 NaN。"""
    s = s.astype(float)
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    interp = s.interpolate(method="linear", limit_area="inside")
    fill_mask = isna & (run_len <= max_gap) & interp.notna()
    return s.where(~fill_mask, interp)


def select_vds(series: dict, pattern: str, n: int, min_valid_ratio: float = 0.8):
    """依有效率由高到低挑 n 個 VDID 符合 pattern 的 VD；回傳 [(vdid, ratio)]。"""
    rx = re.compile(pattern)
    ranked = sorted(
        ((vdid, float(s.notna().mean())) for vdid, s in series.items() if rx.search(vdid)),
        key=lambda x: (-x[1], x[0]),
    )
    ok = [r for r in ranked if r[1] >= min_valid_ratio]
    if len(ok) < n:
        raise ValueError(
            f"符合 '{pattern}' 且有效率 >= {min_valid_ratio:.0%} 的 VD 只有 {len(ok)} 個，"
            f"需要 {n} 個。請放寬 --min-valid-ratio、改 --vd-pattern 或減少 --num-clients。"
        )
    return ok[:n]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("vd_minute_*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir} 找不到 vd_minute_*.csv，請先執行 download_vd.py")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def preprocess(
    raw_dir,
    out_dir,
    pattern: str = "N1",
    n: int = 8,
    vd_ids: list[str] | None = None,
    min_valid_ratio: float = 0.8,
    max_gap: int = 3,
    min_valid_minutes: int = 3,
) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw = load_raw(raw_dir)
    if vd_ids:
        raw = raw[raw["vdid"].isin(vd_ids)]
    else:
        raw = raw[raw["vdid"].astype(str).str.contains(pattern, regex=True)]

    per_vd = {vdid: aggregate_5min(g, min_valid_minutes) for vdid, g in raw.groupby("vdid")}
    if not per_vd:
        raise ValueError("篩選後沒有任何 VD 資料")
    start = min(df.index.min() for df in per_vd.values())
    end = max(df.index.max() for df in per_vd.values())
    full_index = pd.date_range(start, end, freq=FREQ)

    filled = {}
    for vdid, df in per_vd.items():
        df = df.reindex(full_index)
        df["speed"] = fill_short_gaps(df["speed"], max_gap)
        filled[vdid] = df

    if vd_ids:
        missing = [v for v in vd_ids if v not in filled]
        if missing:
            raise ValueError(f"raw 資料中找不到這些 VD：{missing}")
        chosen = [(v, float(filled[v]["speed"].notna().mean())) for v in vd_ids]
    else:
        chosen = select_vds({k: v["speed"] for k, v in filled.items()}, ".", n, min_valid_ratio)
        chosen = sorted(chosen)  # 依 VDID（含里程）排序，partition 編號穩定

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    clients = []
    for i, (vdid, ratio) in enumerate(chosen):
        fname = f"{safe_name(vdid)}.csv"
        df = filled[vdid].copy()
        df.index.name = "time"
        df.reset_index()[OUTPUT_COLUMNS].to_csv(out_dir / fname, index=False, float_format="%.3f")
        clients.append({"partition": i, "vdid": vdid, "file": fname, "valid_ratio": round(ratio, 4)})

    manifest = {
        "freq_minutes": 5,
        "start": str(full_index[0]),
        "end": str(full_index[-1]),
        "num_steps": len(full_index),
        "selection": {"vd_ids": vd_ids, "pattern": pattern, "min_valid_ratio": min_valid_ratio,
                      "max_gap_steps": max_gap, "min_valid_minutes": min_valid_minutes},
        "clients": clients,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
