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


def _partials(df: pd.DataFrame) -> pd.DataFrame:
    """每分鐘資料 → 每 (vdid, 5 分鐘桶) 的可加總量，跨檔案合併時直接相加即可。"""
    d = pd.DataFrame({
        "vdid": df["vdid"].astype(str).to_numpy(),
        "time": _to_local_naive(df["time"]).to_numpy(),
        "speed": pd.to_numeric(df["speed"], errors="coerce").to_numpy(),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0).clip(lower=0).to_numpy(),
        "occupancy": pd.to_numeric(df["occupancy"], errors="coerce").to_numpy(),
    }).drop_duplicates(["vdid", "time"])
    valid = d["speed"].notna()
    w = d["volume"].where(valid, 0.0)
    out = pd.DataFrame({
        "vdid": d["vdid"],
        "bucket": d["time"].dt.floor(FREQ),
        "w_speed": d["speed"].where(valid, 0.0) * w,
        "w": w,
        "speed_sum": d["speed"].where(valid, 0.0),
        "n_valid": valid.astype(int),
        "volume": d["volume"],
        "occ_sum": d["occupancy"].fillna(0.0),
        "occ_n": d["occupancy"].notna().astype(int),
    })
    return out.groupby(["vdid", "bucket"], sort=False).sum().reset_index()


def _finalize(part: pd.DataFrame, min_valid_minutes: int) -> pd.DataFrame:
    """單一 VD 的部分和 → 5 分鐘 speed/volume/occupancy（車速以流量加權，流量為 0 時取平均）。"""
    g = part.groupby("bucket").sum(numeric_only=True).sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq=FREQ)
    g = g.reindex(idx, fill_value=0)
    ok = g["n_valid"] >= min_valid_minutes
    mean_speed = g["speed_sum"] / g["n_valid"].where(g["n_valid"] > 0)
    speed = np.where(g["w"] > 0, g["w_speed"] / g["w"].where(g["w"] > 0, 1.0), mean_speed)
    occ = g["occ_sum"] / g["occ_n"].where(g["occ_n"] > 0)
    out = pd.DataFrame({
        "speed": pd.Series(speed, index=idx).where(ok),
        "volume": g["volume"].astype(float).where(ok),
        "occupancy": occ.where(ok),
    })
    out.index.name = "time"
    return out


def aggregate_5min(df: pd.DataFrame, min_valid_minutes: int = 3) -> pd.DataFrame:
    """單一 VD 的每分鐘資料 → 5 分鐘。"""
    return _finalize(_partials(df), min_valid_minutes)


def aggregate_files(files, pattern: str = "N1", vd_ids=None, min_valid_minutes: int = 3) -> dict:
    """逐檔讀取（記憶體只需一天份），合併部分和後回傳 {vdid: 5 分鐘 DataFrame}。"""
    rx = re.compile(pattern)
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=["vdid", "time", "speed", "volume", "occupancy"],
                         dtype={"vdid": str, "time": str})
        keep = df["vdid"].isin(vd_ids) if vd_ids else df["vdid"].map(lambda v: bool(rx.search(v)))
        df = df[keep]
        if len(df):
            parts.append(_partials(df))
        print(f"  已彙整 {Path(f).name}", flush=True)
    if not parts:
        return {}
    allp = pd.concat(parts, ignore_index=True)
    return {vdid: _finalize(g, min_valid_minutes) for vdid, g in allp.groupby("vdid", sort=True)}


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
    files = sorted(Path(raw_dir).glob("vd_minute_*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir} 找不到 vd_minute_*.csv，請先執行 download_vd.py")
    per_vd = aggregate_files(files, pattern=pattern, vd_ids=vd_ids,
                             min_valid_minutes=min_valid_minutes)
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
