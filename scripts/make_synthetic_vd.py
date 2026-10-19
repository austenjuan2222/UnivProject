#!/usr/bin/env python3
"""產生「合成」的每分鐘 VD 資料（格式同 download_vd.py 輸出），只供離線測試與煙霧測試。

⚠ 這不是真實資料，請勿用於報告結果。
範例：python scripts/make_synthetic_vd.py --out data/raw_synthetic --days 3 --num-vds 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(out: Path, days: int = 3, num_vds: int = 4, seed: int = 0, start="2024-03-04"):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.date_range(start, periods=days * 1440, freq="1min", tz="Asia/Taipei")
    hour = t.hour + t.minute / 60
    rush = np.exp(-((hour - 8) ** 2) / 1.5) + np.exp(-((hour - 18) ** 2) / 2.0)
    frames = []
    for k in range(num_vds):
        base = 95 - 5 * k
        speed = base - (35 + 5 * k) * rush + rng.normal(0, 4, len(t))
        volume = rng.poisson(15 + 40 * rush)
        df = pd.DataFrame({
            "vdid": f"VD-N1-S-{10 + 2.5 * k:.3f}-M-SYN",
            "time": [x.isoformat() for x in t],
            "speed": np.round(np.clip(speed, 5, 130), 1),
            "volume": volume,
            "occupancy": np.round(np.clip(5 + 30 * rush + rng.normal(0, 2, len(t)), 0, 100), 1),
            "status": 0,
        })
        gaps = rng.choice(len(t), size=len(t) // 200, replace=False)  # 零星缺值
        df.loc[gaps, "speed"] = np.nan
        frames.append(df)
    # 一個資料很差的 VD，用來驗證選站邏輯會排除它
    bad = frames[0].copy()
    bad["vdid"] = "VD-N1-S-99.000-M-SYN"
    bad.loc[bad.index % 3 != 0, "speed"] = np.nan
    frames.append(bad)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["day"] = pd.to_datetime(all_rows["time"], format="ISO8601").map(lambda x: x.strftime("%Y%m%d"))
    for day, g in all_rows.groupby("day"):
        g.drop(columns="day").to_csv(out / f"vd_minute_{day}.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw_synthetic")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--num-vds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(f"合成資料已寫到 {make(Path(a.out), a.days, a.num_vds, a.seed)}")
