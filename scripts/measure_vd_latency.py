"""實測即時 VDLive 資料落後多久（第 49 頁限制第 5 項）。

每隔 --every 秒抓一次 https://tisvcloud.freeway.gov.tw/history/motc20/VDLive.xml，記錄：
- update_lag：抓到的時間 − 檔案的 UpdateTime；
- collect_lag：抓到的時間 − 各 VD 的 DataCollectTime（那一分鐘的起點）；
- bin5_lag：整理成 5 分鐘一格（以起點標記、該格最後一分鐘到了才算完整）後，抓到時最新完整那格已經結束多久。
結果每次一列寫進 CSV。只用標準函式庫。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_vd import http_fetch  # noqa: E402
from vd_parse import _children, _local, _text  # noqa: E402

LIVE_URL = "https://tisvcloud.freeway.gov.tw/history/motc20/VDLive.xml"


def _ts(s: str | None):
    try:
        t = dt.datetime.fromisoformat(s) if s else None
    except ValueError:
        return None
    # 沒標時區的當台灣時間；不然和 UTC 的抓取時間相減會丟 TypeError，整筆被吞掉
    return t.replace(tzinfo=dt.timezone(dt.timedelta(hours=8))) if t and t.tzinfo is None else t


def latency(data: bytes, now: dt.datetime, keep: set | None = None) -> dict:
    root = ET.fromstring(data)
    if _local(root.tag) != "VDLiveList":
        raise ValueError(f"不是 VDLiveList 文件（根節點為 {_local(root.tag)}）")
    upd = _ts(_text(root, "UpdateTime"))
    collect, bins = [], []
    for c in _children(root, "VDLives"):
        for v in _children(c, "VDLive"):
            if keep is not None and _text(v, "VDID", "") not in keep:
                continue
            t = _ts(_text(v, "DataCollectTime"))
            if t is None:
                continue
            collect.append(round((now - t).total_seconds()))
            end = t + dt.timedelta(minutes=1)
            end -= dt.timedelta(minutes=end.minute % 5, seconds=end.second, microseconds=end.microsecond)
            bins.append(round((now - end).total_seconds()))
    return {"update_lag_s": round((now - upd).total_seconds()) if upd else None,
            "n_vd": len(collect), "collect_lag_s": sorted(collect), "bin5_lag_s": sorted(bins)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--manifest", help="只算這份 manifest.json 裡的 VD（例如 data/processed_28d_32/manifest.json）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    keep = None
    if a.manifest:
        keep = {c["vdid"] for c in json.loads(Path(a.manifest).read_text(encoding="utf-8"))["clients"]}
    end = time.monotonic() + a.minutes * 60
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fetched_at", "update_lag_s", "n_vd", "collect_lag_med_s", "collect_lag_max_s",
                    "bin5_lag_med_s", "bin5_lag_max_s"])
        while time.monotonic() < end:
            try:
                data = http_fetch(LIVE_URL, timeout=60)
                now = dt.datetime.now(dt.timezone.utc)
                r = latency(data, now, keep)
                if r["n_vd"]:
                    w.writerow([now.isoformat(timespec="seconds"), r["update_lag_s"], r["n_vd"],
                                statistics.median(r["collect_lag_s"]), r["collect_lag_s"][-1],
                                statistics.median(r["bin5_lag_s"]), r["bin5_lag_s"][-1]])
                    fh.flush()
            except Exception as e:  # 網路斷線或檔案寫到一半：記下來、下一次再抓
                print("抓取失敗：", e, file=sys.stderr)
            time.sleep(a.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
