#!/usr/bin/env python3
"""下載高公局歷史 VD 動態資訊（MOTC v2.0，每分鐘一檔），邊下載邊解析，只存 CSV。

來源：https://tisvcloud.freeway.gov.tw/history/motc20/VD/YYYYMMDD/VDLive_HHMM.xml.gz
輸出：data/raw/vd_minute_YYYYMMDD.csv（vdid,time,speed,volume,occupancy,status）
      完成的日期會有 .done 標記，重跑時自動跳過（可續傳）。

範例：
  python scripts/download_vd.py --start 2024-03-04 --days 14 --vd-pattern "N1"
資料授權：政府資料開放授權條款－第 1 版。請勿把 --workers 設太高。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd_parse import FIELDS, parse_vdlive_xml  # noqa: E402

DEFAULT_BASE = "https://tisvcloud.freeway.gov.tw/history/motc20/VD"


def day_urls(day: dt.date, base: str = DEFAULT_BASE) -> list[str]:
    ymd = day.strftime("%Y%m%d")
    return [f"{base}/{ymd}/VDLive_{h:02d}{m:02d}.xml.gz" for h in range(24) for m in range(60)]


def http_fetch(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "UnivProject-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_parse(url, fetch, keep, retries, backoff):
    last = None
    for attempt in range(retries):
        try:
            return parse_vdlive_xml(fetch(url), keep=keep), None
        except urllib.error.HTTPError as e:
            if e.code == 404:  # 該分鐘檔案不存在，不必重試
                return [], "HTTP 404"
            last = e
        except Exception as e:  # noqa: BLE001 - 網路/解析錯誤都記錄後重試
            last = e
        if attempt + 1 < retries:
            time.sleep(backoff * (attempt + 1))
    return [], repr(last)


def download_day(
    day: dt.date,
    out_dir: Path,
    fetch=http_fetch,
    keep=None,
    base: str = DEFAULT_BASE,
    workers: int = 4,
    retries: int = 3,
    backoff: float = 2.0,
    max_failed: int = 30,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ymd = day.strftime("%Y%m%d")
    out_csv = out_dir / f"vd_minute_{ymd}.csv"
    done = out_dir / f"vd_minute_{ymd}.done"
    if done.exists() and out_csv.exists():
        return {"day": ymd, "skipped": True, "failed": [], "rows": None}

    urls = day_urls(day, base)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda u: _fetch_parse(u, fetch, keep, retries, backoff), urls))

    rows, failed = [], []
    for url, (r, err) in zip(urls, results):
        if err is not None:
            failed.append(re.search(r"VDLive_(\d{4})", url).group(1))
        rows.extend(r)
    rows.sort(key=lambda r: (r["time"], r["vdid"]))

    tmp = out_csv.with_suffix(".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out_csv)
    if len(failed) <= max_failed:
        done.write_text(f"failed_minutes={len(failed)}\n", encoding="utf-8")
    return {"day": ymd, "skipped": False, "failed": failed, "rows": len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--vd-pattern", default="N1", help="只保留 VDID 符合此正規表示式的 VD（空字串=全部）")
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-failed", type=int, default=30, help="單日缺檔超過此數不標記完成，下次重抓")
    a = p.parse_args(argv)

    start = dt.date.fromisoformat(a.start)
    rx = re.compile(a.vd_pattern) if a.vd_pattern else None
    keep = (lambda v: bool(rx.search(v))) if rx else None
    bad_days = 0
    for i in range(a.days):
        day = start + dt.timedelta(days=i)
        res = download_day(day, Path(a.out), keep=keep, base=a.base_url,
                           workers=a.workers, max_failed=a.max_failed)
        if res["skipped"]:
            print(f"[{res['day']}] 已完成，跳過")
            continue
        print(f"[{res['day']}] {res['rows']} 列，缺檔 {len(res['failed'])} 分鐘")
        if len(res["failed"]) > a.max_failed:
            bad_days += 1
            print(f"  ⚠ 缺檔過多，未標記完成，可重跑本指令補抓。前幾筆：{res['failed'][:5]}")
    return 1 if bad_days else 0


if __name__ == "__main__":
    sys.exit(main())
