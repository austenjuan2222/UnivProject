#!/usr/bin/env python3
"""每分鐘 VD 資料 → 每個 client（VD）一個 5 分鐘 CSV + manifest.json。

範例：
  python scripts/preprocess_vd.py --num-clients 8 --vd-pattern "N1"
  python scripts/preprocess_vd.py --vd-ids VD-N1-S-10.000-M-LOOP VD-N1-S-12.300-M-LOOP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd_preprocess import preprocess  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--num-clients", type=int, default=8)
    p.add_argument("--vd-pattern", default="N1")
    p.add_argument("--vd-ids", nargs="*", help="直接指定 VD（優先於 pattern）")
    p.add_argument("--min-valid-ratio", type=float, default=0.8)
    p.add_argument("--max-gap", type=int, default=3, help="最多內插幾個連續 5 分鐘缺口")
    a = p.parse_args(argv)

    m = preprocess(a.raw, a.out, pattern=a.vd_pattern, n=a.num_clients, vd_ids=a.vd_ids,
                   min_valid_ratio=a.min_valid_ratio, max_gap=a.max_gap)
    print(f"期間 {m['start']} ~ {m['end']}（{m['num_steps']} 個 5 分鐘時段）")
    for c in m["clients"]:
        print(f"  client {c['partition']}: {c['vdid']}  有效率 {c['valid_ratio']:.1%}")
    print(f"已寫入 {Path(a.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
