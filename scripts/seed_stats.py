#!/usr/bin/env python3
"""同一組 seed 配對比較兩個設定（依 statistical-analysis skill：配對 t 檢定＋Wilcoxon、效果量、95% CI）。

用法：
  python scripts/seed_stats.py results/d28_base results/d28_tod            # 兩批同名 run 配對
  python scripts/seed_stats.py results/d28_tod --within nodp dp_sigma1.1   # 同一批內兩設定依 seed 配對
選項：--metric finetuned_mae_kmh（預設）或 final_mae_kmh

差值 = B − A（MAE 越低越好，負值代表 B 比較好）。
估計對象：同一 seed 下兩設定的 MAE 差的平均；抽樣單位是 seed（n 很小，結果僅供參考）。
Wilcoxon 用精確分布；n=5 時雙尾 p 最小只能到 0.0625。
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

from scipy import stats


def load(batch: Path, metric: str) -> dict[str, float]:
    """讀一個 batch（或其下唯一的時間戳子資料夾）的 summary.csv：run 名稱 → 指標。"""
    files = [batch / "summary.csv"] if (batch / "summary.csv").exists() else sorted(batch.glob("*/summary.csv"))
    out = {}
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    v = float(r[metric])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(v):
                    out[r["run"]] = v
    return out


def _seed(name: str):
    m = re.search(r"_seed(\d+)$", name)
    return int(m.group(1)) if m else None


def pair_across(a: dict, b: dict) -> list[tuple[str, float, float]]:
    return [(k, a[k], b[k]) for k in sorted(a) if k in b]


def pair_within(d: dict, cfg_a: str, cfg_b: str) -> list[tuple[str, float, float]]:
    def by_seed(cfg):
        return {_seed(k): v for k, v in d.items() if k.rsplit("_seed", 1)[0] == cfg and _seed(k) is not None}
    sa, sb = by_seed(cfg_a), by_seed(cfg_b)
    return [(f"seed{s}", sa[s], sb[s]) for s in sorted(sa) if s in sb]


def paired_summary(pairs) -> dict:
    diffs = [b - a for _, a, b in pairs]
    n = len(diffs)
    if n < 2:
        raise ValueError(f"配對數只有 {n}，至少要 2 組")
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    if sd > 0:
        t_res = stats.ttest_rel([b for _, _, b in pairs], [a for _, a, _ in pairs])
        t, p_t = float(t_res.statistic), float(t_res.pvalue)
    else:
        t, p_t = float("nan"), float("nan")
    try:
        p_w = float(stats.wilcoxon(diffs, method="exact").pvalue) if any(diffs) else float("nan")
    except ValueError:
        p_w = float("nan")
    return {"n": n, "mean_a": statistics.fmean(a for _, a, _ in pairs),
            "mean_b": statistics.fmean(b for _, _, b in pairs),
            "mean_diff": mean, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
            "dz": mean / sd if sd > 0 else float("nan"),  # 帶正負號的 Cohen's d_z
            "t": t, "p_t": p_t, "p_wilcoxon": p_w,
            "n_b_better": sum(x < 0 for x in diffs)}


def fmt(s: dict, label_a: str, label_b: str, metric: str) -> str:
    return (f"{metric}：{label_a} {s['mean_a']:.3f} → {label_b} {s['mean_b']:.3f}（n={s['n']} 個 seed）\n"
            f"  差值 B−A = {s['mean_diff']:+.4f} km/h，95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]，"
            f"d_z = {s['dz']:+.2f}\n"
            f"  配對 t({s['n'] - 1}) = {s['t']:.2f}，p = {s['p_t']:.4f}；Wilcoxon 精確 p = {s['p_wilcoxon']:.4f}；"
            f"B 較好 {s['n_b_better']}/{s['n']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--within", nargs=2, metavar=("CFG_A", "CFG_B"))
    ap.add_argument("--metric", default="finetuned_mae_kmh")
    args = ap.parse_args(argv)
    da = load(Path(args.a), args.metric)
    if args.within:
        pairs, la, lb = pair_within(da, *args.within), *args.within
    elif args.b:
        pairs, la, lb = pair_across(da, load(Path(args.b), args.metric)), args.a, args.b
    else:
        ap.error("需要第二個 batch 或 --within")
    if args.b is None or args.within:
        print(fmt(paired_summary(pairs), la, lb, args.metric))
        return 0
    # 兩批比較：依設定（去掉 _seedN）分組各做一次
    groups: dict[str, list] = {}
    for k, a, b in pairs:
        groups.setdefault(k.rsplit("_seed", 1)[0], []).append((k, a, b))
    for cfg, ps in groups.items():
        if len(ps) >= 2:
            print(f"[{cfg}]", fmt(paired_summary(ps), la, lb, args.metric))
    return 0


if __name__ == "__main__":
    sys.exit(main())
