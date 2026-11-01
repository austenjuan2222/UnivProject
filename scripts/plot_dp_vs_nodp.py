#!/usr/bin/env python3
"""公平對照：record-level DP σ=1.1 減去「無 DP＋一般梯度裁剪」的 MAE 差，對 VD 數作圖。

依 scientific-visualization skill：每個 seed 的配對差畫空心點、平均畫實心點，
誤差線是配對差平均的 95% t 信賴區間；零線以下代表 DP 比較好。x 軸是 VD 數（對數刻度）。
所有 run：28 天資料、時段特徵、20 輪；無 DP 用 nodp-grad-clip=1.0。

用法：python scripts/plot_dp_vs_nodp.py [results 根目錄] [輸出 png]
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402


def _within(batch, seeds):
    return [(s, (batch, f"nodp_seed{s}"), (batch, f"dp_sigma1.1_seed{s}")) for s in seeds]


# (VD 數, 預測分鐘) → [(seed, 無 DP 來源, DP 來源)]；來源 = (batch, run 名稱)
PAIRS = {
    (8, 5): _within("d8_fair", (1, 2, 3)) + _within("d8_fair_s45", (4, 5)),
    (8, 30): _within("d8_h6_fair", (1, 2, 3)) + _within("d8_h6_fair_s45", (4, 5)),
    (32, 5): _within("d32_fair", (1, 2, 3)),
    (32, 30): _within("d32_h6_fair", (1, 2, 3)) + _within("d32_h6_fair_s45", (4, 5)),
    (64, 5): _within("d64_fair", (1, 2, 3)),
    (64, 30): _within("d64_h6_fair", (1, 2, 3)) + _within("d64_h6_fair_s45", (4, 5)),
    (128, 5): [(1, ("d128_nodpclip", "nodp_seed1"), ("d128_rec", "dp_sigma1.1_seed1")),
               (2, ("d128_nodpclip", "nodp_seed2"), ("d128_rec", "dp_sigma1.1_seed2")),
               (3, ("d128_nodpclip", "nodp_seed3"), ("m128_rec", "dp_sigma1.1_seed3")),
               (4, ("d128_nodpclip", "nodp_seed4"), ("m128_rec", "dp_sigma1.1_seed4"))],
    (128, 30): _within("d128_h6_nodpclip", (1, 2)) + _within("d128_h6_nodpclip_s34", (3, 4)),
    (256, 5): [(1, ("d256_nodpclip", "nodp_seed1"), ("d256", "dp_sigma1.1")),
               (2, ("d256_nodpclip", "nodp_seed2"), ("d256s", "dp_sigma1.1_seed2")),
               (3, ("d256_s3more", "nodp"), ("d256s", "dp_sigma1.1_seed3"))],
    (8, 15): _within("d8_h3_fair", (1, 2, 3)) + _within("d8_h3_fair_s45", (4, 5)),
    (32, 15): _within("d32_h3_fair", (1, 2, 3)) + _within("d32_h3_fair_s45", (4, 5)),
    (64, 15): _within("d64_h3_fair", (1, 2, 3)),
    (128, 15): _within("d128_h3_fair", (1, 2, 3)),
    (256, 15): [(1, ("d256_h3_nodpclip", "nodp"), ("d256_h3", "dp_sigma1.1")),
                (2, ("d256_h3_s2", "nodp"), ("d256_h3_s2", "dp_sigma1.1")),
                (3, ("d256_h3_s3", "nodp"), ("d256_h3_s3", "dp_sigma1.1"))],
    (256, 30): [(1, ("d256_h6_nodpclip", "nodp_seed1"), ("d256_h6", "dp_sigma1.1")),
                (2, ("d256_h6_nodpclip", "nodp_seed2"), ("d256_h6s2", "dp_sigma1.1")),
                (3, ("d256_h6_s3", "nodp"), ("d256_h6_s3", "dp_sigma1.1"))],
}
STYLE = {5: ("#0072B2", "o", "5-min horizon"), 15: ("#009E73", "^", "15-min horizon"),
         30: ("#D55E00", "s", "30-min horizon")}  # Okabe-Ito


def _table(root: Path, batch: str) -> dict:
    out = {}
    for f in sorted((root / batch).glob("*/summary.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[r["run"]] = r
    return out


def collect(root: Path, col: str) -> dict:
    cache, out = {}, {}
    for key, pairs in PAIRS.items():
        diffs = []
        for _, (bn, rn), (bd, rd) in pairs:
            tn = cache.setdefault(bn, _table(root, bn))
            td = cache.setdefault(bd, _table(root, bd))
            if rn in tn and rd in td:
                a, b = float(tn[rn][col]), float(td[rd][col])
                if math.isfinite(a) and math.isfinite(b):
                    diffs.append(b - a)
        if len(diffs) >= 2:
            out[key] = diffs
    return out


def _ci(d):
    m, se = statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h


def plot(root: Path, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    summary = {}
    for ax, col, title in [(axes[0], "final_mae_kmh", "Global model"),
                           (axes[1], "finetuned_mae_kmh", "After local fine-tuning")]:
        data = collect(root, col)
        for h, (color, marker, label) in STYLE.items():
            keys = sorted(k for k in data if k[1] == h)
            if not keys:
                continue
            shift = {5: 0.92, 15: 1.0, 30: 1.08}[h]  # 只為避免重疊（對數軸上的小位移）
            xs = [k[0] * shift for k in keys]
            cis = [_ci(data[k]) for k in keys]
            means = [c[0] for c in cis]
            ax.errorbar(xs, means, yerr=[[m - lo for m, lo, _ in cis], [hi - m for m, _, hi in cis]],
                        color=color, marker=marker, markersize=6, linewidth=1.3, capsize=3, label=label, zorder=3)
            for x, k in zip(xs, keys):
                ax.scatter([x] * len(data[k]), data[k], facecolors="none", edgecolors=color, marker=marker,
                           s=18, linewidths=0.8, zorder=2)
            summary[(col, h)] = {k[0]: c for k, c in zip(keys, cis)}
        ax.axhline(0, color="#777777", linewidth=1)
        ax.text(0.99, 0.98, "above 0: DP worse", transform=ax.transAxes, fontsize=7, color="#777777",
                ha="right", va="top")
        ax.text(0.99, 0.02 if ax is axes[1] else 0.30, "below 0: DP better", transform=ax.transAxes,
                fontsize=7, color="#777777", ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([8, 32, 64, 128, 256], ["8", "32", "64", "128", "256"])
        ax.minorticks_off()
        ax.set_xlabel("Number of VDs (clients)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("MAE difference, DP σ=1.1 − no DP (km/h)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Fair comparison vs no-DP with gradient clipping (open = each seed, bars = 95% CI of paired mean)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path("results")
    out = Path(argv[1]) if len(argv) > 1 else root / "dp_vs_nodp.png"
    summary = plot(root, out)
    for (col, h), rows in summary.items():
        for n, (m, lo, hi) in sorted(rows.items()):
            print(f"{col} {h}min {n} VD: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"已輸出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
