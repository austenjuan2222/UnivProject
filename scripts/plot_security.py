#!/usr/bin/env python3
"""攻擊與可用性實驗（32 個 VD、DP σ=1.1）：各情境的全域模型與微調後 MAE。

依 scientific-visualization skill：每個 seed 畫空心點、平均畫實心點，不用長條圖遮住分布；
x 軸用對數刻度，因為被攻擊時誤差可能大上好幾倍。虛線是 naive 基準。

用法：python scripts/plot_security.py [results 根目錄] [輸出 png]
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

# (標籤, batch, 分組)；分組決定顏色
SCENARIOS = [
    ("No security checks (earlier code)", "d32_fair", "base"),
    ("HMAC + model signature on", "s32_secure", "base"),
    ("25% of uploads tampered, no HMAC", "s32_tamper_noverify", "attack"),
    ("25% of uploads tampered, HMAC on", "s32_tamper_verify", "defended"),
    ("3 compromised VDs, FedAvg", "s32_atk_fedavg", "attack"),
    ("3 compromised VDs, median", "s32_atk_median", "defended"),
    ("3 compromised VDs, trimmed mean", "s32_atk_trimmed", "defended"),
    ("6 compromised VDs, median", "s32_atk6_median", "defended"),
    ("6 compromised VDs, trimmed mean", "s32_atk6_trimmed", "defended"),
    ("10 compromised VDs, median", "s32_atk10_median", "defended"),
    ("10 compromised VDs, trimmed mean", "s32_atk10_trimmed", "attack"),
    ("3 compromised VDs, server clipping", "s32_atk3_clip", "defended"),
    ("10 compromised VDs, server clipping", "s32_atk10_clip", "defended"),
    ("10 compromised VDs, clipping + median", "s32_atk10_clipmed", "defended"),
    ("14 compromised VDs, server clipping", "s32_atk14_clip", "attack"),
    ("14 compromised VDs, median", "s32_atk14_median", "attack"),
    ("No attack, server clipping", "s32_clip_noatk", "base"),
    ("3 VDs send random noise, FedAvg", "s32_atknoise_fedavg", "attack"),
    ("3 VDs send random noise, median", "s32_atknoise_median", "defended"),
    ("Secure aggregation (server sees only the sum)", "s32_secagg", "base"),
    ("No attack, median", "s32_median", "base"),
    ("No attack, trimmed mean", "s32_trimmed", "base"),
    ("30% of VDs drop out each round", "s32_drop30", "base"),
]
COLORS = {"base": "#0072B2", "attack": "#D55E00", "defended": "#009E73"}  # Okabe-Ito


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load(root: Path, batch: str, prefix: str = "dp_sigma1.1") -> list[dict]:
    rows = []
    for f in sorted((root / batch).glob("*/summary.csv")):
        with f.open(encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if r["run"].startswith(prefix)]
    return rows


def collect(root: Path) -> list[dict]:
    out = []
    for label, batch, group in SCENARIOS:
        rows = load(root, batch)
        if not rows:
            continue
        out.append({"label": label, "group": group, "n": len(rows),
                    "g": [v for v in (_num(r["final_mae_kmh"]) for r in rows) if v is not None],
                    "ft": [v for v in (_num(r["finetuned_mae_kmh"]) for r in rows) if v is not None],
                    "naive": _num(rows[0]["naive_mae_kmh"])})
    return out


def plot(data: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(data) + 1.6), dpi=200, sharey=True)
    ys = list(range(len(data)))[::-1]
    for ax, key, title in [(axes[0], "g", "Global model (shared with everyone)"),
                           (axes[1], "ft", "After local fine-tuning (each VD)")]:
        for y, d in zip(ys, data):
            c = COLORS[d["group"]]
            if d[key]:
                ax.scatter(d[key], [y] * len(d[key]), facecolors="none", edgecolors=c, s=22, linewidths=0.9)
                ax.scatter([statistics.fmean(d[key])], [y], color=c, s=50, zorder=3)
        ax.axvline(data[0]["naive"], color="#999999", linestyle="--", linewidth=1)
        ax.text(data[0]["naive"], -0.7, " naive", color="#777777", fontsize=7, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Test MAE (km/h, log scale), lower is better")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ys, [d["label"] for d in data], fontsize=8)
    handles = [plt.Line2D([], [], color=COLORS[g], marker="o", linestyle="", label=l)
               for g, l in (("base", "Normal operation"), ("attack", "Under attack, undefended"),
                            ("defended", "Under attack, defended"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("32 VDs, record-level DP σ=1.1: attacks and defences (open = each seed, filled = mean)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path("results")
    out = Path(argv[1]) if len(argv) > 1 else root / "security_32.png"
    data = collect(root)
    if not data:
        print(f"{root} 底下找不到結果")
        return 1
    plot(data, out)
    for d in data:
        g = f"{statistics.fmean(d['g']):.3f}" if d["g"] else "–"
        ft = f"{statistics.fmean(d['ft']):.3f}" if d["ft"] else "–"
        print(f"{d['label']:<40} n={d['n']}  global {g}  fine-tuned {ft}  (naive {d['naive']:.3f})")
    print(f"已輸出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
