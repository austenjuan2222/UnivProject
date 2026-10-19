#!/usr/bin/env python3
"""比較同一批實驗（一個資料夾底下多個 run）：輸出 comparison.png 與 summary.csv。

範例：python scripts/plot_results.py results/20261020-0300
左圖：各 run 每輪測試 MAE（km/h）；右圖：DP run 的累計 ε（最大 client）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# 固定順序的分類色（對照組永遠是第 1 色，DP run 依 σ 由大到小取後續色）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8")).get("run_config", {})
        rows = list(csv.DictReader((d / "metrics.csv").open(encoding="utf-8")))
        use_dp = bool(cfg.get("use-dp"))
        sigma = _f(cfg.get("noise-multiplier"))
        runs.append({
            "name": d.name, "use_dp": use_dp, "sigma": sigma,
            "label": f"DP σ={sigma:g}" if use_dp else "No DP (baseline)",
            "round": [int(r["round"]) for r in rows],
            "mae": [_f(r.get("mae_kmh")) for r in rows],
            "rmse": [_f(r.get("rmse_kmh")) for r in rows],
            "eps": [_f(r.get("epsilon_max")) for r in rows],
        })
    runs.sort(key=lambda r: (r["use_dp"], -(r["sigma"] or 0), r["name"]))
    return runs


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("Round", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot(runs: list[dict], out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for i, r in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        pts = [(x, y) for x, y in zip(r["round"], r["mae"]) if y is not None]
        if pts:
            a1.plot(*zip(*pts), color=c, linewidth=2, marker="o", markersize=3, label=r["label"])
        pts = [(x, y) for x, y in zip(r["round"], r["eps"]) if y is not None]
        if r["use_dp"] and pts:
            a2.plot(*zip(*pts), color=c, linewidth=2, marker="o", markersize=3, label=r["label"])
            a2.annotate(f"ε={pts[-1][1]:.2f}", pts[-1], textcoords="offset points",
                        xytext=(4, 0), color=INK, fontsize=8, va="center")
    _style(a1, "Test MAE per round", "MAE (km/h)")
    _style(a2, "Cumulative privacy budget (max over clients)", "ε")
    a2.set_ylim(bottom=0)
    a1.legend(frameon=False, fontsize=8)
    if any(r["use_dp"] for r in runs):
        a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_summary(runs: list[dict], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "label", "use_dp", "sigma", "rounds", "final_mae_kmh", "final_rmse_kmh",
                    "best_mae_kmh", "final_epsilon"])
        for r in runs:
            maes = [m for m in r["mae"] if m is not None]
            last = lambda xs: xs[-1] if xs and xs[-1] is not None else ""  # noqa: E731
            w.writerow([r["name"], r["label"], r["use_dp"], r["sigma"] if r["use_dp"] else "",
                        len(r["round"]), last(r["mae"]), last(r["rmse"]),
                        min(maes) if maes else "", last(r["eps"]) if r["use_dp"] else ""])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    root = Path(argv[0])
    runs = collect_runs(root)
    if not runs:
        print(f"{root} 底下沒有任何含 metrics.csv 的 run")
        return 1
    plot(runs, root / "comparison.png")
    write_summary(runs, root / "summary.csv")
    print(f"已輸出 {root / 'comparison.png'} 與 {root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
