#!/usr/bin/env python3
"""比較同一批實驗（一個資料夾底下多個 run）：輸出 comparison.png 與 summary.csv。

範例：python scripts/plot_results.py results/20261020-0300
左圖：各設定每輪測試 MAE（km/h）；右圖：DP 設定的累計 ε（最大 client）。
同一設定跑了多個 seed（run 名稱 *_seed<N>）時，圖上畫平均，summary_by_config.csv 給平均 ± 標準差。
"""
from __future__ import annotations

import csv
import json
import statistics
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
        with (d / "metrics.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        use_dp = bool(cfg.get("use-dp"))
        sigma = _f(cfg.get("noise-multiplier"))
        runs.append({
            "name": d.name, "use_dp": use_dp, "sigma": sigma,
            "label": f"DP σ={sigma:g}" if use_dp else "No DP (baseline)",
            "round": [int(r["round"]) for r in rows],
            "mae": [_f(r.get("mae_kmh")) for r in rows],
            "rmse": [_f(r.get("rmse_kmh")) for r in rows],
            "eps": [_f(r.get("epsilon_max")) for r in rows],
            "naive": [_f(r.get("naive_mae_kmh")) for r in rows],
        })
    runs.sort(key=lambda r: (r["use_dp"], -(r["sigma"] or 0), r["name"]))
    return runs


def _mean_curve(curves):
    """逐輪平均（略過缺值）；各 run 輪數不同時取最短。"""
    n = min(len(c) for c in curves)
    out = []
    for i in range(n):
        vals = [c[i] for c in curves if c[i] is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def group_runs(runs: list[dict]) -> list[dict]:
    """依 label（設定）合併多個 seed 的 run，保持 collect_runs 的順序。"""
    groups: dict[str, dict] = {}
    for r in runs:
        g = groups.setdefault(r["label"], {"label": r["label"], "use_dp": r["use_dp"],
                                           "sigma": r["sigma"], "members": []})
        g["members"].append(r)
    for g in groups.values():
        m = g["members"]
        g["n"] = len(m)
        g["round"] = m[0]["round"][:min(len(x["round"]) for x in m)]
        for k in ("mae", "rmse", "eps", "naive"):
            g[k] = _mean_curve([x.get(k, []) for x in m])
    return list(groups.values())


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
    for i, r in enumerate(group_runs(runs)):
        if r["n"] > 1:
            r = dict(r, label=f"{r['label']} (mean of {r['n']} seeds)")
        c = PALETTE[i % len(PALETTE)]
        pts = [(x, y) for x, y in zip(r["round"], r["mae"]) if y is not None]
        if pts:
            a1.plot(*zip(*pts), color=c, linewidth=2, marker="o", markersize=3, label=r["label"])
        pts = [(x, y) for x, y in zip(r["round"], r["eps"]) if y is not None]
        if r["use_dp"] and pts:
            a2.plot(*zip(*pts), color=c, linewidth=2, marker="o", markersize=3, label=r["label"])
            a2.annotate(f"ε={pts[-1][1]:.2f}", pts[-1], textcoords="offset points",
                        xytext=(4, 0), color=INK, fontsize=8, va="center")
    naive = next((g["naive"][-1] for g in group_runs(runs) if g["naive"] and g["naive"][-1]), None)
    if naive is not None:
        a1.axhline(naive, color=MUTED, linestyle="--", linewidth=1,
                   label=f"Naive: next = last ({naive:.2f})")
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
                    "best_mae_kmh", "final_epsilon", "naive_mae_kmh", "improve_vs_naive_pct"])
        for r in runs:
            maes = [m for m in r["mae"] if m is not None]
            last = lambda xs: xs[-1] if xs and xs[-1] is not None else ""  # noqa: E731
            naive, mae = last(r.get("naive", [])), last(r["mae"])
            improve = 100 * (naive - mae) / naive if naive != "" and mae != "" and naive else ""
            w.writerow([r["name"], r["label"], r["use_dp"], r["sigma"] if r["use_dp"] else "",
                        len(r["round"]), mae, last(r["rmse"]),
                        min(maes) if maes else "", last(r["eps"]) if r["use_dp"] else "",
                        naive, improve])


def write_group_summary(runs: list[dict], out: Path) -> None:
    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "", ""
        return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else "")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "use_dp", "sigma", "n_seeds", "final_mae_mean", "final_mae_std",
                    "final_rmse_mean", "final_epsilon_mean", "naive_mae_mean"])
        for g in group_runs(runs):
            last = [m["mae"][-1] if m["mae"] else None for m in g["members"]]
            mae_mean, mae_std = stats(last)
            rmse_mean, _ = stats([m["rmse"][-1] if m["rmse"] else None for m in g["members"]])
            eps_mean, _ = stats([m["eps"][-1] if m["eps"] else None for m in g["members"]])
            naive_mean, _ = stats([m["naive"][-1] if m.get("naive") else None for m in g["members"]])
            w.writerow([g["label"], g["use_dp"], g["sigma"] if g["use_dp"] else "", g["n"],
                        mae_mean, mae_std, rmse_mean, eps_mean if g["use_dp"] else "", naive_mean])


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
    write_group_summary(runs, root / "summary_by_config.csv")
    print(f"已輸出 {root / 'comparison.png'}、{root / 'summary.csv'} 與 {root / 'summary_by_config.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
