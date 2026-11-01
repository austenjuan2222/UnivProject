#!/usr/bin/env python3
"""256 個 VD：各預測距離（5／15／30 分鐘）下，各設定比 naive 好多少（%）。

依 scientific-visualization skill：每個 seed 畫空心點、平均畫實心點，顏色＋形狀雙重編碼（Okabe-Ito），
y 軸是相對各自 naive 的改善百分比（不同預測距離的 MAE 尺度不同，不能直接比 km/h）。
無 DP 用「一般梯度裁剪」的版本（nodp-grad-clip=1.0）：沒有裁剪的無 DP 在 30 分鐘 3/3 seed 都發散。

用法：python scripts/plot_horizon_summary.py [results 根目錄] [輸出 png]
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_privacy_tradeoff import _num, _rows  # noqa: E402

# 設定 → 預測距離（分鐘）→ [(batch, run 名稱前綴)]
SOURCES = {
    "No DP + grad clip (not private)": {
        5: [("d256_nodpclip", "nodp"), ("d256_s3more", "nodp")],
        15: [("d256_h3_nodpclip", "nodp"), ("d256_h3_s2", "nodp")],
        30: [("d256_h6_nodpclip", "nodp")],
    },
    "Record σ=1.1 (ε≈2.5)": {
        5: [("d256", "dp_sigma1.1"), ("d256s", "dp_sigma1.1_seed")],
        15: [("d256_h3", "dp_sigma1.1"), ("d256_h3_s2", "dp_sigma1.1")],
        30: [("d256_h6", "dp_sigma1.1"), ("d256_h6s2", "dp_sigma1.1")],
    },
    "Record σ=3 (ε≈0.65)": {
        5: [("d256_rec3", "dp_sigma3"), ("d256_s3more", "dp_sigma3")],
        15: [("d256_h3", "dp_sigma3"), ("d256_h3_s2", "dp_sigma3")],
        30: [("d256_h6_rec3", "dp_sigma3")],
    },
    "Whole-VD ε=0.79 (25% VDs/round)": {
        5: [("d256_samp6", "cldp_sigma6"), ("d256_samp6_s3", "cldp_sigma6")],
        15: [("d256_h3cl", "cldp_sigma6"), ("d256_h3cl_s2", "cldp_sigma6")],
        30: [("d256_h6cl", "cldp_sigma6"), ("d256_h6cls2", "cldp_sigma6"), ("d256_h6cl_s3", "cldp_sigma6")],
    },
    "Both: record σ=1.1 + whole-VD ε=2.45": {
        5: [("d256_bothfull", "cldp_sigma8"), ("d256_bothfull_s23", "cldp_sigma8")],
        30: [("d256_h6_both", "cldp_sigma8"), ("d256_h6_both_s2", "cldp_sigma8")],
    },
}
STYLE = {  # Okabe-Ito
    "No DP + grad clip (not private)": ("#000000", "o"),
    "Record σ=1.1 (ε≈2.5)": ("#0072B2", "s"),
    "Record σ=3 (ε≈0.65)": ("#56B4E9", "^"),
    "Whole-VD ε=0.79 (25% VDs/round)": ("#D55E00", "D"),
    "Both: record σ=1.1 + whole-VD ε=2.45": ("#009E73", "v"),
}


def collect(root: Path) -> dict:
    """設定 → 欄位（g／ft）→ [(分鐘, [各 seed 改善 %])]。"""
    out = {}
    for label, by_h in SOURCES.items():
        for h, srcs in by_h.items():
            rows = [r for b, p in srcs for r in _rows(root, b, p)]
            for key, col in (("g", "final_mae_kmh"), ("ft", "finetuned_mae_kmh")):
                vals = []
                for r in rows:
                    mae, naive = _num(r[col]), _num(r["naive_mae_kmh"])
                    if mae is not None and naive:
                        vals.append(100 * (naive - mae) / naive)
                if vals:
                    out.setdefault(label, {}).setdefault(key, []).append((h, vals))
    return out


def plot(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    offsets = {lab: (i - 2) * 0.5 for i, lab in enumerate(SOURCES)}  # 只為避免重疊，x 為類別位置
    for ax, key, title in [(axes[0], "g", "Global model"), (axes[1], "ft", "After local fine-tuning")]:
        for label, series in data.items():
            if key not in series:
                continue
            color, marker = STYLE[label]
            pts = sorted(series[key])
            xs = [h + offsets[label] for h, _ in pts]
            means = [statistics.fmean(v) for _, v in pts]  # noqa: F841
            # 缺的預測距離留空（NaN 斷線），不在中間內插
            by_h = dict(zip([h for h, _ in pts], means))
            line_x = [h + offsets[label] for h in (5, 15, 30)]
            line_y = [by_h.get(h, float("nan")) for h in (5, 15, 30)]
            ax.plot(line_x, line_y, color=color, marker=marker, markersize=6, linewidth=1.3, label=label, zorder=3)
            for x, (_, vals) in zip(xs, pts):
                ax.scatter([x] * len(vals), vals, facecolors="none", edgecolors=color, marker=marker,
                           s=18, linewidths=0.8, zorder=2)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
        ax.text(31.5, 0.3, "naive", color="#777777", fontsize=7, ha="right")
        ax.set_xticks([5, 15, 30], ["5 min", "15 min", "30 min"])
        ax.set_xlim(2, 33)
        ax.set_xlabel("Forecast horizon")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Improvement over naive (%), higher is better")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("256 VDs, 28 days: every DP setting beats naive at 5/15/30 min (open = each seed, filled = mean)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path("results")
    out = Path(argv[1]) if len(argv) > 1 else root / "horizon_summary_256.png"
    data = collect(root)
    if not data:
        print(f"{root} 底下找不到結果")
        return 1
    plot(data, out)
    for label, series in data.items():
        for key in ("g", "ft"):
            pts = ", ".join(f"{h}min {statistics.fmean(v):.1f}% (n={len(v)})" for h, v in sorted(series.get(key, [])))
            print(f"{label} [{key}]: {pts}")
    print(f"已輸出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
