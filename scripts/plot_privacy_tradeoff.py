#!/usr/bin/env python3
"""256 個 VD：隱私預算 ε 與預測誤差的取捨圖（依 scientific-visualization skill 的原則）。

- 每個 seed 都畫原始點（空心），平均畫實心大點；不平滑、不內插。
- 顏色＋標記形狀雙重編碼（Okabe-Ito 色盲友善配色）。
- x 軸為對數 ε；虛線是無 DP 平均與 naive 基準。
- 左：全域模型（record-level 用聯邦訓練的 ε；client-level 用整個 VD 的 ε）。
  右：本地微調後（record-level 的 ε 含微調；client-level 微調只在本地使用，ε 仍指發布的全域模型）。

用法：python scripts/plot_privacy_tradeoff.py [results 根目錄] [輸出 png]
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

OKABE = {"record": "#0072B2", "client": "#D55E00", "both": "#009E73", "nodp": "#000000", "naive": "#999999"}

# (類別, 標籤, [(batch, run 名稱前綴)])；所有 run 都是 256 VD、28 天、時段特徵、預測 5 分鐘後
GROUPS = [
    ("record", "σ=1.1", [("d256", "dp_sigma1.1"), ("d256s", "dp_sigma1.1_seed")]),
    ("record", "σ=3", [("d256_rec3", "dp_sigma3_seed")]),
    ("client", "σ=8, all VDs", [("d256_cl", "cldp_sigma8"), ("d256s_cl", "cldp_sigma8_seed")]),
    ("client", "σ=4, 25%/round", [("d256_samp", "cldp_sigma4_seed")]),
    ("client", "σ=6, 25%/round", [("d256_samp6", "cldp_sigma6_seed")]),
    # 兩層：record σ1.1＋client σ8（裁剪 0.2）；x 軸畫整個 VD 的 ε（2.45），record ε 另見圖說
    ("both", "record σ1.1 + VD σ8", [("d256_bothfull", "cldp_sigma8"), ("d256_bothfull_s23", "cldp_sigma8_seed")]),
]
NODP = [("d256", "nodp"), ("d256s", "nodp_seed2"), ("d256s3", "nodp")]


def _rows(root: Path, batch: str, prefix: str):
    for f in sorted((root / batch).glob("*/summary.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["run"] == prefix or r["run"].startswith(prefix):
                    yield r


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def collect(root: Path):
    out = []
    for kind, label, srcs in GROUPS:
        rows = [r for b, p in srcs for r in _rows(root, b, p)]
        if not rows:
            continue
        eps_g = _num(rows[0]["final_epsilon"])
        eps_ft = eps_g if kind == "both" else (_num(rows[0].get("finetuned_epsilon")) or eps_g)
        out.append({"kind": kind, "label": label, "n": len(rows), "eps": eps_g, "eps_ft": eps_ft,
                    "g": [_num(r["final_mae_kmh"]) for r in rows],
                    "ft": [_num(r["finetuned_mae_kmh"]) for r in rows],
                    "naive": _num(rows[0]["naive_mae_kmh"])})
    nod = [r for b, p in NODP for r in _rows(root, b, p)]
    nodp = {"g": [_num(r["final_mae_kmh"]) for r in nod], "ft": [_num(r["finetuned_mae_kmh"]) for r in nod]}
    return out, nodp


def plot(groups, nodp, out: Path):
    naive = groups[0]["naive"]
    marker = {"record": "o", "client": "s", "both": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    for ax, key, epskey, title in [(axes[0], "g", "eps", "Global model"),
                                   (axes[1], "ft", "eps_ft", "After local fine-tuning")]:
        for kind in ("record", "client", "both"):
            gs = sorted((g for g in groups if g["kind"] == kind), key=lambda g: g[epskey])
            xs = [g[epskey] for g in gs]
            means = [statistics.fmean(v for v in g[key] if v is not None) for g in gs]
            ax.plot(xs, means, color=OKABE[kind], linewidth=1.2, alpha=0.6, zorder=1)
            for g, m in zip(gs, means):
                vals = [v for v in g[key] if v is not None]
                ax.scatter([g[epskey]] * len(vals), vals, facecolors="none", edgecolors=OKABE[kind],
                           marker=marker[kind], s=22, linewidths=0.9, zorder=2)
                ax.scatter([g[epskey]], [m], color=OKABE[kind], marker=marker[kind], s=55, zorder=3)
                if kind == "both" and key == "ft":
                    continue  # 和 client σ=8 同一個 ε，圖例已說明
                ax.annotate(f"{g['label']} (n={len(vals)})" if key == "g" else f"ε={g[epskey]:.2f}",
                            (g[epskey], m), textcoords="offset points", ha="right" if kind == "both" else "center",
                            xytext={"record": (0, -14), "client": (0, 9), "both": (-4, 12)}[kind],
                            fontsize=6.5, color=OKABE[kind])
        nd = [v for v in nodp[key] if v is not None]
        if nd:
            ax.axhline(statistics.fmean(nd), color=OKABE["nodp"], linestyle=":", linewidth=1,
                       label=f"No DP (mean of {len(nd)} seeds)")
        ax.axhline(naive, color=OKABE["naive"], linestyle="--", linewidth=1, label=f"Naive: next = last ({naive:.3f})")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3], ["0.5", "0.75", "1", "1.5", "2", "3"], fontsize=7.5)
        ax.set_xlim(0.5, 3.2)
        ax.minorticks_off()
        ax.set_xlabel("Privacy budget ε (log scale, δ=1e-5; smaller = stronger)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Test MAE (km/h), lower is better")
    handles = [plt.Line2D([], [], color=OKABE["record"], marker="o", linestyle="-",
                          label="Record-level DP (protects single measurements)"),
               plt.Line2D([], [], color=OKABE["client"], marker="s", linestyle="-",
                          label="Client-level DP (protects a whole VD)"),
               plt.Line2D([], [], color=OKABE["both"], marker="D", linestyle="",
                          label="Both layers (x = whole-VD ε; record ε 2.54)")]
    h2, l2 = axes[0].get_legend_handles_labels()
    fig.legend(handles + h2, [h.get_label() for h in handles] + l2, loc="lower center", ncol=2,
               frameon=False, fontsize=7.5)
    fig.suptitle("256 VDs, 28 days, 5-min horizon: privacy vs accuracy (open = each seed, filled = mean)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path("results")
    out = Path(argv[1]) if len(argv) > 1 else root / "privacy_tradeoff_256.png"
    groups, nodp = collect(root)
    if not groups:
        print(f"{root} 底下找不到 256 VD 的結果")
        return 1
    plot(groups, nodp, out)
    print(f"已輸出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
