"""主表重跑的同 seed 配對差值：每組對它的同版對照（沒有安全檢查／無攻擊同聚合器），配對 t 檢定、整張表 Holm 校正後 p < 0.05 才算顯著。
用法：python scripts/paired_main32.py [結果資料夾 ...]（多個資料夾時同組別的 seed 合併，例如 seed 1～3 和補跑的 4～10）
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from seed_stats import paired_summary  # noqa: E402

ROOTS = [Path(a) for a in sys.argv[1:]] or [Path("results/night26/main32")]
# 每組 → 對照組（同一版程式、同 seed）
CONTROL = {
    "hmac": "nocheck", "tamper_nover": "nocheck", "tamper_hmac": "nocheck", "replay_nover": "nocheck",
    "replay_hmac": "nocheck", "drop30": "nocheck", "a0_median": "nocheck", "a0_trimmed": "nocheck",
    "clip_a0": "nocheck", "a3_fedavg": "nocheck", "noise_fedavg": "nocheck",
    "a3_median": "a0_median", "a6_median": "a0_median", "a10_median": "a0_median", "a14_median": "a0_median",
    "noise_median": "a0_median", "a3_trimmed": "a0_trimmed", "a6_trimmed": "a0_trimmed", "a10_trimmed": "a0_trimmed",
    "clip_a3": "clip_a0", "clip_a10": "clip_a0", "clip_a14": "clip_a0", "clipmed_a10": "a10_median",
    "alldef_a0": "nocheck", "alldef_attack": "alldef_a0",
}


# 防禦之間直接比較（同攻擊、同 seed）
COMPARE = [("clip_a10", "a10_median"), ("clipmed_a10", "clip_a10"), ("a3_median", "a3_trimmed"),
           ("a6_median", "a6_trimmed"), ("clip_a3", "a3_median"), ("clip_a14", "a14_median"),
           ("noise_median", "noise_fedavg"), ("tamper_hmac", "tamper_nover"), ("replay_hmac", "replay_nover")]


def load(label):
    rows = {r["run"]: r for root in ROOTS for f in sorted((root / label).rglob("summary.csv"))
            for r in csv.DictReader(f.open(encoding="utf-8"))}
    if not rows:
        return None
    return {k: (float(r["final_mae_kmh"]), float(r["finetuned_mae_kmh"])) for k, r in sorted(rows.items())}


def paired(a, b):
    """a − b 的逐 seed 差值，和 seed_stats.paired_summary 的結果（平均差、95% CI、配對 t 的 p_t）。"""
    keys = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in keys], paired_summary([(k, b[k], a[k]) for k in keys])


def holm(ps: list[float]) -> list[float]:
    """Holm 校正：整張主表一起做 30 幾個檢定，只看未校正 p < 0.05 會有偽陽性。NaN（沒有差值）不算進檢定數。"""
    idx = sorted((i for i, p in enumerate(ps) if not math.isnan(p)), key=lambda i: ps[i])
    out, run = [float("nan")] * len(ps), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - rank) * ps[i]))
        out[i] = run
    return out


def verdict(p: float, p_holm: float) -> str:
    """以 Holm 校正後的 p 判定；只有未校正 p < 0.05 的另外標出來，不算顯著。"""
    if math.isnan(p):
        return "–"
    return "是" if p_holm < 0.05 else ("否（未校正 p < 0.05）" if p < 0.05 else "否")


def _cells(d, s, p_holm) -> str:
    diffs = "／".join(f"{x:+.3f}" for x in d) if len(d) <= 3 else f"n = {len(d)}"
    return (f"{diffs} | {s['mean_diff']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] | "
            f"{s['p_t']:.3f} | {p_holm:.3f} | {verdict(s['p_t'], p_holm)}")


def _g(r):
    return {k: v[0] for k, v in r.items()}


def main():
    rows, tests = [], []  # 先算完所有檢定，才能一起做 Holm 校正
    for label in ["nocheck", *CONTROL]:
        r = load(label)
        if r is None:
            continue
        ctl = CONTROL.get(label)
        c = load(ctl) if ctl else None
        rows.append((label, ctl if c else None, r, len(tests) if c else None))
        if c:
            tests.append(paired(_g(r), _g(c)))
    comps = []
    for a, b in COMPARE:
        ra, rb = load(a), load(b)
        if ra and rb:
            comps.append((a, b, len(tests)))
            tests.append(paired(_g(ra), _g(rb)))
    adj = holm([s["p_t"] for _, s in tests])
    print("Holm 校正的檢定數：", sum(not math.isnan(x) for x in adj))
    print("| 組別 | 對照 | 全域 MAE 平均 | 逐 seed 配對差值 | 平均差 [95% CI] | 配對 t p | Holm 校正 p | 顯著？ | 微調後平均 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, ctl, r, t in rows:
        mean = sum(_g(r).values()) / len(r)
        ft = sum(v[1] for v in r.values()) / len(r)
        cells = "– | – | – | – | –" if t is None else _cells(*tests[t], adj[t])
        print(f"| {label} | {ctl or '–'} | {mean:.3f} | {cells} | {ft:.3f} |")
    print("\n| 比較 | 逐 seed 配對差值 | 平均差 [95% CI] | 配對 t p | Holm 校正 p | 顯著？ |\n|---|---|---|---|---|---|")
    for a, b, t in comps:
        print(f"| {a} − {b} | {_cells(*tests[t], adj[t])} |")


if __name__ == "__main__":
    main()
