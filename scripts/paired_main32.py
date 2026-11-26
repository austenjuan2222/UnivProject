"""主表重跑的同 seed 配對差值：每組對它的同版對照（沒有安全檢查／無攻擊同聚合器），並和 seed 間差距比較。
用法：python scripts/paired_main32.py [results/night26/main32]
"""
import csv
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/night26/main32")
# 每組 → 對照組（同一版程式、同 seed）
CONTROL = {
    "hmac": "nocheck", "tamper_nover": "nocheck", "tamper_hmac": "nocheck", "replay_nover": "nocheck",
    "replay_hmac": "nocheck", "drop30": "nocheck", "a0_median": "nocheck", "a0_trimmed": "nocheck",
    "clip_a0": "nocheck", "a3_fedavg": "nocheck", "noise_fedavg": "nocheck",
    "a3_median": "a0_median", "a6_median": "a0_median", "a10_median": "a0_median", "a14_median": "a0_median",
    "noise_median": "a0_median", "a3_trimmed": "a0_trimmed", "a6_trimmed": "a0_trimmed", "a10_trimmed": "a0_trimmed",
    "clip_a3": "clip_a0", "clip_a10": "clip_a0", "clip_a14": "clip_a0", "clipmed_a10": "a10_median",
}


# 防禦之間直接比較（同攻擊、同 seed）
COMPARE = [("clip_a10", "a10_median"), ("clipmed_a10", "clip_a10"), ("a3_median", "a3_trimmed"),
           ("a6_median", "a6_trimmed"), ("clip_a3", "a3_median"), ("clip_a14", "a14_median"),
           ("noise_median", "noise_fedavg"), ("tamper_hmac", "tamper_nover"), ("replay_hmac", "replay_nover")]


def load(label):
    f = next((ROOT / label).rglob("summary.csv"), None)
    if f is None:
        return None
    rows = {r["run"]: r for r in csv.DictReader(f.open(encoding="utf-8"))}
    return {k: (float(r["final_mae_kmh"]), float(r["finetuned_mae_kmh"])) for k, r in sorted(rows.items())}


def paired(a, b):
    """回傳 (逐 seed 差值, 兩組 seed 間差距的較大者, 是否顯著：同號且每個差值都大於 seed 間差距)。"""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    spread = max(max(a[k] for k in keys) - min(a[k] for k in keys), max(b[k] for k in keys) - min(b[k] for k in keys))
    sig = len(d) == 3 and (all(x > 0 for x in d) or all(x < 0 for x in d)) and min(abs(x) for x in d) > spread
    return d, spread, sig


def main():
    print("| 組別 | 對照 | 全域 MAE 平均 | 逐 seed 配對差值 | seed 間差距 | 顯著？ | 微調後平均 |")
    print("|---|---|---|---|---|---|---|")
    for label in ["nocheck", *CONTROL]:
        r = load(label)
        if r is None:
            continue
        g = {k: v[0] for k, v in r.items()}
        ft = sum(v[1] for v in r.values()) / len(r)
        mean = sum(g.values()) / len(g)
        ctl = CONTROL.get(label)
        c = load(ctl) if ctl else None
        if c is None:
            print(f"| {label} | – | {mean:.3f} | – | – | – | {ft:.3f} |")
            continue
        d, spread, sig = paired(g, {k: v[0] for k, v in c.items()})
        print(f"| {label} | {ctl} | {mean:.3f} | {'／'.join(f'{x:+.3f}' for x in d)} | {spread:.3f} | "
              f"{'是' if sig else '否'} | {ft:.3f} |")
    print("\n| 比較 | 逐 seed 配對差值 | seed 間差距 | 顯著？ |\n|---|---|---|---|")
    for a, b in COMPARE:
        ra, rb = load(a), load(b)
        if ra and rb:
            d, spread, sig = paired({k: v[0] for k, v in ra.items()}, {k: v[0] for k, v in rb.items()})
            print(f"| {a} − {b} | {'／'.join(f'{x:+.3f}' for x in d)} | {spread:.3f} | {'是' if sig else '否'} |")


if __name__ == "__main__":
    main()
