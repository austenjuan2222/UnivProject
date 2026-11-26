"""Flower 結果的同 seed 配對比較：python flower_paired.py <results 批次資料夾>（讀其中 summary.csv）。"""
import csv, re, sys
from collections import defaultdict
import numpy as np
from scipy import stats

batch = sys.argv[1]
rows = list(csv.DictReader(open(f"{batch}/summary.csv", encoding="utf-8")))
d = defaultdict(dict)  # (config, metric) -> {seed: value}
eps = defaultdict(list)
for r in rows:
    m = re.match(r"(.+)_seed(\d+)$", r["run"])
    if not m:
        continue
    cfg, seed = m.group(1), int(m.group(2))
    d[(cfg, "global")][seed] = float(r["final_mae_kmh"])
    d[(cfg, "ft")][seed] = float(r["finetuned_mae_kmh"])
    if r.get("finetuned_epsilon"):
        eps[cfg].append(float(r["finetuned_epsilon"]))

def ci(x):
    x = np.asarray(x); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.std(ddof=1), x.mean() - h, x.mean() + h

cfgs = ["nodp", "dp_sigma0", "dp_sigma1.1", "dp_sigma3"]
print(f"batch {batch}")
for metric in ("global", "ft"):
    print(f"\n== {metric} MAE (km/h) ==")
    for c in cfgs:
        v = d[(c, metric)]; s = sorted(v)
        mu, sd, lo, hi = ci([v[k] for k in s])
        e = f" eps~{np.mean(eps[c]):.2f}" if eps[c] and c != "dp_sigma0" else ""
        line = f"{c:12s} n={len(s)} {mu:.3f} SD {sd:.3f} 95%CI [{lo:.3f},{hi:.3f}]{e}"
        for ref in ("dp_sigma0", "nodp"):
            if ref == c:
                continue
            r = d[(ref, metric)]; ks = sorted(set(v) & set(r))
            diff = np.array([v[k] - r[k] for k in ks])
            _, dsd, dlo, dhi = ci(diff)
            p = stats.ttest_rel([v[k] for k in ks], [r[k] for k in ks]).pvalue
            line += f" | vs {ref} {diff.mean():+.3f} [{dlo:+.3f},{dhi:+.3f}] p={p:.3g}"
        print(line)
