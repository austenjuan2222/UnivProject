import glob, json, re, numpy as np
from scipy import stats
R = {}
for f in glob.glob("res/*.npz"):
    m = re.match(r"res/(processed_28d(?:_\d+)?)_(.+)_f(-?\d+)_s(\d+)\.npz", f.replace("\\", "/"))
    data, tag, fold, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    z = np.load(f); eps = json.loads(open(f[:-4] + ".jsonl").readline())["eps"]
    R[(data, tag, fold, seed)] = dict(err=z["err"], naive=z["naive"], sizes=z["sizes"], eps=eps)
def mae(r): return r["err"].mean()
def ci(x):
    x = np.asarray(x); n = len(x); m = x.mean()
    if n < 2: return m, np.nan, np.nan, np.nan
    s = x.std(ddof=1); h = stats.t.ppf(.975, n - 1) * s / np.sqrt(n); return m, s, m - h, m + h
def table(data, fold, tags, base):
    seeds = sorted({k[3] for k in R if k[0] == data and k[2] == fold and k[1] == base})
    print(f"\n== {data} fold {fold} seeds {seeds} ==")
    r0 = R[(data, base, fold, seeds[0])]
    print(f"test samples {len(r0['err'])}, naive MAE {r0['naive'].mean():.3f}")
    for t in tags:
        ss = [s for s in seeds if (data, t, fold, s) in R and (data, base, fold, s) in R]
        if not ss: continue
        v = [mae(R[(data, t, fold, s)]) for s in ss]; b = [mae(R[(data, base, fold, s)]) for s in ss]
        d = np.array(v) - np.array(b)
        m, sd, lo, hi = ci(v); dm, dsd, dlo, dhi = ci(d)
        p = stats.ttest_rel(v, b).pvalue if len(ss) > 1 and t != base else np.nan
        eps = R[(data, t, fold, ss[0])]["eps"]
        print(f"{t:10s} n={len(ss):2d} ε={eps:7.2f} MAE {m:.3f} (SD {sd:.3f}, 95%CI {lo:.3f}–{hi:.3f}) | Δ vs {base} {dm:+.3f} [{dlo:+.3f},{dhi:+.3f}] p={p:.3g}")
tags = ["sg-1", "sg0", "sg0.5", "sg1.1", "sg3", "sg8", "sg20"]
table("processed_28d", -1, tags, "sg-1")
table("processed_28d", -1, ["ft0sg-1", "ft0sg1.1"], "ft0sg-1")
table("processed_28d_32", -1, ["sg-1", "sg1.1", "sg3"], "sg-1")
print("\n== block CV (8 VD): per-fold mean over seeds ==")
for fo in range(5):
    row = []
    for t in ["sg-1", "sg1.1", "sg3"]:
        v = [mae(R[k]) for k in R if k[0] == "processed_28d" and k[1] == t and k[2] == fo]
        row.append(np.mean(v) if v else np.nan)
    nv = [R[k]["naive"].mean() for k in R if k[0] == "processed_28d" and k[2] == fo][:1]
    print(f"fold {fo}: naive {nv[0] if nv else np.nan:.3f} nodp {row[0]:.3f} σ1.1 {row[1]:.3f} (Δ{row[1]-row[0]:+.3f}) σ3 {row[2]:.3f} (Δ{row[2]-row[0]:+.3f})")
# test-period uncertainty: day-block bootstrap of the seed-averaged per-sample error difference
def blocks(data, a, b, fold=-1, L=288, B=5000):
    ss = sorted({k[3] for k in R if k[0] == data and k[1] == a and k[2] == fold} & {k[3] for k in R if k[0] == data and k[1] == b and k[2] == fold})
    ea = np.mean([R[(data, a, fold, s)]["err"] for s in ss], 0); eb = np.mean([R[(data, b, fold, s)]["err"] for s in ss], 0)
    sizes = R[(data, a, fold, ss[0])]["sizes"]; d = ea - eb
    # per-VD consecutive blocks of L samples (≈1 day); same block index across VDs = same calendar day
    parts = np.split(d, np.cumsum(sizes)[:-1]); nb = min(len(p) for p in parts) // L
    day = np.array([[p[j*L:(j+1)*L].mean() for j in range(nb)] for p in parts]).mean(0)  # per-day diff
    rng = np.random.default_rng(0); boot = [day[rng.integers(0, nb, nb)].mean() for _ in range(B)]
    nbin = np.array([[p[j*L:(j+1)*L].mean() for j in range(nb)] for p in parts])
    # lag-1 autocorrelation of per-sample diff (shows samples are not independent)
    ac = np.mean([np.corrcoef(p[:-1], p[1:])[0, 1] for p in parts])
    neff = len(d) * (1 - ac) / (1 + ac)
    print(f"{data} {a} vs {b}: {nb} day-blocks, per-day Δ {np.round(day,3)}, mean {day.mean():+.4f}, day-bootstrap 95% CI [{np.percentile(boot,2.5):+.4f},{np.percentile(boot,97.5):+.4f}]; lag1 autocorr {ac:.2f} → effective n≈{neff:.0f} of {len(d)}")
print("\n== test-period (day-block) uncertainty ==")
for a in ["sg1.1", "sg3", "sg8", "sg20", "sg0"]:
    blocks("processed_28d", a, "sg-1")
blocks("processed_28d", "sg-1", "sg-1")  # sanity
# naive vs model
ss = sorted({k[3] for k in R if k[0]=="processed_28d" and k[1]=="sg-1" and k[2]==-1})
print("\nper-VD MAE (nodp, seed-avg):", np.round([np.mean([np.split(R[('processed_28d','sg-1',-1,s)]['err'], np.cumsum(R[('processed_28d','sg-1',-1,s)]['sizes'])[:-1])[i].mean() for s in ss]) for i in range(8)],2))
print("\n== noise cost alone (vs clip-only σ0) ==")
for t in ["sg0.5","sg1.1","sg3"]:
    v=np.array([mae(R[("processed_28d",t,-1,s)]) for s in range(1,11)]); b=np.array([mae(R[("processed_28d","sg0",-1,s)]) for s in range(1,11)])
    m,sd,lo,hi=ci(v-b); print(t, f"Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={stats.ttest_rel(v,b).pvalue:.3g} wilcoxon p={stats.wilcoxon(v,b).pvalue:.3g}")
v=np.array([mae(R[("processed_28d","sg1.1",-1,s)]) for s in range(1,11)]); b=np.array([mae(R[("processed_28d","sg-1",-1,s)]) for s in range(1,11)])
print("σ1.1 vs nodp wilcoxon p", stats.wilcoxon(v,b).pvalue)
from statsmodels.stats.power import TTestPower
