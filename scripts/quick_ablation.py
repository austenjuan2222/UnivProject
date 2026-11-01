"""快速消融實驗：不經 Flower，在同一個行程內做 FedAvg + Opacus DP-SGD。

用途是大量比較設定（每個設定數個 seed 只要幾分鐘）；訓練流程與 client_app.py 相同
（逐 client 本地 1 epoch、Poisson 取樣、依樣本數加權平均），找到好的設定後再用
run_experiments.py 走 Flower 正式驗證。報告裡標「快速實驗」的數字都來自這支程式。

python scripts/quick_ablation.py --data-dir data/processed --name dp                # DP σ=1.1
python scripts/quick_ablation.py --data-dir data/processed --name nodp --sigma -1   # 無 DP
python scripts/quick_ablation.py --data-dir data/processed --name h3 --horizon 3    # 預測 15 分鐘後
python scripts/quick_ablation.py --data-dir data/processed --name mia --mia         # 成員推論攻擊
"""
import argparse, copy, json, sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task import set_seed, make_loader, LOSSES, unwrap  # noqa: E402
from privacy import compute_epsilon  # noqa: E402
from data import calendar_features  # noqa: E402
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

def time_feats(t, kinds):
    return calendar_features(t, ",".join(kinds))

def load(data_dir, lb, h, frac=0.8, neigh="", mia=False, val=False, cal="", feats="", lagday=False,
         fold=-1, nfolds=5):
    man = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    dfs = [pd.read_csv(Path(data_dir) / c["file"]) for c in man["clients"]]
    raw = [d["speed"].to_numpy(float) for d in dfs]
    cal_all = [time_feats(d["time"], cal.split(",")) for d in dfs]
    # 本 VD 的其他量測（流量、佔有率），同一 client 的私有資料，跟車速一起受 DP 保護
    ext = [[d[f].to_numpy(float) for f in feats.split(",") if f] for d in dfs]
    if val:
        # 調參用：完全不碰測試期間，只用訓練期間，再切出後段當驗證集
        raw = [r[:int(len(r) * frac)] for r in raw]
        cal_all = [c[:int(len(c) * frac)] for c in cal_all]
        ext = [[e[:int(len(e) * frac)] for e in es] for es in ext]
    sp = int(len(raw[0]) * frac)
    # 區塊交叉驗證：整段切成 nfolds 個連續區塊，第 fold 塊當測試、其餘當訓練
    # （訓練視窗不碰測試區塊；測試視窗的輸入可來自前一段，屬於預測當下已知的過去）
    trmask = np.arange(len(raw[0])) < sp
    if fold >= 0:
        blk = np.array_split(np.arange(len(raw[0])), nfolds)[fold]
        lo_t, hi_t = blk[0], blk[-1] + 1
        trmask = np.ones(len(raw[0]), bool); trmask[lo_t:hi_t] = False
    nz = lambda a: (a - np.nanmean(a[trmask])) / (np.nanstd(a[trmask]) or 1.0)  # noqa: E731
    z = [nz(s) for s in raw]
    zext = [[nz(e) for e in es] for es in ext]
    out = []
    for i, c in enumerate(man["clients"]):
        s = raw[i]; tr = s[trmask]
        m, sd = np.nanmean(tr), np.nanstd(tr) or 1.0
        # 相鄰 VD（manifest 依里程排序）：北上車流往里程小的方向，i-1 是下游
        nb = {"down": [i - 1 if i > 0 else i], "up": [i + 1 if i + 1 < len(raw) else i]}
        nb["both"] = nb["down"] + nb["up"]
        def win(lo, hi, zi=None):
            zi = z[i] if zi is None else zi
            series = [zi[lo:hi]] + [z[j][lo:hi] for j in nb.get(neigh, [])] + [e[lo:hi] for e in zext[i]]
            n = hi - lo - lb - h + 1
            idx = np.arange(lb)[None] + np.arange(n)[:, None]
            parts = [q[idx] for q in series] + [cal_all[i][lo + lb + h - 1: lo + lb + h - 1 + n]]
            if lagday:
                # 昨天同一時刻（目標時間 − 288 筆）的車速：本 VD 的歷史值，週期性基準
                tgt = lo + lb + h - 1 + np.arange(n) - 288
                parts.append(np.where(tgt >= 0, zi[np.clip(tgt, 0, None)], np.nan)[:, None])
            X = np.concatenate(parts, 1)
            y = zi[lo:hi][lb + h - 1: lb + h - 1 + n]
            ok = ~np.isnan(X).any(1) & ~np.isnan(y)
            return X[ok], y[ok]
        if fold >= 0:
            zm, zt = z[i].copy(), z[i].copy()
            zm[~trmask] = np.nan
            zt[:max(lo_t - lb - h + 1, 0)] = np.nan; zt[hi_t:] = np.nan
            Xtr, ytr = win(0, len(s), zm); Xte, yte = win(0, len(s), zt)
        else:
            Xtr, ytr = win(0, sp); Xte, yte = win(sp, len(s))
        extra = {}
        if mia:
            # 成員推論：訓練期間每 4 天保留 1 天不訓練（非成員），比較模型在成員/非成員上的誤差
            day = np.arange(len(s)) // 288
            hold = (day % 4 == 3) & (np.arange(len(s)) < sp)
            zm, zn = z[i].copy(), z[i].copy()
            zm[hold] = np.nan; zn[~hold] = np.nan
            Xtr, ytr = win(0, sp, zm); Xn, yn = win(0, sp, zn)
            if len(yn) == 0:
                raise SystemExit("--mia 需要訓練期間至少 4 天的資料")
            extra = dict(Xnon=Xn.astype(np.float32), ynon=yn.astype(np.float32))
        f = lambda a: a.astype(np.float32)
        out.append(dict(**extra, last=lb - 1, Xtr=f(Xtr), ytr=f(ytr), Xte=f(Xte), yte=f(yte), mean=m, std=sd, vd=c["vdid"]))
    return out

class MLP(nn.Module):
    def __init__(s, lb, hid, residual=True, depth=2, nin=None):
        super().__init__(); s.residual = residual; s.last = lb - 1
        layers, d = [], nin or lb
        for _ in range(depth): layers += [nn.Linear(d, hid), nn.ReLU()]; d = hid
        layers.append(nn.Linear(d, 1)); s.net = nn.Sequential(*layers)
    def forward(s, x):
        o = s.net(x).squeeze(-1); return o + x[:, s.last] if s.residual else o

class GRU(nn.Module):
    def __init__(s, lb, hid, residual=True):
        super().__init__(); s.residual = residual
        from opacus.layers import DPGRU
        s.rnn = DPGRU(1, hid, batch_first=True); s.head = nn.Linear(hid, 1)
    def forward(s, x):
        o, _ = s.rnn(x.unsqueeze(-1)); o = s.head(o[:, -1]).squeeze(-1)
        return o + x[:, -1] if s.residual else o

def build(a):
    if a.model == "gru": return GRU(a.lookback, a.hidden)
    k = {"": 1, "down": 2, "up": 2, "both": 3}[a.neigh] + len([f for f in a.feats.split(",") if f])
    ncal = 2 * ("tod" in a.cal) + ("dow" in a.cal)
    return MLP(a.lookback, a.hidden, not a.noresid, a.depth, a.lookback * k + ncal + a.lag_day)

@torch.no_grad()
def ev(model, d):
    model.eval(); X, y = torch.from_numpy(d["Xte"]), torch.from_numpy(d["yte"])
    p = model(X); e = ((p - y) * d["std"]).abs()
    nv = ((X[:, d["last"]] - y) * d["std"]).abs()
    return e.sum().item(), nv.sum().item(), (((p - y) * d["std"]) ** 2).sum().item(), len(y)

def run(a, seed, data):
    set_seed(seed); g = build(a); hist = {i: [] for i in range(len(data))}; tail = []
    for rnd in range(1, a.rounds + 1):
        lr = a.lr if not a.cosine else a.lr * 0.5 * (1 + math.cos(math.pi * (rnd - 1) / a.rounds))
        states, ws = [], []
        gstate = copy.deepcopy(g.state_dict())
        for i, d in enumerate(data):
            set_seed(seed + 1000 * i + rnd)
            m = copy.deepcopy(g).train()
            opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=a.momentum, weight_decay=a.wd)
            ld = make_loader(d["Xtr"], d["ytr"], a.bs, True, seed + 1000 * i + rnd)
            if a.sigma >= 0:
                from opacus import PrivacyEngine
                pe = PrivacyEngine(accountant="rdp")
                m, opt, ld = pe.make_private(module=m, optimizer=opt, data_loader=ld,
                                             noise_multiplier=a.sigma, max_grad_norm=a.clip, poisson_sampling=True)
                q = float(ld.sample_rate)
            lf = LOSSES[a.loss]; steps = 0
            for _ in range(a.epochs):
                for X, y in ld:
                    opt.zero_grad()
                    if len(X) == 0: l = m(X).sum() * 0
                    else:
                        l = lf(m(X), y)
                        if a.mu > 0:
                            l = l + a.mu / 2 * sum(((p - gstate[k.replace("_module.", "")]) ** 2).sum()
                                                   for k, p in m.named_parameters())
                    l.backward(); opt.step(); steps += 1
            if a.sigma >= 0: hist[i].append((a.sigma, q, steps))
            states.append(unwrap(m).state_dict()); ws.append(len(d["ytr"]))
        W = np.array(ws) / sum(ws)
        g.load_state_dict({k: sum(w * s[k] for w, s in zip(W, states)) for k in states[0]})
        if a.avg_last > 1:
            tail = (tail + [copy.deepcopy(g.state_dict())])[-a.avg_last:]
    if a.avg_last > 1:
        # 後處理：最後幾輪全域模型取平均（不碰資料，不花隱私預算）
        g.load_state_dict({k: sum(t[k] for t in tail) / len(tail) for k in tail[0]})
    tot = np.zeros(4); per = []
    for i, d in enumerate(data):
        m = g
        if a.finetune > 0:
            # 個人化：拿最終全域模型在本地再訓練 finetune 個 epoch（同樣 DP），只在本地使用
            set_seed(seed + 1000 * i + 999)
            m = copy.deepcopy(g).train()
            if a.ft_head:
                # 只微調最後一層：要加噪的參數少，雜訊對每個參數的相對影響較小
                for k, prm in m.named_parameters():
                    prm.requires_grad = k.startswith(f"net.{len(m.net) - 1}.")
            opt = torch.optim.SGD([q for q in m.parameters() if q.requires_grad], lr=a.ft_lr or a.lr)
            ld = make_loader(d["Xtr"], d["ytr"], a.bs, True, seed + 1000 * i + 999)
            if a.sigma >= 0:
                from opacus import PrivacyEngine
                fs = a.ft_sigma if a.ft_sigma > 0 else a.sigma
                m, opt, ld = PrivacyEngine(accountant="rdp").make_private(
                    module=m, optimizer=opt, data_loader=ld, noise_multiplier=fs,
                    max_grad_norm=a.clip, poisson_sampling=True)
                q = float(ld.sample_rate)
            steps = 0
            for _ in range(a.finetune):
                for X, y in ld:
                    opt.zero_grad()
                    l = m(X).sum() * 0 if len(X) == 0 else LOSSES[a.loss](m(X), y)
                    l.backward(); opt.step(); steps += 1
            if a.sigma >= 0: hist[i].append((fs, q, steps))
            m = unwrap(m)
            if a.ft_blend < 1:
                # 後處理：預測 = α·微調模型 + (1-α)·全域模型（不碰資料，不多花 ε）
                fm, gm, al = m, copy.deepcopy(g), a.ft_blend
                class Blend(nn.Module):
                    def forward(s, x): return al * fm(x) + (1 - al) * gm(x)
                m = Blend()
        r = ev(m, d); tot += r; per.append(r[0] / r[3])
    mia = None
    if "Xnon" in data[0]:
        def roc_auc_score(y, sc):
            from scipy.stats import rankdata
            y, r = np.array(y), rankdata(sc); n1 = y.sum(); n0 = len(y) - n1
            return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        g.eval(); L, Y = [], []
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for d in data:
                k = len(d["ynon"]); idx = rng.choice(len(d["ytr"]), k, replace=False)
                lm = ((g(torch.from_numpy(d["Xtr"][idx])) - torch.from_numpy(d["ytr"][idx])) ** 2).numpy()
                ln = ((g(torch.from_numpy(d["Xnon"])) - torch.from_numpy(d["ynon"])) ** 2).numpy()
                L += list(-lm) + list(-ln); Y += [1] * k + [0] * k
        auc = roc_auc_score(Y, L)
        tr_mae = np.mean([ev(g, dict(d, Xte=d["Xtr"], yte=d["ytr"]))[0] / len(d["ytr"]) for d in data])
        non_mae = np.mean([ev(g, dict(d, Xte=d["Xnon"], yte=d["ynon"]))[0] / len(d["ynon"]) for d in data])
        mia = dict(auc=float(auc), train_mae=float(tr_mae), non_mae=float(non_mae))
    eps = compute_epsilon(hist[0], 1e-5) if a.sigma > 0 else float("inf")
    return dict(mae=tot[0] / tot[3], naive=tot[1] / tot[3], rmse=math.sqrt(tot[2] / tot[3]), eps=eps, per=per, mia=mia)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="results/quick_ablation.jsonl", help="結果逐行附加到這個 JSONL")
    p.add_argument("--sigma", type=float, default=1.1, help="<0 = 無 DP；0 = 只裁剪不加噪")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--model", default="mlp")
    p.add_argument("--loss", default="mse")
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=0.0, help="聯邦訓練的 weight decay")
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--neigh", default="", choices=["", "down", "up", "both"])
    p.add_argument("--cal", default="", help="日曆特徵：tod（時段 sin/cos）、dow（週末），逗號分隔")
    p.add_argument("--feats", default="", help="本 VD 額外輸入序列：volume、occupancy，逗號分隔")
    p.add_argument("--lag-day", type=int, default=0, choices=[0, 1], help="1 = 加入昨天同一時刻的車速")
    p.add_argument("--avg-last", type=int, default=0, help="最後 k 輪全域模型取平均（後處理）")
    p.add_argument("--fold", type=int, default=-1, help="區塊交叉驗證：第幾塊當測試（0..nfolds-1）；-1 = 原本的前 80%%/後 20%%")
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument("--mia", action="store_true")
    p.add_argument("--val", action="store_true", help="只用訓練期間，後 20%% 當驗證集（調參時不看測試集）")
    p.add_argument("--finetune", type=int, default=0, help="最終模型在各 client 本地再訓練幾個 epoch（個人化）")
    p.add_argument("--ft-lr", type=float, default=0.0, help="個人化的學習率；0 = 同 --lr")
    p.add_argument("--ft-sigma", type=float, default=0.0, help="微調階段的 σ；0 = 同 --sigma（分配隱私預算用）")
    p.add_argument("--ft-head", action="store_true", help="個人化只微調最後一層")
    p.add_argument("--ft-blend", type=float, default=1.0, help="微調模型與全域模型預測的混合比例 α")
    p.add_argument("--noresid", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = p.parse_args(argv)
    data = load(a.data_dir, a.lookback, a.horizon, neigh=a.neigh, mia=a.mia, val=a.val, cal=a.cal, feats=a.feats, lagday=bool(a.lag_day),
                fold=a.fold, nfolds=a.nfolds)
    rs = [run(a, s, data) for s in a.seeds]
    maes = [r["mae"] for r in rs]
    out = dict(name=a.name, args=vars(a), mae=float(np.mean(maes)), std=float(np.std(maes, ddof=1)) if len(maes) > 1 else 0,
               naive=rs[0]["naive"], rmse=float(np.mean([r["rmse"] for r in rs])), eps=rs[0]["eps"],
               per=list(np.mean([r["per"] for r in rs], 0)), maes=maes,
               mia=[r["mia"] for r in rs])
    print(f"{a.name}: MAE {out['mae']:.3f} ± {out['std']:.3f} km/h（naive {out['naive']:.3f}），ε={out['eps']:.2f}",
          flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    main()
