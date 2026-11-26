import sys, numpy as np, torch
import os; _R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(_R, "scripts")); sys.path.insert(0, _R)
DATA_ROOT = os.environ.get("VD_DATA_ROOT", "/mnt/project-files/real-vd")
import quick_ablation as qa
from torch.func import functional_call, vmap, grad
torch.manual_seed(0)
a = qa.main.__globals__  # noqa
import argparse
ns = argparse.Namespace(model="mlp", neigh="", feats="", cal="tod", lookback=12, hidden=32, noresid=False, depth=2, lag_day=0)
data = qa.load(os.path.join(DATA_ROOT, "processed_28d"), 12, 1, cal="tod")
d = data[0]; X = torch.from_numpy(d["Xtr"]); y = torch.from_numpy(d["ytr"])
m = qa.build(ns); print("params", sum(p.numel() for p in m.parameters()), "train n", len(y))
opt = torch.optim.SGD(m.parameters(), lr=0.2)
def stats(tag):
    params = {k: v.detach() for k, v in m.named_parameters()}
    def loss(p, xb, yb): return ((functional_call(m, p, (xb[None],)) - yb[None]) ** 2).mean()
    idx = torch.randperm(len(y))[:2048]
    g = vmap(grad(loss), in_dims=(None, 0, 0))(params, X[idx], y[idx])
    flat = torch.cat([v.reshape(len(idx), -1) for v in g.values()], 1)
    nrm = flat.norm(dim=1); clipped = flat * torch.clamp(1.0 / nrm, max=1.0)[:, None]
    B = 64; sig = 1.1; D = flat.shape[1]
    sig_norm = clipped.mean(0).norm().item()
    noise_norm = sig * 1.0 / B * np.sqrt(D)
    print(f"{tag}: median per-sample grad norm {nrm.median():.3f}, clipped frac {(nrm>1).float().mean():.2f}, "
          f"mean clipped grad norm {sig_norm:.4f}, per-step noise norm (σ1.1,B64) {noise_norm:.3f}, "
          f"noise/signal per step {noise_norm/sig_norm:.1f}x; over 100 steps/epoch noise/signal ≈ {noise_norm/sig_norm/np.sqrt(100):.2f}x")
stats("init")
for ep in range(3):
    perm = torch.randperm(len(y))
    for i in range(0, len(y), 64):
        b = perm[i:i+64]; opt.zero_grad(); ((m(X[b]) - y[b]) ** 2).mean().backward(); opt.step()
    stats(f"after {ep+1} epoch nodp")
