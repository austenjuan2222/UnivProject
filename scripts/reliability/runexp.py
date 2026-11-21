"""資料量與 DP 可靠度檢查（2026-11-20）。用法：
  set VD_DATA_ROOT=<含 processed_28d 的資料夾>
  在本資料夾內：python runexp.py --tag sg1.1 --sigma 1.1 --seed 1（jobs.txt 為全部 140 組），跑完 python analyze.py
"""
import sys, json, argparse, time
import numpy as np, torch
import os; _R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(_R, "scripts")); sys.path.insert(0, _R)
DATA_ROOT = os.environ.get("VD_DATA_ROOT", "/mnt/project-files/real-vd")
import quick_ablation as qa
REC = []
_ev = qa.ev
def ev(model, d):
    with torch.no_grad():
        model.eval(); X, y = torch.from_numpy(d["Xte"]), torch.from_numpy(d["yte"])
        REC.append((((model(X) - y) * d["std"]).abs().numpy().ravel().copy(),
                    ((X[:, d["last"]] - y) * d["std"]).abs().numpy().ravel().copy()))
    return _ev(model, d)
qa.ev = ev
p = argparse.ArgumentParser(); p.add_argument("--data", default="processed_28d"); p.add_argument("--tag", required=True)
p.add_argument("--sigma", type=float, required=True); p.add_argument("--seed", type=int, required=True)
p.add_argument("--fold", type=int, default=-1); p.add_argument("--ft", type=int, default=3)
o = p.parse_args()
t = time.time()
qa.main(["--name", o.tag, "--data-dir", os.path.join(DATA_ROOT, o.data), "--sigma", str(o.sigma),
         "--cal", "tod", "--finetune", str(o.ft), "--ft-lr", "0.05", "--fold", str(o.fold), "--seeds", str(o.seed),
         "--out", f"res/{o.data}_{o.tag}_f{o.fold}_s{o.seed}.jsonl"])
np.savez_compressed(f"res/{o.data}_{o.tag}_f{o.fold}_s{o.seed}.npz",
                    err=np.concatenate([r[0] for r in REC]), naive=np.concatenate([r[1] for r in REC]),
                    sizes=np.array([len(r[0]) for r in REC]))
print("time", round(time.time() - t, 1))
