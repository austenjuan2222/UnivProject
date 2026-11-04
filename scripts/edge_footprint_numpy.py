#!/usr/bin/env python3
"""不用 PyTorch 時，一台 VD 本地訓練要多少記憶體（和 scripts/edge_footprint.py 的 PyTorch＋Opacus 版比較）。

父行程用 data.py 切好這台 VD 的訓練／測試視窗存成 .npz（VD 上這一步可以用 numpy 自己做），
再開一個只 import numpy、psutil 和 edge_numpy 的子行程量記憶體與時間；最後在父行程用同樣的設定
跑 PyTorch＋Opacus 版本地訓練，比較本地測試 MAE（DP 雜訊不同，只能比分布，不會一模一樣）。

  python scripts/edge_footprint_numpy.py --data-dir data/processed_28d_32 --rounds 20 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r"""
import sys, time, json
import psutil
proc = psutil.Process(); base = proc.memory_info().rss
import numpy as np
sys.path.insert(0, sys.argv[1])
from edge_numpy import dp_train, predict
libs = proc.memory_info().rss
d = np.load(sys.argv[2]); c = json.loads(sys.argv[3])
p = {k[2:]: d[k] for k in d.files if k.startswith("p:")}
rng = np.random.default_rng(c["seed"]); peak = proc.memory_info().rss; times = []
for _ in range(c["rounds"]):
    t0 = time.perf_counter()
    p, q, steps = dp_train(p, d["X_train"], d["y_train"], c["epochs"], c["batch_size"], c["lr"], c["sigma"], c["clip"], rng)
    times.append(time.perf_counter() - t0); peak = max(peak, proc.memory_info().rss)
mae = float(np.abs((predict(p, d["X_test"]) - d["y_test"]) * float(d["std"])).mean())
print(json.dumps({"base": base, "libs": libs, "peak": peak, "sec": sorted(times)[len(times) // 2], "mae": mae, "q": q, "steps": steps}))
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pid", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = p.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch

    from data import load_client_arrays, n_calendar
    from task import TrafficModel, evaluate, make_loader, set_seed, unwrap

    torch.set_num_threads(1)
    rc = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    n = len(json.loads((Path(a.data_dir) / "manifest.json").read_text(encoding="utf-8"))["clients"])
    d = load_client_arrays(a.data_dir, a.pid, n, int(rc["lookback"]), float(rc.get("train-frac", 0.8)),
                           int(rc.get("horizon", 1)), str(rc.get("calendar", "")), bool(rc.get("lag-day", False)))
    c = {"epochs": int(rc["local-epochs"]), "batch_size": int(rc["batch-size"]), "lr": float(rc["learning-rate"]),
         "sigma": float(rc["noise-multiplier"]), "clip": float(rc["max-grad-norm"]), "rounds": a.rounds}
    assert bool(rc.get("residual", False)), "edge_numpy 只實作 residual=true（目前預設）"

    def new_model():
        return TrafficModel(int(rc["lookback"]), int(rc["hidden-size"]), True,
                            n_calendar(str(rc.get("calendar", ""))) + int(bool(rc.get("lag-day", False))))

    print(f"VD {a.pid}：訓練 {len(d['y_train'])} 筆，每輪 {c['epochs']} epoch，{a.rounds} 輪，只在本地訓練（不聯邦）")
    print("| seed | numpy：Python 本身 | 載入 numpy 後 | 峰值 | 每輪中位數 | 本地測試 MAE | PyTorch＋Opacus MAE |")
    print("|---|---|---|---|---|---|---|")
    from client_app import _dp_train
    with tempfile.TemporaryDirectory() as tmp:
        for seed in a.seeds:
            set_seed(seed)
            init = new_model().state_dict()
            f = Path(tmp) / f"vd{seed}.npz"
            np.savez(f, X_train=d["X_train"], y_train=d["y_train"], X_test=d["X_test"], y_test=d["y_test"],
                     std=np.float32(d["std"]), **{f"p:{k}": v.numpy() for k, v in init.items()})
            out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(ROOT), str(f), json.dumps(c | {"seed": seed})],
                                 capture_output=True, text=True, check=True)
            r = json.loads(out.stdout.strip().splitlines()[-1])
            # 對照：同樣起點、同樣設定，用 client_app 的 PyTorch＋Opacus DP-SGD
            state, prev = init, None
            cfg = {"sigma": c["sigma"], "clip": c["clip"], "delta": float(rc["target-delta"]), "loss": "mse"}
            for rnd in range(1, a.rounds + 1):
                set_seed(seed * 1000 + rnd)
                m = new_model()
                m.load_state_dict(state)
                opt = torch.optim.SGD(m.parameters(), lr=c["lr"])
                loader = make_loader(d["X_train"], d["y_train"], c["batch_size"], shuffle=True, seed=seed * 1000 + rnd)
                m, _, prev, _, _, _ = _dp_train(m, opt, loader, cfg, prev, c["epochs"])
                state = unwrap(m).state_dict()
            m = new_model()
            m.load_state_dict(state)
            t = evaluate(m, d["X_test"], d["y_test"], d["mean"], d["std"])["mae_kmh"]
            mb = 2 ** 20
            print(f"| {seed} | {r['base'] / mb:.0f} MB | {r['libs'] / mb:.0f} MB | {r['peak'] / mb:.0f} MB | "
                  f"{r['sec']:.3f} 秒 | {r['mae']:.3f} | {t:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
