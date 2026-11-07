#!/usr/bin/env python3
"""量一台 VD（Edge）本地訓練一輪要多少記憶體與時間（對應講義 IoT 架構的 Edge 資源限制）。

用 client_app.py 同一套程式（同樣的設定、資料、DP-SGD），但只跑一個 VD、一個行程，
這樣行程的記憶體就是這台 VD 自己的用量，不會混到 Flower 模擬時其他 VD 的行程。

  python scripts/edge_footprint.py --data-dir data/processed_28d_32 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mb(n: int) -> float:
    return n / 2**20


def crypto_costs(arrays: dict, peers: int, reps: int = 200) -> dict:
    """VD 每輪要做的資安運算，各重複 reps 次取中位數（毫秒）；金鑰在記憶體裡臨時產生，不讀 keys/。"""
    import os
    from statistics import median

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    from secagg import mask_update
    from security import sign_model, sign_update, verify_model

    def ms(fn, n: int = reps) -> float:
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return median(ts)

    server, vd = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    sig = sign_model(server, arrays, 1)
    xs = [X25519PrivateKey.generate() for _ in range(peers)]
    pubs = {i: k.public_key() for i, k in enumerate(xs)}
    masked = mask_update(arrays, 1, 0, range(peers), xs[0], pubs, 1)
    return {
        "verify_model_ms": ms(lambda: verify_model(server.public_key(), arrays, 1, sig)),
        "hmac_ms": ms(lambda: sign_update(os.urandom(32), arrays, 1, 0)),
        "ed25519_sign_ms": ms(lambda: sign_update(vd, arrays, 1, 0)),
        # 安全聚合：和其他 peers-1 台各做一次 X25519 金鑰交換再產生遮罩
        "secagg_mask_ms": ms(lambda: mask_update(arrays, 1, 0, range(peers), xs[0], pubs, 1), max(reps // 10, 5)),
        "hmac_tag_bytes": 32, "ed25519_sig_bytes": 64,
        "secagg_upload_bytes": int(sum(v.nbytes for v in masked.values())),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pid", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--peers", type=int, default=32, help="安全聚合時同一輪的 VD 數")
    a = p.parse_args(argv)

    proc = psutil.Process()
    base = proc.memory_info().rss
    import torch  # noqa: E402  先量 Python 本身，再量載入 PyTorch／Opacus 後
    import opacus  # noqa: F401,E402

    from client_app import _cfg, _dp_train  # noqa: E402
    from data import load_client_arrays, n_calendar  # noqa: E402
    from task import TrafficModel, make_loader, set_seed, unwrap  # noqa: E402

    torch.set_num_threads(1)  # 路邊設備通常只有一兩個核心
    libs = proc.memory_info().rss
    rc = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    manifest_n = len(json.loads((Path(a.data_dir) / "manifest.json").read_text(encoding="utf-8"))["clients"])
    rc.update({"data-dir": a.data_dir, "num-server-rounds": a.rounds})
    cfg = _cfg(SimpleNamespace(run_config=rc))
    d = load_client_arrays(cfg["data_dir"], a.pid, manifest_n, cfg["lookback"], cfg["train_frac"],
                           cfg["horizon"], cfg["calendar"], cfg["lag_day"])
    data_mb = sum(v.nbytes for v in d.values() if hasattr(v, "nbytes")) / 2**20
    def new_model():
        return TrafficModel(cfg["lookback"], cfg["hidden"], cfg["residual"],
                            n_calendar(cfg["calendar"]) + int(cfg["lag_day"]))
    state = new_model().state_dict()
    n_params = sum(t.numel() for t in state.values())

    peak, times, prev = proc.memory_info().rss, [], None
    for rnd in range(1, a.rounds + 1):
        set_seed(cfg["seed"] + rnd)
        model = new_model()  # 和 client_app 一樣：每輪收到全域模型後重建
        model.load_state_dict(state)
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
        loader = make_loader(d["X_train"], d["y_train"], cfg["batch_size"], shuffle=True, seed=rnd)
        t0 = time.perf_counter()
        m, _, prev, eps, steps, _ = _dp_train(model, opt, loader, cfg, prev, cfg["epochs"])
        times.append(time.perf_counter() - t0)
        state = unwrap(m).state_dict()
        peak = max(peak, proc.memory_info().rss)

    print(f"VD {a.pid}：訓練樣本 {len(d['y_train'])}、模型參數 {n_params}（{n_params * 4 / 1024:.1f} KB）、"
          f"資料陣列 {data_mb:.2f} MB")
    print(f"記憶體：Python 本身 {mb(base):.0f} MB，載入 PyTorch＋Opacus 後 {mb(libs):.0f} MB，"
          f"訓練時峰值 {mb(peak):.0f} MB（訓練本身多用 {mb(peak - libs):.0f} MB）")
    print(f"每輪 DP-SGD 時間（單執行緒）：{', '.join(f'{t:.2f}' for t in times)} 秒；{a.rounds} 輪後 ε={eps:.2f}")
    c = crypto_costs({k: v.numpy() for k, v in state.items()}, a.peers)
    print(f"資安運算（中位數）：驗全域模型簽章 {c['verify_model_ms']:.3f} ms、HMAC {c['hmac_ms']:.3f} ms、"
          f"Ed25519 簽更新 {c['ed25519_sign_ms']:.3f} ms、安全聚合遮罩（{a.peers} 台）{c['secagg_mask_ms']:.1f} ms")
    print(f"額外上傳：HMAC 標籤 {c['hmac_tag_bytes']} B、Ed25519 簽章 {c['ed25519_sig_bytes']} B；"
          f"安全聚合後上傳 {c['secagg_upload_bytes'] / 1024:.1f} KB（mod 2^32 的 uint32）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
