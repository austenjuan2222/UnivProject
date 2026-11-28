#!/usr/bin/env python3
"""把即時預測用的模型與標準化參數匯出成一個 JSON（給 Vercel 上不跑 PyTorch 的版本用）。

  python scripts/export_bundle.py --run-dir <訓練結果>/dp_sigma1.1_seed1 --data-dir ../UnivProject-run/data/processed_28d_32 \
      --source-commit <產生 run 的 commit> --out deploy/model_bundle.json

模型與標準化參數都經過 forecast_live 的 load_model／vd_stats 取得，和網頁伺服器用的完全相同；
匯出後用純 numpy 照 bundle 重算一次，和 PyTorch 的輸出比對，不一致就失敗。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_live as fl  # noqa: E402


def km(vdid: str) -> float:
    m = re.match(r"^VD-N\d+-[NS]-([\d.]+)-", vdid)
    if not m:
        raise ValueError(f"看不出里程：{vdid}")
    return float(m.group(1))


def numpy_forward(bundle: dict, x: np.ndarray) -> np.ndarray:
    """照 bundle 的 layers 重算（Linear→ReLU→Linear→ReLU→Linear，residual 加回最後一筆）。"""
    w = {k: np.array(v["data"], dtype=np.float64).reshape(v["shape"]) for k, v in bundle["state_dict"].items()}
    h = x
    for i, name in enumerate(bundle["model"]["layers"]):
        h = h @ w[f"{name}.weight"].T + w[f"{name}.bias"]
        if i < len(bundle["model"]["layers"]) - 1:
            h = np.maximum(h, 0)
    out = h[:, 0]
    return out + x[:, -1] if bundle["model"]["residual"] else out


def build(run_dir: Path, data_dir: Path, source_commit: str) -> tuple[dict, object]:
    model, cfg = fl.load_model(run_dir)  # 已拒絕 horizon != 1 與 lag-day
    stats = fl.vd_stats(data_dir, cfg["train-frac"])
    sd = model.state_dict()
    bundle = {
        "format": "univproject-forecast-bundle/1",
        "source": {"run_name": "/".join(run_dir.parts[-4:]),
                   "commit": source_commit, "data_dir": data_dir.name,
                   "final_model_sha256": hashlib.sha256((run_dir / "final_model.pt").read_bytes()).hexdigest()},
        "model": {
            "type": "TrafficModel", "lookback": cfg["lookback"], "horizon": int(cfg.get("horizon", 1)),
            "hidden_size": cfg["hidden-size"], "residual": bool(cfg["residual"]), "calendar": cfg["calendar"],
            "step_minutes": 5, "layers": ["net.0", "net.2", "net.4"],
            "input": "[日曆特徵（目標時刻；tod = sin、cos of 一天中的分鐘/1440*2π，台灣時間）] + 最後 lookback 筆標準化車速（舊→新）",
            "output": "下一個 5 分鐘的標準化車速；km/h = 輸出 * std + mean。10、15 分鐘把輸出接回輸入再算（遞迴）",
        },
        "state_dict": {k: {"shape": list(v.shape), "data": v.detach().double().flatten().tolist()} for k, v in sd.items()},
        "vds": [{"vdid": v, "km": km(v), "mean": stats[v][0], "std": stats[v][1]} for v in sorted(stats)],
    }
    return bundle, model


def self_check(bundle: dict, model) -> float:
    import torch
    rng = np.random.default_rng(0)
    n_in = bundle["state_dict"]["net.0.weight"]["shape"][1]
    x = rng.normal(size=(64, n_in))
    with torch.no_grad():
        ref = model(torch.tensor(x, dtype=torch.float32)).double().numpy()
    err = float(np.max(np.abs(numpy_forward(bundle, x) - ref)))
    if err > 1e-4:
        raise SystemExit(f"bundle 重算和 PyTorch 差 {err}，不匯出")
    return err


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--out", type=Path, default=Path("deploy/model_bundle.json"))
    a = p.parse_args(argv)
    bundle, model = build(a.run_dir, a.data_dir, a.source_commit)
    err = self_check(bundle, model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.out}：{len(bundle['vds'])} 台 VD，numpy 與 PyTorch 最大差 {err:.2e}")


if __name__ == "__main__":
    main()
