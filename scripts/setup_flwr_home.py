#!/usr/bin/env python3
"""依 manifest 的 VD 數寫入專案專用的 Flower 設定檔（$FLWR_HOME/config.toml）。

Flower >= 1.26 的 num-supernodes 只能寫在 Flower 設定檔；為了不改動家目錄設定，
run_experiments.sh 會把 FLWR_HOME 指到專案內 .flwr_home/。
標準輸出只印 client 數，方便 shell 擷取：N=$(python scripts/setup_flwr_home.py ...)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_config(flwr_home: Path, num_supernodes: int, num_cpus: float = 1, num_gpus: float = 0) -> Path:
    flwr_home = Path(flwr_home)
    flwr_home.mkdir(parents=True, exist_ok=True)
    path = flwr_home / "config.toml"
    path.write_text(
        "# 由 scripts/setup_flwr_home.py 產生，請勿手動修改\n"
        '[superlink]\ndefault = "local"\n\n'
        "[superlink.local]\n"
        f"options.num-supernodes = {int(num_supernodes)}\n"
        f"options.backend.client-resources.num-cpus = {num_cpus}\n"
        f"options.backend.client-resources.num-gpus = {num_gpus}\n",
        encoding="utf-8",
    )
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--flwr-home", required=True)
    p.add_argument("--num-cpus", type=float, default=1)
    p.add_argument("--num-gpus", type=float, default=0)
    a = p.parse_args(argv)
    manifest = Path(a.data_dir) / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"找不到 {manifest}，請先執行 scripts/preprocess_vd.py")
    n = len(json.loads(manifest.read_text(encoding="utf-8"))["clients"])
    write_config(Path(a.flwr_home), n, a.num_cpus, a.num_gpus)
    return n


if __name__ == "__main__":
    print(main())
