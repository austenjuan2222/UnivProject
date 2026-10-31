#!/usr/bin/env python3
"""跨平台（含 Windows，不需 bash）的實驗執行器：無 DP 對照組 + 各 σ 的 DP，最後畫比較圖。

  python scripts/run_experiments.py                         # 20 輪；無 DP + DP σ=1.1
  python scripts/run_experiments.py --rounds 30 --sigmas 0.8 1.1 1.5
  python scripts/run_experiments.py --sigmas 0.8 1.1 1.5 --seeds 1 2 3   # 每個設定重複 3 個 seed
  python scripts/run_experiments.py --smoke                 # 合成資料、2 輪，檢查輸出
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _p(path) -> str:
    """TOML 字串內用正斜線，避免 Windows 反斜線被當成跳脫字元。"""
    return str(path).replace("\\", "/")


def build_runs(n_clients, rounds, data_dir, batch_dir, sigmas, skip_nodp, extra, seeds=None,
               client_sigmas=()):
    """回傳 [(run 名稱, --run-config 字串)]。

    seeds 多於 1 個時，每個設定都用每個 seed 重跑一次（名稱加 _seed<N>），
    plot_results.py 會把同設定的 run 合併成平均 ± 標準差，用來判斷差異是否大於隨機波動。
    """
    common = (f"num-clients={n_clients} num-server-rounds={rounds} "
              f"data-dir='{_p(data_dir)}' results-dir='{_p(batch_dir)}' {extra}").strip()
    configs = [] if skip_nodp else [("nodp", "use-dp=false")]
    configs += [(f"dp_sigma{s}", f"use-dp=true noise-multiplier={s}") for s in sigmas]
    configs += [(f"cldp_sigma{z}", f"use-dp=false client-dp-noise={z}") for z in client_sigmas]
    seeds = list(seeds or [None])
    runs = []
    for seed in seeds:
        for name, opts in configs:
            if seed is None:
                runs.append((name, f"{common} {opts} run-name='{name}'"))
                continue
            name = f"{name}_seed{seed}" if len(seeds) > 1 else name
            runs.append((name, f"{common} {opts} seed={seed} run-name='{name}'"))
    return runs


def _flwr_exe() -> str:
    here = Path(sys.executable).parent
    for name in ("flwr", "flwr.exe"):
        if (here / name).exists():
            return str(here / name)
    found = shutil.which("flwr")
    if not found:
        raise SystemExit("找不到 flwr 指令，請先 pip install -e .")
    return found


def run_batch(data_dir: Path, results_dir: Path, rounds: int, sigmas, skip_nodp=False,
              extra="", num_cpus=1.0, seeds=None, client_sigmas=()) -> Path:
    from setup_flwr_home import main as setup_home

    flwr_home = ROOT / ".flwr_home"
    n = setup_home(["--data-dir", str(data_dir), "--flwr-home", str(flwr_home),
                    "--num-cpus", str(num_cpus)])
    batch = results_dir / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch.mkdir(parents=True, exist_ok=True)
    print(f"client 數 = {n}（來自 {data_dir / 'manifest.json'}）；結果寫到 {batch}", flush=True)

    env = dict(os.environ, FLWR_HOME=str(flwr_home), PYTHONUTF8="1")
    # 讓 flwr 啟動的 flower-simulation 也找得到同一個 venv
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    flwr = _flwr_exe()
    for name, cfg in build_runs(n, rounds, data_dir, batch, sigmas, skip_nodp, extra, seeds,
                                client_sigmas):
        print(f"=== {name} ===", flush=True)
        subprocess.run([flwr, "run", str(ROOT), "local", "--run-config", cfg], check=True, env=env)

    from plot_results import main as plot_main
    plot_main([str(batch)])
    return batch


def _rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_smoke_outputs(batch: Path, rounds: int, n_clients: int, sigma: str) -> list[float]:
    for run in ("nodp", f"dp_sigma{sigma}"):
        rows = _rows(batch / run / "metrics.csv")
        assert len(rows) == rounds, (run, rows)
        assert all(r["mae_kmh"] for r in rows), (run, rows)
        assert len(_rows(batch / run / "clients.csv")) == rounds * n_clients, run
        assert (batch / run / "final_model.pt").exists(), run
    eps = [float(r["epsilon_max"]) for r in _rows(batch / f"dp_sigma{sigma}" / "metrics.csv")]
    assert eps[0] > 0 and all(b > a for a, b in zip(eps, eps[1:])), f"ε 應隨輪次遞增：{eps}"
    assert (batch / "comparison.png").exists() and (batch / "summary.csv").exists()
    return eps


def smoke() -> None:
    from make_synthetic_vd import make
    from vd_preprocess import preprocess

    tmp = ROOT / ".smoke"
    shutil.rmtree(tmp, ignore_errors=True)
    make(tmp / "raw", days=3, num_vds=3)
    preprocess(tmp / "raw", tmp / "processed", pattern="N1", n=3)
    batch = run_batch(tmp / "processed", tmp / "results", rounds=2, sigmas=["1.1"])
    eps = check_smoke_outputs(batch, rounds=2, n_clients=3, sigma="1.1")
    print(f"SMOKE TEST OK  ε 逐輪：{eps}  輸出：{batch}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--sigmas", nargs="*", default=["1.1"])
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="每個設定重複跑的 seed（例如 1 2 3）；不給則用 pyproject 的 seed")
    p.add_argument("--client-sigmas", nargs="*", default=[],
                   help="client-level DP（保護整個 VD）的雜訊倍率，例如 2 4；裁剪上限用 --extra client-dp-clip=…")
    p.add_argument("--skip-nodp", action="store_true")
    p.add_argument("--num-cpus", type=float, default=1.0)
    p.add_argument("--extra", default="", help='例如 "learning-rate=0.02 local-epochs=2"')
    p.add_argument("--smoke", action="store_true", help="用合成資料跑 2 輪並檢查輸出")
    a = p.parse_args(argv)
    if a.smoke:
        smoke()
        return
    run_batch(Path(a.data_dir).resolve(), Path(a.results_dir).resolve(), a.rounds, a.sigmas,
              a.skip_nodp, a.extra, a.num_cpus, a.seeds, a.client_sigmas)


if __name__ == "__main__":
    main()
