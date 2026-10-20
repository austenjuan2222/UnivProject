"""ServerApp（Flower 1.26 Message API）：FedAvg + 實驗紀錄。

輸出到 <results-dir>/<run-name>/：metrics.csv、clients.csv、config.json、final_model.pt
"""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from results_log import RunLogger, make_run_name
from task import TrafficModel, set_seed

# Opacus 會把 flwr logger 調成 DEBUG，調回 INFO
logging.getLogger("flwr").setLevel(logging.INFO)

app = ServerApp()


def _versions() -> dict:
    out = {}
    for pkg in ("flwr", "torch", "opacus", "numpy", "pandas"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _manifest(data_dir: str) -> dict:
    p = Path(data_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _check_num_nodes(grid: Grid, expected: int, wait_s: float = 60.0) -> None:
    """FedAvg 在節點不足時會無限等待；這裡先確認數量，不一致就明確報錯。"""
    deadline = time.time() + wait_s
    n = len(list(grid.get_node_ids()))
    while n < expected and time.time() < deadline:
        time.sleep(1.0)
        n = len(list(grid.get_node_ids()))
    if n != expected:
        raise RuntimeError(
            f"已連線的 client 數為 {n}，但 num-clients={expected}。請讓 Flower 設定檔的 "
            "options.num-supernodes 與 num-clients、manifest 中的 VD 數一致"
            "（建議用 scripts/run_experiments.sh 執行，它會自動設定）。"
        )


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    num_rounds = int(cfg["num-server-rounds"])
    num_clients = int(cfg["num-clients"])
    set_seed(int(cfg["seed"]))

    manifest = _manifest(str(cfg["data-dir"]))
    vdids = {c["partition"]: c["vdid"] for c in manifest.get("clients", [])}
    if vdids and len(vdids) != num_clients:
        raise RuntimeError(f"num-clients={num_clients} 與 manifest 的 VD 數 {len(vdids)} 不一致")

    run_dir = Path(str(cfg["results-dir"])) / make_run_name(cfg)
    logger = RunLogger(run_dir, cfg, extra={
        "versions": _versions(),
        "data": {k: manifest.get(k) for k in ("start", "end", "num_steps", "selection")},
        "clients": manifest.get("clients", []),
    }, vdid_map=vdids, expected_clients=num_clients)

    _check_num_nodes(grid, num_clients)

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        train_metrics_aggr_fn=lambda replies, _key: MetricRecord(
            logger.on_train([dict(r["metrics"]) for r in replies])),
        evaluate_metrics_aggr_fn=lambda replies, _key: MetricRecord(
            logger.on_eval([dict(r["metrics"]) for r in replies])),
    )
    init = ArrayRecord(TrafficModel(int(cfg["lookback"]), int(cfg["hidden-size"])).state_dict())
    result = strategy.start(grid=grid, initial_arrays=init, num_rounds=num_rounds,
                            train_config=ConfigRecord({}), evaluate_config=ConfigRecord({}))

    torch.save(result.arrays.to_torch_state_dict(), run_dir / "final_model.pt")
    print(f"\n實驗紀錄已寫到 {run_dir}")
