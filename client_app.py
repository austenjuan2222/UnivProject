"""ClientApp（Flower 1.26 Message API）：每個 client = 一個 VD 偵測器。

- 資料：data.load_client_arrays（依時間切 train/test，標準化只用 train）。
- DP：use-dp=true 時以 Opacus 做 DP-SGD（Poisson 取樣、逐樣本裁剪、加噪）。
- ε：每輪把（σ, q, 步數）存進 context.state，用全部歷史計算累計 ε 回報。
"""
from __future__ import annotations

import logging
import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data import load_client_arrays
from task import TrafficModel, evaluate, make_loader, set_seed, train, unwrap

warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
logging.getLogger("flwr").setLevel(logging.INFO)

DP_STATE_KEY = "dp-history"

app = ClientApp()


def _cfg(context: Context) -> dict:
    c = context.run_config
    return {
        "data_dir": str(c["data-dir"]),
        "lookback": int(c["lookback"]),
        "hidden": int(c["hidden-size"]),
        "residual": bool(c.get("residual", False)),
        "loss": str(c.get("loss", "mse")),
        "train_frac": float(c["train-frac"]),
        "batch_size": int(c["batch-size"]),
        "lr": float(c["learning-rate"]),
        "epochs": int(c["local-epochs"]),
        "use_dp": bool(c["use-dp"]),
        "sigma": float(c["noise-multiplier"]),
        "clip": float(c["max-grad-norm"]),
        "delta": float(c["target-delta"]),
        "seed": int(c["seed"]),
    }


def _client_data(context: Context, cfg: dict):
    pid = int(context.node_config["partition-id"])
    n = int(context.node_config["num-partitions"])
    return pid, load_client_arrays(cfg["data_dir"], pid, n, cfg["lookback"], cfg["train_frac"])


def _model(cfg: dict, msg: Message) -> TrafficModel:
    model = TrafficModel(cfg["lookback"], cfg["hidden"], cfg["residual"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    return model


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    from privacy import append_history, compute_epsilon, history_tuples

    cfg = _cfg(context)
    pid, d = _client_data(context, cfg)
    rnd = int(msg.content["config"]["server-round"])
    set_seed(cfg["seed"] + 1000 * pid + rnd)

    model = _model(cfg, msg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    loader = make_loader(d["X_train"], d["y_train"], cfg["batch_size"], shuffle=True,
                         seed=cfg["seed"] + 1000 * pid + rnd)
    metrics = {"num-examples": len(d["y_train"]), "round": rnd, "partition-id": pid}

    if cfg["use_dp"]:
        from opacus import PrivacyEngine

        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=cfg["sigma"], max_grad_norm=cfg["clip"], poisson_sampling=True,
        )
        # Poisson 取樣的真實機率（= 1/批次數）；Opacus 內部 accountant 用 1/len(loader)，
        # 少數 n 下會略大（較保守），這裡用真實值。
        sample_rate = float(loader.sample_rate)
        loss, steps = train(model, loader, optimizer, cfg["epochs"], loss=cfg["loss"])

        prev = context.state[DP_STATE_KEY] if DP_STATE_KEY in context.state else None
        prev = {k: list(prev[k]) for k in ("sigma", "q", "steps")} if prev is not None else None
        hist = append_history(prev, cfg["sigma"], sample_rate, steps)
        context.state[DP_STATE_KEY] = MetricRecord(hist)
        metrics["epsilon"] = compute_epsilon(history_tuples(hist), cfg["delta"])
        metrics["dp-steps"] = int(steps)
        metrics["sample-rate"] = sample_rate
    else:
        loss, steps = train(model, loader, optimizer, cfg["epochs"], loss=cfg["loss"])

    metrics["train-loss"] = float(loss)
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(unwrap(model).state_dict()),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = _cfg(context)
    pid, d = _client_data(context, cfg)
    model = _model(cfg, msg)
    r = evaluate(model, d["X_test"], d["y_test"], d["mean"], d["std"])
    metrics = {
        "num-examples": len(d["y_test"]),
        "round": int(msg.content["config"]["server-round"]),
        "partition-id": pid,
        "eval-loss": r["loss"],
        "mae-kmh": r["mae_kmh"],
        "mse-kmh": r["mse_kmh"],
        "naive-mae-kmh": r["naive_mae_kmh"],
    }
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
