"""ClientApp（Flower 1.26 Message API）：每個 client = 一個 VD 偵測器。

- 資料：data.load_client_arrays（依時間切 train/test，標準化只用 train）。
- DP：use-dp=true 時以 Opacus 做 DP-SGD（Poisson 取樣、逐樣本裁剪、加噪）。
- ε：每輪把（σ, q, 步數）存進 context.state，用全部歷史計算累計 ε 回報。
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np

import torch
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data import load_client_arrays, n_calendar
from guard import AUTH, arrays_of, attach_update_tag
from secagg import load_agreement_keys, mask_update
from security import (drops_out, is_attacker, load_server_public_key, load_vd_key, poison_update,
                      verify_model)
from task import TrafficModel, all_finite, evaluate, make_loader, set_seed, train, unwrap

warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
logging.getLogger("flwr").setLevel(logging.INFO)

DP_STATE_KEY = "dp-history"

app = ClientApp()


def _cfg(context: Context) -> dict:
    c = context.run_config
    return {
        "data_dir": str(c["data-dir"]),
        "lookback": int(c["lookback"]),
        "horizon": int(c.get("horizon", 1)),
        "hidden": int(c["hidden-size"]),
        "residual": bool(c.get("residual", False)),
        "calendar": str(c.get("calendar", "")),
        "lag_day": bool(c.get("lag-day", False)),
        "loss": str(c.get("loss", "mse")),
        "train_frac": float(c["train-frac"]),
        "batch_size": int(c["batch-size"]),
        "lr": float(c["learning-rate"]),
        "epochs": int(c["local-epochs"]),
        "use_dp": bool(c["use-dp"]),
        "sigma": float(c["noise-multiplier"]),
        "clip": float(c["max-grad-norm"]),
        # 無 DP 時的一般梯度裁剪（整批梯度範數；0 = 不裁剪），作為「穩定但不私密」的對照組
        "nodp_grad_clip": float(c.get("nodp-grad-clip", 0.0)),
        "delta": float(c["target-delta"]),
        "seed": int(c["seed"]),
        "rounds": int(c["num-server-rounds"]),
        "ft_epochs": int(c.get("finetune-epochs", 0)),
        "ft_lr": float(c.get("finetune-lr", 0.05)),
        # 通訊安全與攻擊模擬（見 security.py、guard.py）
        "key_dir": str(c.get("key-dir", "keys")),
        "verify_updates": bool(c.get("verify-updates", False)),
        "sign_models": bool(c.get("sign-models", False)),
        "attack_clients": int(c.get("attack-clients", 0)),
        "attack_type": str(c.get("attack-type", "flip")),
        "attack_scale": float(c.get("attack-scale", 10.0)),
        "dropout_rate": float(c.get("dropout-rate", 0.0)),
        "secure_agg": bool(c.get("secure-agg", False)),
    }


def _check_global_signature(cfg: dict, msg: Message, rnd: int) -> None:
    """VD 只接受伺服器簽過名的全域模型（避免被騙去載入假的模型更新）。"""
    if not cfg["sign_models"]:
        return
    sig = str(msg.content[AUTH]["sig"]) if AUTH in msg.content else ""
    if not (sig and verify_model(load_server_public_key(cfg["key_dir"]), arrays_of(msg.content["arrays"]),
                                 rnd, sig)):
        raise RuntimeError(f"第 {rnd} 輪全域模型的簽章驗證失敗，VD 拒絕使用這個模型")


def _client_data(context: Context, cfg: dict):
    pid = int(context.node_config["partition-id"])
    n = int(context.node_config["num-partitions"])
    return pid, load_client_arrays(cfg["data_dir"], pid, n, cfg["lookback"], cfg["train_frac"],
                                cfg["horizon"], cfg["calendar"], cfg["lag_day"])


def _model(cfg: dict, msg: Message) -> TrafficModel:
    n_extra = n_calendar(cfg["calendar"]) + int(cfg["lag_day"])
    model = TrafficModel(cfg["lookback"], cfg["hidden"], cfg["residual"], n_extra)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    return model


def _dp_history(context: Context):
    prev = context.state[DP_STATE_KEY] if DP_STATE_KEY in context.state else None
    return {k: list(prev[k]) for k in ("sigma", "q", "steps")} if prev is not None else None


def _dp_train(model, optimizer, loader, cfg: dict, prev, epochs: int):
    """DP-SGD 訓練 epochs 個 epoch；回傳（模型, loss, 新歷史, 累計 ε, 步數, 取樣率）。"""
    from opacus import PrivacyEngine

    from privacy import append_history, compute_epsilon, history_tuples

    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=cfg["sigma"], max_grad_norm=cfg["clip"], poisson_sampling=True,
    )
    # Poisson 取樣的真實機率（= 1/批次數）；Opacus 內部 accountant 用 1/len(loader)，
    # 少數 n 下會略大（較保守），這裡用真實值。
    sample_rate = float(loader.sample_rate)
    loss, steps = train(model, loader, optimizer, epochs, loss=cfg["loss"])
    hist = append_history(prev, cfg["sigma"], sample_rate, steps)
    return model, loss, hist, compute_epsilon(history_tuples(hist), cfg["delta"]), steps, sample_rate


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    cfg = _cfg(context)
    pid, d = _client_data(context, cfg)
    rnd = int(msg.content["config"]["server-round"])
    if drops_out(cfg["seed"], pid, rnd, cfg["dropout_rate"]):
        raise RuntimeError(f"模擬 VD {pid} 在第 {rnd} 輪斷線")  # 伺服器會收到錯誤回覆並略過這個 VD
    _check_global_signature(cfg, msg, rnd)
    set_seed(cfg["seed"] + 1000 * pid + rnd)
    t0 = time.perf_counter()

    model = _model(cfg, msg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    loader = make_loader(d["X_train"], d["y_train"], cfg["batch_size"], shuffle=True,
                         seed=cfg["seed"] + 1000 * pid + rnd)
    # 每個 client 都要有同一組 metric 鍵，Flower 才能彙整（diverged 只在發散時設成 1）
    metrics = {"num-examples": len(d["y_train"]), "round": rnd, "partition-id": pid, "diverged": 0}

    if cfg["use_dp"]:
        model, loss, hist, eps, steps, sample_rate = _dp_train(
            model, optimizer, loader, cfg, _dp_history(context), cfg["epochs"])
        context.state[DP_STATE_KEY] = MetricRecord(hist)
        metrics["epsilon"] = eps
        metrics["dp-steps"] = int(steps)
        metrics["sample-rate"] = sample_rate
    else:
        loss, steps = train(model, loader, optimizer, cfg["epochs"], loss=cfg["loss"],
                            grad_clip=cfg["nodp_grad_clip"])

    if not all_finite(model):
        # 本地訓練發散（例如沒有逐樣本裁剪時，標準差很小的 VD 遇到壅塞，標準化後輸入達 −20 以上，
        # 梯度爆掉）：回傳收到的全域模型（更新量 = 0），避免一個 client 的 NaN 汙染整個聯邦。
        # 判斷只看訓練結果本身（DP 時屬於後處理），不額外使用資料。
        model = _model(cfg, msg)
        metrics["diverged"] = 1
    metrics["train-loss"] = float(loss)
    # 個人化參考：本輪剛在本地訓練完、尚未聚合的模型在本地測試集的 MAE。
    # 這個模型本來就是 DP 訓練的輸出，不會多花隱私預算；它只留在本地，不影響聚合。
    r = evaluate(unwrap(model), d["X_test"], d["y_test"], d["mean"], d["std"])
    metrics["local-mae-kmh"] = r["mae_kmh"]
    metrics["num-test"] = len(d["y_test"])
    arrays = ArrayRecord(unwrap(model).state_dict())
    n = int(context.node_config["num-partitions"])
    if is_attacker(pid, n, cfg["attack_clients"]):
        # 被入侵的 VD：送出惡意更新（它仍有合法金鑰，所以下面照樣簽章；MAC 擋不住這種攻擊）
        g = arrays_of(msg.content["arrays"])
        bad = poison_update(g, arrays_of(arrays), cfg["attack_type"], cfg["attack_scale"],
                            np.random.default_rng([cfg["seed"], pid, rnd]))
        arrays = ArrayRecord({k: Array(v) for k, v in bad.items()})
    if cfg["secure_agg"]:
        # 安全聚合：上傳「樣本數 × 參數 ＋ 成對遮罩」，伺服器只能還原所有 VD 的加總
        priv, pubs = load_agreement_keys(cfg["key_dir"], pid)
        masked = mask_update(arrays_of(arrays), len(d["y_train"]), pid, range(n), priv, pubs, rnd)
        arrays = ArrayRecord({k: Array(v) for k, v in masked.items()})
    # Edge 運算成本：上傳量（模型參數）vs. 若把原始訓練資料傳上雲的量
    metrics["upload-bytes"] = int(sum(a.numpy().nbytes for a in arrays.values()))
    # 原始資料量以「訓練期每 5 分鐘一筆車速」計（float32）；不含輸入視窗的重複
    metrics["raw-bytes"] = int(len(d["y_train"]) * 4)
    metrics["train-seconds"] = time.perf_counter() - t0
    content = RecordDict({"arrays": arrays, "metrics": MetricRecord(metrics)})
    if cfg["verify_updates"]:
        attach_update_tag(content, load_vd_key(cfg["key_dir"], pid), rnd, pid)  # 只讀自己的金鑰
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = _cfg(context)
    pid, d = _client_data(context, cfg)
    rnd = int(msg.content["config"]["server-round"])
    _check_global_signature(cfg, msg, rnd)
    model = _model(cfg, msg)
    r = evaluate(model, d["X_test"], d["y_test"], d["mean"], d["std"])
    metrics = {
        "num-examples": len(d["y_test"]),
        "round": rnd,
        "partition-id": pid,
        "eval-loss": r["loss"],
        "mae-kmh": r["mae_kmh"],
        "mse-kmh": r["mse_kmh"],
        "naive-mae-kmh": r["naive_mae_kmh"],
    }
    if cfg["ft_epochs"] > 0 and rnd == cfg["rounds"]:
        metrics.update(_finetune(model, d, cfg, context, pid))
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


def _finetune(model, d, cfg: dict, context: Context, pid: int) -> dict:
    """個人化：最後一輪拿全域模型在本地再訓練 ft_epochs 個 epoch，只在本地使用、不上傳。

    用的是同一份私有資料，DP 時這幾步也要算進 ε（回報 ft-epsilon = 聯邦訓練 + 微調的總和）。
    """
    set_seed(cfg["seed"] + 1000 * pid + 999)
    import copy
    base = copy.deepcopy(model)
    model.train()  # 剛評估完是 eval 模式；Opacus 要求 train 模式
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["ft_lr"])
    loader = make_loader(d["X_train"], d["y_train"], cfg["batch_size"], shuffle=True,
                         seed=cfg["seed"] + 1000 * pid + 999)
    out = {}
    if cfg["use_dp"]:
        model, _, _, eps, _, _ = _dp_train(model, optimizer, loader, cfg, _dp_history(context),
                                           cfg["ft_epochs"])
        out["ft-epsilon"] = eps
    else:
        train(model, loader, optimizer, cfg["ft_epochs"], loss=cfg["loss"], grad_clip=cfg["nodp_grad_clip"])
    out["ft-diverged"] = 0
    if not all_finite(model):
        # 微調發散（無 DP、沒有逐樣本裁剪時可能發生）：保留全域模型，等於不微調
        model = base
        out["ft-diverged"] = 1
    out["ft-mae-kmh"] = evaluate(unwrap(model), d["X_test"], d["y_test"], d["mean"], d["std"])["mae_kmh"]
    return out
