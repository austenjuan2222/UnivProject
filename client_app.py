"""ClientApp（Flower 1.26 Message API）：每個 client = 一個 VD 偵測器。

- 資料：data.load_client_arrays（依時間切 train/test，標準化只用 train）。
- DP：use-dp=true 時以 Opacus 做 DP-SGD（Poisson 取樣、逐樣本裁剪、加噪）。
- ε：每輪把（σ, q, 步數）存進 context.state，用全部歷史計算累計 ε 回報。
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import numpy as np

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from compress import quantize_update
from data import load_client_arrays, n_calendar
from guard import AUTH, arrays_of, attach_update_tag
from secagg import (neighbors, share_holders, share_threshold, consist_mac, consistency_threshold, count_clipped, eph_bytes, ephemeral_key, epub_mac, encrypt_shares, load_agreement_keys, mask_update, open_share, reveal_seeds,
                    seal_seeds, seal_shares, seed_tag, self_seed)
from security import (adaptive_shrink, drops_out, hide_congestion, is_attacker, load_server_public_key, load_vd_key, load_vd_signing_key, verify_recover,
                      poison_update, verify_model)
from task import TrafficModel, all_finite, jam_scorer, evaluate, make_loader, set_seed, train, unwrap

warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
logging.getLogger("flwr").setLevel(logging.INFO)

DP_STATE_KEY = "dp-history"
SECAGG_STATE_KEY = "secagg-state"

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
        "attack_ids": tuple(int(i) for i in str(c.get("attack-ids", "")).split(",") if i.strip()),
        "attack_type": str(c.get("attack-type", "flip")),
        "attack_scale": float(c.get("attack-scale", 10.0)),
        "attack_shift_kmh": float(c.get("attack-shift-kmh", 30.0)),
        "attack_adaptive_tol": float(c.get("attack-adaptive-tol", 0.0)),
        "dropout_rate": float(c.get("dropout-rate", 0.0)),
        "secure_agg": bool(c.get("secure-agg", False)),
        "secagg_self_mask": bool(c.get("secagg-self-mask", True)),
        "secagg_shamir": bool(c.get("secagg-shamir", False)),
        "secagg_neighbors": int(c.get("secagg-neighbors", 0)),
        "secagg_bits": int(c.get("secagg-bits", 32)),
        "secagg_frac_bits": int(c.get("secagg-frac-bits", 12)),
        "upload_bits": int(c.get("upload-bits", 32)),
        "update_auth": str(c.get("update-auth", "hmac")),
        "edge_runtime": str(c.get("edge-runtime", "torch")),
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


_ATTACKER_SCORERS: dict = {}


def _attacker_scorer(context: Context, cfg: dict):
    """自適應攻擊者用的評分函式：最壞情況假設它拿得到伺服器的參考 VD（同 validation-dir 的挑法）。每個行程只建一次。"""
    c = dict(context.run_config)
    key = str(c.get("validation-dir", ""))
    if not key:
        raise ValueError("attack-adaptive-tol > 0 需要設定 validation-dir（攻擊者要模仿伺服器的檢查）")
    if key not in _ATTACKER_SCORERS:
        from server_app import _manifest, _reference_sets  # 只在自適應攻擊時才需要
        fed = {x["vdid"] for x in _manifest(cfg["data_dir"]).get("clients", [])}
        # 攻擊者猜伺服器用的是塞車最多的那幾個（attack-ref-offset）；伺服器若用 validation-offset 保密，兩者就不同
        refs = _reference_sets(c, fed, int(c.get("validation-vds", 4)), int(c.get("attack-ref-offset", 0)))
        n_extra = n_calendar(cfg["calendar"]) + int(cfg["lag_day"])
        _ATTACKER_SCORERS[key] = jam_scorer(refs, lambda: TrafficModel(cfg["lookback"], cfg["hidden"], cfg["residual"], n_extra))
    return _ATTACKER_SCORERS[key]


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


def _numpy_dp_train(model, X, y, cfg: dict, prev, pid: int, rnd: int):
    """同樣的 DP-SGD，但用只靠 numpy 的 edge_numpy（模擬記憶體小的 VD）；回傳格式和 _dp_train 一樣。"""
    from edge_numpy import dp_train, predict

    from privacy import append_history, compute_epsilon, history_tuples

    if not cfg["residual"] or cfg["loss"] != "mse":
        raise ValueError("edge-runtime='numpy' 只支援 residual=true、loss='mse'")
    p = {k: v.detach().numpy().copy() for k, v in unwrap(model).state_dict().items()}
    p, q, steps = dp_train(p, X, y, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["sigma"], cfg["clip"],
                           np.random.default_rng([cfg["seed"], pid, rnd, 11]))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in p.items()})
    loss = float(((predict(p, X) - y) ** 2).mean())  # 訓練後的 MSE（torch 版是訓練中各批平均，僅供參考）
    hist = append_history(prev, cfg["sigma"], q, steps)
    return model, loss, hist, compute_epsilon(history_tuples(hist), cfg["delta"]), steps, q


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    cfg = _cfg(context)
    pid, d = _client_data(context, cfg)
    rnd = int(msg.content["config"]["server-round"])
    if cfg["secure_agg"] and cfg["secagg_shamir"]:
        use_advertised_round(context, rnd)
    if drops_out(cfg["seed"], pid, rnd, cfg["dropout_rate"]):
        raise RuntimeError(f"模擬 VD {pid} 在第 {rnd} 輪斷線")  # 伺服器會收到錯誤回覆並略過這個 VD
    _check_global_signature(cfg, msg, rnd)
    set_seed(cfg["seed"] + 1000 * pid + rnd)
    t0 = time.perf_counter()

    model = _model(cfg, msg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    y_train = d["y_train"]
    if cfg["attack_type"] == "hidejam" and is_attacker(pid, int(context.node_config["num-partitions"]),
                                                       cfg["attack_clients"], cfg["attack_ids"]):
        y_train = hide_congestion(y_train, d["mean"], d["std"], shift_kmh=cfg["attack_shift_kmh"])  # 針對性攻擊：用竄改過的標籤訓練
    loader = make_loader(d["X_train"], y_train, cfg["batch_size"], shuffle=True,
                         seed=cfg["seed"] + 1000 * pid + rnd)
    # 每個 client 都要有同一組 metric 鍵，Flower 才能彙整（diverged 只在發散時設成 1）
    metrics = {"num-examples": len(d["y_train"]), "round": rnd, "partition-id": pid, "diverged": 0}
    if cfg["attack_adaptive_tol"] > 0:
        metrics["attack-alpha"] = -1.0  # Flower 要求每個回覆的指標鍵相同；-1 表示不是攻擊者

    if cfg["use_dp"] and cfg["edge_runtime"] == "numpy":
        model, loss, hist, eps, steps, sample_rate = _numpy_dp_train(model, d["X_train"], y_train, cfg,
                                                                      _dp_history(context), pid, rnd)
        context.state[DP_STATE_KEY] = MetricRecord(hist)
        metrics["epsilon"] = eps
        metrics["dp-steps"] = int(steps)
        metrics["sample-rate"] = sample_rate
    elif cfg["use_dp"]:
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
    if is_attacker(pid, n, cfg["attack_clients"], cfg["attack_ids"]):
        # 被入侵的 VD：送出惡意更新（它仍有合法金鑰，所以下面照樣簽章；MAC 擋不住這種攻擊）
        g = arrays_of(msg.content["arrays"])
        bad = poison_update(g, arrays_of(arrays), cfg["attack_type"], cfg["attack_scale"],
                            # collude：同一輪所有被入侵 VD 共用種子，才會往同一個方向偏移
                            np.random.default_rng([cfg["seed"], rnd] if cfg["attack_type"] == "collude"
                                                  else [cfg["seed"], pid, rnd]))
        if cfg["attack_adaptive_tol"] > 0:
            bad, metrics["attack-alpha"] = adaptive_shrink(g, bad, _attacker_scorer(context, cfg), cfg["attack_adaptive_tol"])
        arrays = ArrayRecord({k: Array(v) for k, v in bad.items()})
    upload_bytes = None
    if cfg["upload_bits"] < 32:
        # 量化更新量再上傳（LPWAN 情境）；伺服器收到的就是還原後的值
        q, upload_bytes = quantize_update(arrays_of(msg.content["arrays"]), arrays_of(arrays),
                                          cfg["upload_bits"], np.random.default_rng([cfg["seed"], pid, rnd, 7]))
        arrays = ArrayRecord({k: Array(v) for k, v in q.items()})
    if cfg["secure_agg"]:
        # 安全聚合：上傳「樣本數 × 參數 ＋ 成對遮罩」，伺服器只能還原所有 VD 的加總
        priv, pubs = load_agreement_keys(cfg["key_dir"], pid)
        # 上傳相對於（簽過名的）全域模型的差值：數值小，定點數範圍才夠；伺服器還原後加回全域模型
        g = arrays_of(msg.content["arrays"])
        delta = {k: np.asarray(v, dtype=np.float64) - g[k] for k, v in arrays_of(arrays).items()}
        ring = {"bits": cfg["secagg_bits"], "frac_bits": cfg["secagg_frac_bits"]}
        metrics["secagg-clipped"] = count_clipped(delta, len(d["y_train"]), n, **ring)
        pair = {}
        if cfg["secagg_shamir"]:  # 成對遮罩用這輪公告過的臨時金鑰（訓練斷線的 VD 才能由別人的份還原）
            if not cfg["secagg_self_mask"] or "secagg-eph" not in msg.content.keys():
                raise RuntimeError(f"VD {pid} 第 {rnd} 輪沒收到臨時公鑰，拒絕遮罩")
            pair["pair_priv"], pair["pair_pubs"] = pair_keys_from_config(priv, pubs, pid, n, rnd, context.run_id,
                                                                         msg.content["secagg-eph"],
                                                                         k=cfg["secagg_neighbors"],
                                                                         min_peers=int(context.node_config.get(
                                                                             "secagg-min-present", 0)))
        k = cfg["secagg_neighbors"] if cfg["secagg_shamir"] else 0
        # 秘密分享模式只和這輪公告過（伺服器轉交、驗過）的鄰居遮罩
        peers = sorted(pair["pair_pubs"]) if pair else neighbors(pid, n, k, rnd, context.run_id)
        masked = mask_update(delta, len(d["y_train"]), pid, peers, priv, pubs, rnd, context.run_id,
                             self_mask=cfg["secagg_self_mask"], **ring, **pair, num_vd=n)
        claim_mask_round(context, rnd, adv=[int(i) for i in msg.content["secagg-eph"]["present"]] if pair else None)
        arrays = ArrayRecord({k: Array(v) for k, v in masked.items()})
        if cfg["secagg_shamir"]:  # 自我遮罩種子切份、各自加密給持有者，經伺服器轉交
            b = self_seed(priv, pid, rnd, context.run_id)
            hold = [j for j in share_holders(pid, n, k, rnd, context.run_id) if j == pid or j in pair["pair_pubs"]]
            shares = encrypt_shares(b, pid, hold, priv, pubs, rnd, context.run_id,
                                    t=share_threshold(n, k))
            metrics["secagg-self-tag"] = seed_tag(b)  # HMAC 涵蓋：伺服器還原後拿來比對
        metrics["secagg-num-vd"] = n
        metrics["secagg-bits"], metrics["secagg-frac-bits"] = cfg["secagg_bits"], cfg["secagg_frac_bits"]
        metrics["secagg-self-mask"] = int(cfg["secagg_self_mask"])  # 伺服器比對設定一致（HMAC 有涵蓋）
    # Edge 運算成本：上傳量（模型參數）vs. 若把原始訓練資料傳上雲的量
    metrics["upload-bytes"] = (upload_bytes if upload_bytes is not None
                               else int(sum(a.numpy().nbytes for a in arrays.values())))
    # 原始資料量以「訓練期每 5 分鐘一筆車速」計（float32）；不含輸入視窗的重複
    metrics["raw-bytes"] = int(len(d["y_train"]) * 4)
    metrics["train-seconds"] = time.perf_counter() - t0
    content = RecordDict({"arrays": arrays, "metrics": MetricRecord(metrics)})
    if cfg["secure_agg"] and cfg["secagg_shamir"]:
        # 不在 HMAC 範圍內：每份都有 AEAD（綁擁有者、持有者、輪次），伺服器和中間人改不了也讀不到
        content["secagg-shares"] = ConfigRecord({str(j): bytes.fromhex(ct) for j, ct in shares.items()})  # 二進位，LPWAN 上省一半
    if cfg["verify_updates"]:
        attach_update_tag(content, _vd_key(cfg, pid), rnd, pid)
    return Message(content=content, reply_to=msg)


def _vd_key(cfg, pid):
    return (load_vd_signing_key(cfg["key_dir"], pid) if cfg["update_auth"] == "ed25519"
            else load_vd_key(cfg["key_dir"], pid))  # 只讀自己的金鑰


def _hex(x) -> str:
    return x.hex() if isinstance(x, bytes) else str(x)


def pair_keys_from_config(priv, pubs: dict, pid: int, n: int, rnd: int, run: int, rec, k: int = 0,
                          min_peers: int = 0) -> tuple:
    """驗證伺服器轉交的臨時公鑰：每一把都要有擁有者用兩兩長期金鑰做的 MAC，自己那把要是自己的。"""
    import hmac
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    eph = ephemeral_key(priv, pid, rnd, run)
    try:
        epubs, macs = [_hex(x) for x in rec["epubs"]], [_hex(x) for x in rec["macs"]]
        if len(epubs) != n or len(macs) != n or epubs[pid] != eph.public_key().public_bytes_raw().hex():
            raise ValueError
        # 在場名單（伺服器說這輪有公告的 VD）：自己要在裡面、至少一致性門檻那麼多台（不然伺服器把誠實 VD 都說成
        # 缺席，就能讓這台只和共謀者遮罩），而且鄰居的公鑰有沒有給要和名單一致；名單之後綁進一致性證明
        present = {int(i) for i in rec["present"]}
        if pid not in present or len(present) < consistency_threshold(n) or any(not 0 <= i < n for i in present):
            raise ValueError
        # 只需要（也只驗）鄰居的臨時公鑰；空的表示那台這輪沒公告，不和它遮罩
        nb = [i for i in neighbors(pid, n, k, rnd, run) if epubs[i]]
        # 本地下限（伺服器改不了）：伺服器可把某台的鄰居集中說成缺席，只剩門檻數的遮罩對象
        if set(nb) != set(neighbors(pid, n, k, rnd, run)) & present or len(nb) < min_peers:
            raise ValueError
        if not all(hmac.compare_digest(macs[i], epub_mac(priv, pubs[i], i, pid, rnd, run, epubs[i], k=k)) for i in nb):
            raise ValueError
        # 在場的持有者（含自己）要夠門檻：不然伺服器謊稱鄰居都沒公告，就能讓這台只剩一兩個成對遮罩
        if len([j for j in share_holders(pid, n, k, rnd, run) if j == pid or j in nb]) < share_threshold(n, k):
            raise ValueError
        keys = {i: X25519PublicKey.from_public_bytes(bytes.fromhex(epubs[i])) for i in nb}
        for i in nb:
            eph.exchange(keys[i])  # 低階點（例如全零）會丟 ValueError：當成驗證失敗，不讓一台惡意 VD 弄當大家
        return eph, keys
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(f"VD {pid} 第 {rnd} 輪的臨時公鑰驗證失敗") from None


ADV_KEY = SECAGG_STATE_KEY + "-advertised"  # 上次公告的輪次
SKIP_KEY = SECAGG_STATE_KEY + "-skipped"  # 最近跳過（沒公告到）的輪次


def use_advertised_round(context: Context, rnd: int) -> None:
    """訓練的輪次要等於剛公告的那輪（不能挑之前公告過的某輪；同一輪重複訓練由 claim_mask_round 擋）。"""
    adv = context.state[ADV_KEY] if ADV_KEY in context.state else None
    if adv is None or int(adv["round"]) != int(rnd):
        raise RuntimeError(f"第 {rnd} 輪不是剛公告的輪次，拒絕訓練")


def claim_mask_round(context: Context, rnd: int, adv: list | None = None) -> None:
    """每輪只遮罩一次：重送同一輪（或更舊）的訓練請求會重設「每輪只回答一次復原」，讓惡意伺服器問兩次不同名單。"""
    st = context.state[SECAGG_STATE_KEY] if SECAGG_STATE_KEY in context.state else None
    if st is not None and int(st["masked-round"]) >= int(rnd):
        raise RuntimeError(f"第 {rnd} 輪已經遮罩過（或比上次更舊），拒絕重新遮罩")
    context.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": int(rnd), "revealed-round": -1,
                                                    "attested-round": -1})
    if adv is not None:  # 這輪在場（公告過）的鄰居＋自己：復原時只交這些 VD 的份
        context.state[SECAGG_STATE_KEY + "-present"] = ConfigRecord({"list": json.dumps(sorted(int(j) for j in adv))})


@app.query()
def secagg_recover_fn(msg: Message, context: Context) -> Message:
    """安全聚合斷線復原：交出自己和斷線 VD 之間的成對種子，讓伺服器扣掉那些沒抵銷的遮罩。"""
    pid = int(context.node_config["partition-id"])
    n = int(context.node_config["num-partitions"])
    q = msg.content["secagg"]
    rnd, dropped = int(q["round"]), sorted({int(j) for j in (q["dropped"] if "dropped" in q else [])})
    phase = str(q["phase"]) if "phase" in q else ""
    if phase == "advertise":
        # 公告回合（訓練前）：這輪的臨時公鑰＋給每台的 MAC，臨時私鑰切份加密給每台（經伺服器轉交）
        if not (bool(context.run_config.get("secagg-shamir", False))
                and bool(context.run_config.get("secagg-self-mask", True))):
            # 沒有自我遮罩時交出臨時私鑰的份，伺服器謊稱這台斷線就能解開它的更新
            raise RuntimeError(f"VD {pid} 沒開秘密分享＋自我遮罩，不公告臨時金鑰")
        # 輪次只能連續往前（第一次是 1）：鄰居圖由 (run, 輪次) 決定，伺服器跳輪就能挑一張對它有利的圖
        # shortcut: 伺服器仍可「只公告不訓練」一輪一輪往前試（每試一次就廢一輪，看得出來）；不鎖「公告過要訓練過」
        # 是因為斷線的 VD 丟例外時狀態不會存，會被永久鎖住；要擋時改用 VD 共同 commit-reveal 的洗牌種子
        # 容許連續漏收 2 次公告（網路掉包、公告時斷線）：代價是伺服器每輪最多能在 3 張圖裡挑，而且每跳一輪就廢一輪
        # shortcut: 連續漏收 3 次以上的 VD 會被鎖到這次執行結束（下次執行狀態重來），這期間它只是不在場
        if drops_out(int(context.run_config.get("seed", 0)), pid, rnd + 200_000,
                     float(context.run_config.get("secagg-advertise-dropout", 0.0))):
            raise RuntimeError(f"VD {pid} 第 {rnd} 輪公告時斷線（模擬）")
        last = int(context.state[ADV_KEY]["round"]) if ADV_KEY in context.state else 0
        # 跳過的輪數在滑動視窗內也有上限（本地設定）：不然伺服器可以每輪都在 3 張圖裡挑；
        # 用視窗而不是整次執行總額，誠實斷線的 VD 漏收的輪次會過期，不會一路累積到被永久鎖住
        sk = SKIP_KEY in context.state and json.loads(str(context.state[SKIP_KEY]["list"])) or []
        win = int(context.node_config.get("secagg-skip-window", 20))
        sk = [r for r in [*sk, *range(last + 1, rnd)] if r > rnd - win]
        if not last < rnd <= last + 3 or len(sk) > int(context.node_config.get("secagg-max-skips", 6)):
            raise RuntimeError(f"VD {pid} 拒絕第 {rnd} 輪公告：上次是第 {last} 輪，最近 {win} 輪已跳過 {len(sk)} 輪")
        # shortcut: run_id 仍由伺服器決定，可離線試；要更強時改用 VD 共同 commit-reveal 的亂數當洗牌種子
        priv, pubs = load_agreement_keys(str(context.run_config.get("key-dir", "keys")), pid)
        eph = ephemeral_key(priv, pid, rnd, context.run_id)
        epub = eph.public_key().public_bytes_raw().hex()
        k = int(context.run_config.get("secagg-neighbors", 0))
        floor = int(context.node_config.get("secagg-min-neighbors", 0))  # VD 本地設定，伺服器改不了
        if k and k < floor:
            raise RuntimeError(f"VD {pid} 拒絕 k={k}：低於本地下限 {floor}")
        context.state[ADV_KEY] = MetricRecord({"round": rnd})  # 全部檢查過才記錄
        context.state[SKIP_KEY] = ConfigRecord({"list": json.dumps(sk)})
        nb, hold = set(neighbors(pid, n, k, rnd, context.run_id)), share_holders(pid, n, k, rnd, context.run_id)
        macs = [epub_mac(priv, pubs[j], pid, j, rnd, context.run_id, epub, k=k) if j in nb else "" for j in range(n)]
        cts = encrypt_shares(eph_bytes(eph), pid, hold, priv, pubs, rnd, context.run_id, t=share_threshold(n, k),
                             kind="eshare")
        # 二進位傳（LPWAN 上十六進位會多一倍）
        return Message(content=RecordDict({"secagg": ConfigRecord(
            {"pid": pid, "epub": bytes.fromhex(epub), "epub-macs": [bytes.fromhex(m) for m in macs],
             "eshares": [bytes.fromhex(cts[j]) if j in cts else b"" for j in range(n)]})}), reply_to=msg)
    consist = phase == "consist"
    # shortcut: 只模擬復原時斷線，一致性回合大家都回；要模擬一致性回合斷線時再加
    if not consist and drops_out(int(context.run_config.get("seed", 0)), pid, rnd + 100_000,
                 float(context.run_config.get("secagg-recover-dropout", 0.0))):
        # 模擬：上傳後、復原時才斷線的 VD
        raise RuntimeError(f"VD {pid} 第 {rnd} 輪復原時斷線（模擬）")
    st = context.state[SECAGG_STATE_KEY] if SECAGG_STATE_KEY in context.state else None
    # 限制惡意伺服器：只回答自己這一輪真的送過遮罩的請求、每輪只回答一次、自己不能在斷線名單裡、
    # 在線 VD 至少 2/3（伺服器不能謊稱大多數 VD 斷線來解開少數幾台的遮罩）
    if (st is None or int(st["masked-round"]) != rnd or int(st["revealed-round"]) == rnd or pid in dropped
            or any(not 0 <= j < n for j in dropped) or 3 * (n - len(dropped)) < 2 * n):
        raise RuntimeError(f"VD {pid} 拒絕第 {rnd} 輪的斷線復原請求")
    # 一律要伺服器簽章（不論 sign-models）：不然中間人可以假稱某台斷線來解開它已送達的遮罩
    sig = str(q["sig"]) if "sig" in q else ""
    if not (sig and verify_recover(load_server_public_key(str(context.run_config.get("key-dir", "keys"))),
                                   rnd, dropped, sig, context.run_id)):
        raise RuntimeError(f"VD {pid} 拒絕第 {rnd} 輪沒有伺服器簽章的斷線復原請求")
    priv, pubs = load_agreement_keys(str(context.run_config.get("key-dir", "keys")), pid)
    att_round = int(st["attested-round"]) if "attested-round" in st else -1
    att_key = SECAGG_STATE_KEY + "-attested"
    # 存整份名單（不存短雜湊：48 bits 的雜湊伺服器可以離線找碰撞，讓 VD 替一份名單作證、卻算進另一份）
    att_list = str(context.state[att_key]["list"]) if att_key in context.state else ""
    shamir = bool(context.run_config.get("secagg-shamir", False))
    knb = int(context.run_config.get("secagg-neighbors", 0)) if shamir else 0
    pk = SECAGG_STATE_KEY + "-present"  # 這輪在場名單（訓練時驗過、存下來的）
    if shamir and pk not in context.state:  # 沒照流程遮罩過：不當成全部在場
        raise RuntimeError(f"VD {pid} 第 {rnd} 輪沒有在場名單，拒絕復原")
    present = set(json.loads(str(context.state[pk]["list"]))) if pk in context.state else set(range(n))
    bind = sorted(present) if shamir else None
    # 在線＝在場名單 − 斷線名單（缺席的 VD 不會作證）；斷線名單只能是在場的 VD
    if shamir and (not set(dropped) <= present or len(present - set(dropped)) < consistency_threshold(n)):
        # 在線的少於一致性門檻：證明一定湊不滿，直接拒絕（例如 n=3 有任何一台斷線）
        raise RuntimeError(f"VD {pid} 第 {rnd} 輪在線 VD 不足一致性門檻，無法復原")
    if consist:
        # 一致性回合：替自己收到的名單向每台 VD 做 MAC；每輪只做一次（不能替兩份名單作證）
        if att_round == rnd:
            raise RuntimeError(f"VD {pid} 第 {rnd} 輪已經證明過名單")
        context.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": rnd, "revealed-round": int(st["revealed-round"]),
                                                        "attested-round": rnd})
        context.state[att_key] = ConfigRecord({"list": json.dumps(dropped)})
        macs = ["" if j == pid else consist_mac(priv, pubs[j], pid, j, rnd, context.run_id, dropped, k=knb, present=bind)
                for j in range(n)]
        return Message(content=RecordDict({"secagg": ConfigRecord({"pid": pid, "macs": macs})}), reply_to=msg)
    reply = {"pid": pid}
    if shamir:
        # 秘密分享要先過一致性回合：自己證明過同一份名單，而且湊滿 2n/3+1 份（含自己）別人對同一份名單的證明，
        # 不然惡意伺服器可以對不同 VD 說不同名單，同時拿到某台的份和它的所有成對種子
        frm = [int(i) for i in q["attest-from"]] if "attest-from" in q else []
        macs = [str(x) for x in q["attest-macs"]] if "attest-macs" in q else []
        import hmac
        ok = {i for i, m in zip(frm, macs) if i in present and i != pid and i not in dropped
              and hmac.compare_digest(m, consist_mac(priv, pubs[i], i, pid, rnd, context.run_id, dropped, k=knb,
                                                    present=bind))}
        if (att_round != rnd or att_list != json.dumps(dropped) or len(frm) != len(macs)
                or len(ok) + 1 < consistency_threshold(n)):
            raise RuntimeError(f"VD {pid} 第 {rnd} 輪的斷線名單沒有足夠一致的證明")
        # 秘密分享：不直接交出自己的種子，改交出名單上在線的 VD 的份；說它斷線的那台一律不交（只給成對種子）
        owners = [int(i) for i in q["share-owners"]] if "share-owners" in q else []
        cts = [str(x) for x in q["share-cts"]] if "share-cts" in q else []
        # 份只給「名單上在線」的那些 VD，而且要剛好是名單的補集（不能挑著要）
        held = set(range(n)) if not knb else set(neighbors(pid, n, knb, rnd, context.run_id))  # 我持有誰的份
        if len(owners) != len(cts) or sorted(owners) != sorted((held & present) - set(dropped)):
            raise RuntimeError(f"VD {pid} 拒絕第 {rnd} 輪交出斷線 VD 的份")
        got = {i: open_share(ct, i, pid, priv, pubs, rnd, context.run_id) for i, ct in zip(owners, cts)}
        if any(v is None for v in got.values()):
            raise RuntimeError(f"VD {pid} 第 {rnd} 輪收到解不開的份")
        sealed = seal_shares(got, pid, priv, pubs["server"], rnd, context.run_id)
        reply["share-owners"], reply["shares"] = list(sealed), list(sealed.values())
    elif bool(context.run_config.get("secagg-self-mask", True)):
        # 雙重遮罩（沒有秘密分享）：自己在線才交出自我遮罩種子，同樣加密給伺服器
        reply["self"] = seal_seeds({pid: self_seed(priv, pid, rnd, context.run_id).hex()}, pid, priv, pubs["server"],
                                   rnd, context.run_id)[pid]
    if shamir:
        # 斷線 VD 交出的是它臨時私鑰的份（不是成對種子），伺服器自己算它和每台在線 VD 的種子；
        # 擁有者要剛好是斷線名單（在線 VD 的臨時私鑰份一律不交，否則加上自我遮罩份就能解開它）
        eo = [int(i) for i in q["eshare-owners"]] if "eshare-owners" in q else []
        ec = [str(x) for x in q["eshare-cts"]] if "eshare-cts" in q else []
        if len(eo) != len(ec) or sorted(eo) != sorted(held & set(dropped)):
            raise RuntimeError(f"VD {pid} 拒絕第 {rnd} 輪交出在線 VD 的臨時私鑰份")
        eg = {i: open_share(ct, i, pid, priv, pubs, rnd, context.run_id, kind="eshare") for i, ct in zip(eo, ec)}
        if any(v is None for v in eg.values()):
            raise RuntimeError(f"VD {pid} 第 {rnd} 輪收到解不開的臨時私鑰份")
        es = seal_shares(eg, pid, priv, pubs["server"], rnd, context.run_id, kind="eshare")
        reply["eshare-owners"], reply["eshares"] = list(es), list(es.values())
        seeds = {}
    else:
        seeds = seal_seeds(reveal_seeds(pid, dropped, priv, pubs, rnd, context.run_id), pid, priv, pubs["server"],
                           rnd, context.run_id)
    # 所有檢查都過了才記下「這輪答過」（格式錯的請求不會耗掉唯一一次回答）
    context.state[SECAGG_STATE_KEY] = MetricRecord({"masked-round": rnd, "revealed-round": rnd,
                                                    "attested-round": att_round})
    reply["dropped"], reply["seeds"] = list(seeds), list(seeds.values())
    return Message(content=RecordDict({"secagg": ConfigRecord(reply)}), reply_to=msg)


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
        "num-jam": r["n_jam"],
        "jam-mae-kmh": r["jam_mae_kmh"],
        "jam-bias-kmh": r["jam_bias_kmh"],
    }
    if cfg["ft_epochs"] > 0 and rnd == cfg["rounds"]:
        metrics.update(_finetune(model, d, cfg, context, pid))
    content = RecordDict({"metrics": MetricRecord(metrics)})
    if cfg["verify_updates"]:
        attach_update_tag(content, _vd_key(cfg, pid), rnd, pid)  # 評估指標也簽，中間人改誤差會被發現
    return Message(content=content, reply_to=msg)


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
