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
from flwr.serverapp.strategy import (DifferentialPrivacyServerSideAdaptiveClipping,
                                     DifferentialPrivacyServerSideFixedClipping, FedAvg, FedMedian,
                                     FedTrimmedAvg)

from data import load_client_arrays, n_calendar
from guard import CollusionFilter, GuardedStrategy, ValidationFilter
from secagg import provision_agreement_keys
from security import load_server_secrets, load_vd_verify_keys, provision_keys
from privacy import client_level_epsilon
from results_log import RunLogger, make_run_name
from task import JAM_KMH, TrafficModel, jam_scorer, set_seed

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


def _reference_sets(cfg: dict, exclude: set, n: int) -> list[dict]:
    """伺服器自己的參考 VD（不在聯邦裡，例如高公局自己管的路段）：取 validation-dir 裡塞車樣本最多的 n 個，只用訓練段。"""
    vdir = str(cfg["validation-dir"])
    clients = _manifest(vdir).get("clients", [])
    sets = []
    match = str(cfg.get("validation-match", ""))  # 例如 "-N-"：只用同方向（北向）的參考 VD
    for c in clients:
        if c["vdid"] in exclude or match not in c["vdid"]:
            continue
        d = load_client_arrays(vdir, int(c["partition"]), len(clients), int(cfg["lookback"]),
                               float(cfg.get("train-frac", 0.8)), int(cfg.get("horizon", 1)),
                               str(cfg.get("calendar", "")), bool(cfg.get("lag-day", False)))
        n_jam = int(((d["y_train"] * d["std"] + d["mean"]) < JAM_KMH).sum())
        sets.append((n_jam, c["vdid"], {"X": d["X_train"], "y": d["y_train"], "mean": d["mean"], "std": d["std"]}))
    sets.sort(key=lambda t: (-t[0], t[1]))
    if len(sets) < n:
        raise ValueError(f"validation-dir 只有 {len(sets)} 個不在聯邦裡的 VD，不夠 validation-vds={n}")
    print("伺服器參考 VD（塞車樣本數）：" + "、".join(f"{v}（{j}）" for j, v, _ in sets[:n]))
    return [r for _, _, r in sets[:n]]


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

    # client-level DP（選用）：保護「整個 VD 有沒有參與」，由伺服器裁剪各 client 的更新量再加噪。
    # 威脅模型不同：伺服器需可信（它看得到未裁剪的更新），保護的是對外發布的全域模型。
    # 雜訊量以「各 client 等權平均」計算；FedAvg 依樣本數加權，各 VD 樣本數相同時兩者一致。
    cl_sigma = float(cfg.get("client-dp-noise", 0.0))
    cl_clip = float(cfg.get("client-dp-clip", 0.1))
    # 每輪抽多少比例的 client 參與訓練（評估仍用全部 client）；< 1 時 client-level ε 有取樣放大
    frac = float(cfg.get("client-fraction", 1.0))
    if not 0 < frac <= 1:
        raise ValueError(f"client-fraction 必須在 (0, 1]，收到 {frac}")
    n_train = max(1, round(frac * num_clients))
    client_dp = None
    if cl_sigma > 0:
        q = n_train / num_clients
        client_dp = {"noise_multiplier": cl_sigma, "clipping_norm": cl_clip, "sample_rate": q,
                     "adaptive": bool(cfg.get("client-dp-adaptive", False)),
                     "epsilon": client_level_epsilon(cl_sigma, num_rounds, float(cfg["target-delta"]), q),
                     "delta": float(cfg["target-delta"])}
        print(f"client-level DP：σ={cl_sigma}、C={cl_clip}，{num_rounds} 輪後 ε={client_dp['epsilon']:.2f}")

    run_dir = Path(str(cfg["results-dir"])) / make_run_name(cfg)
    logger = RunLogger(run_dir, cfg, extra={
        "client_level_dp": client_dp,
        "versions": _versions(),
        "data": {k: manifest.get(k) for k in ("start", "end", "num_steps", "selection")},
        "clients": manifest.get("clients", []),
    }, vdid_map=vdids, expected_clients=num_clients, expected_train=n_train)

    _check_num_nodes(grid, num_clients)

    # 聚合方式：fedavg（加權平均）、median（逐參數中位數）、trimmed（去掉頭尾各 20% 再平均）
    # 中位數／截尾平均可抵抗少數被入侵 VD 的惡意更新（講義：IoT 攻擊面、botnet）
    agg_name = str(cfg.get("aggregator", "fedavg"))
    agg_cls = {"fedavg": FedAvg, "median": FedMedian, "trimmed": FedTrimmedAvg}.get(agg_name)
    if agg_cls is None:
        raise ValueError(f"aggregator 必須是 fedavg／median／trimmed，收到 {agg_name}")
    strategy = agg_cls(
        fraction_train=n_train / num_clients,
        fraction_evaluate=1.0,
        min_train_nodes=n_train,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        train_metrics_aggr_fn=lambda replies, _key: MetricRecord(
            logger.on_train([dict(r["metrics"]) for r in replies])),
        evaluate_metrics_aggr_fn=lambda replies, _key: MetricRecord(
            logger.on_eval([dict(r["metrics"]) for r in replies])),
    )
    if client_dp and client_dp["adaptive"]:
        # 自適應裁剪（Andrew et al. 2021）：每輪依「被裁剪的 client 比例」調整裁剪值，目標中位數；
        # 計數查詢也加噪，Flower 會調高更新量的雜訊讓總隱私等同 noise_multiplier，ε 計算不變。
        strategy = DifferentialPrivacyServerSideAdaptiveClipping(
            strategy, noise_multiplier=cl_sigma, num_sampled_clients=n_train,
            initial_clipping_norm=cl_clip)
    elif client_dp:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, noise_multiplier=cl_sigma, clipping_norm=cl_clip, num_sampled_clients=n_train)
    # 安全閘道（最外層）：簽全域模型、剔除斷線 VD、驗證上傳更新的 HMAC，再交給 DP／聚合
    verify = bool(cfg.get("verify-updates", False))
    sign = bool(cfg.get("sign-models", False))
    priv, vd_keys = None, None
    if verify or sign:
        # 模擬：第一次執行時產生金鑰（已有的不會換）；實際部署由安裝流程配發
        key_dir = str(cfg.get("key-dir", "keys"))
        provision_keys(key_dir, num_clients)
        priv, vd_keys = load_server_secrets(key_dir)
        auth = str(cfg.get("update-auth", "hmac"))
        if auth == "ed25519":
            vd_keys = load_vd_verify_keys(key_dir)  # 伺服器只拿公鑰：偽造不了 VD 的簽章
        elif auth != "hmac":
            raise ValueError(f"update-auth 必須是 hmac 或 ed25519，收到 {auth}")
    secure_agg = bool(cfg.get("secure-agg", False))
    if secure_agg:
        # 伺服器看不到個別更新：不能做伺服器端裁剪（client-level DP）或中位數；基本版也不容許斷線、抽樣
        if (client_dp or agg_name != "fedavg" or frac < 1
                or int(cfg.get("upload-bits", 32)) < 32 or bool(cfg.get("collusion-filter", False))
                or str(cfg.get("validation-dir", ""))):
            raise ValueError("secure-agg 只能搭配 aggregator='fedavg'、client-fraction=1、不開 client-level DP、"
                             "upload-bits=32、collusion-filter=false、不設 validation-dir（可以和 record-level DP 一起用）")
        provision_agreement_keys(str(cfg.get("key-dir", "keys")), num_clients)
    validation_filter = None
    if str(cfg.get("validation-dir", "")):
        refs = _reference_sets(cfg, set(vdids.values()), int(cfg.get("validation-vds", 4)))
        validation_filter = ValidationFilter(jam_scorer(refs, lambda: TrafficModel(
            int(cfg["lookback"]), int(cfg["hidden-size"]), bool(cfg.get("residual", False)),
            n_calendar(str(cfg.get("calendar", ""))) + int(bool(cfg.get("lag-day", False))))),
            float(cfg.get("validation-z", 3.0)), margin=float(cfg.get("validation-margin", 0.0)))
    strategy = GuardedStrategy(strategy, priv if sign else None, vd_keys if verify else None,
                               float(cfg.get("tamper-rate", 0.0)), int(cfg["seed"]), logger.on_security,
                               secure_agg_clients=num_clients if secure_agg else 0, on_audit=logger.on_audit,
                               collusion_filter=CollusionFilter(float(cfg.get("collusion-z", 3.0)))
                               if bool(cfg.get("collusion-filter", False)) else None,
                               replay_rate=float(cfg.get("replay-rate", 0.0)),
                               validation_filter=validation_filter,
                               download_bits=int(cfg.get("download-bits", 32)))
    init = ArrayRecord(TrafficModel(int(cfg["lookback"]), int(cfg["hidden-size"]),
                                    bool(cfg.get("residual", False)),
                                    n_calendar(str(cfg.get("calendar", "")))
                                    + int(bool(cfg.get("lag-day", False)))).state_dict())
    result = strategy.start(grid=grid, initial_arrays=init, num_rounds=num_rounds,
                            train_config=ConfigRecord({}), evaluate_config=ConfigRecord({}))

    torch.save(result.arrays.to_torch_state_dict(), run_dir / "final_model.pt")
    if strategy.validation_scores:
        (run_dir / "validation_scores.json").write_text(json.dumps(strategy.validation_scores), encoding="utf-8")
    if strategy.download_bytes:
        (run_dir / "download_bytes.json").write_text(json.dumps(strategy.download_bytes), encoding="utf-8")
        print(f"下行廣播 {len(strategy.download_bytes)} 次，每個 VD 共收 {sum(strategy.download_bytes)} bytes")
    print(f"\n實驗紀錄已寫到 {run_dir}")
