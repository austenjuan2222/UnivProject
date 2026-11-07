"""實驗紀錄：彙整 client 指標並寫出 metrics.csv / clients.csv / config.json（純 Python）。"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
import warnings
from pathlib import Path

ROUND_FIELDS = ["round", "n_train_clients", "n_eval_clients", "train_loss", "eval_loss",
                "mae_kmh", "rmse_kmh", "jam_mae_kmh", "jam_bias_kmh", "epsilon_max", "epsilon_mean", "naive_mae_kmh", "local_mae_kmh",
                "ft_mae_kmh", "ft_epsilon_max", "n_diverged", "n_ft_diverged",
                "n_rejected", "n_eval_rejected", "n_tampered", "n_replayed", "n_dropped", "secagg_failed", "secagg_recovered", "flagged", "val_rejected", "n_clipped", "clipped", "upload_kb", "raw_kb", "train_seconds_max"]
CLIENT_FIELDS = ["round", "partition", "vdid", "num_train", "train_loss", "epsilon",
                 "dp_steps", "sample_rate", "local_mae_kmh",
                 "num_test", "eval_loss", "mae_kmh", "rmse_kmh", "naive_mae_kmh",
                 "ft_mae_kmh", "ft_epsilon", "diverged", "ft_diverged", "attack_alpha"]


def _wmean(replies, key):
    tot = sum(float(r["num-examples"]) for r in replies)
    return sum(float(r["num-examples"]) * float(r[key]) for r in replies) / tot if tot else float("nan")


def aggregate_train(replies: list[dict]) -> dict:
    out = {"train_loss": _wmean(replies, "train-loss"),
           # 本地訓練發散、改回傳全域模型的 client 數（見 client_app.train_fn）
           "n_diverged": sum(int(r.get("diverged", 0)) for r in replies)}
    if replies and all("local-mae-kmh" in r and "num-test" in r for r in replies):
        tot = sum(float(r["num-test"]) for r in replies)
        out["local_mae_kmh"] = sum(float(r["num-test"]) * float(r["local-mae-kmh"]) for r in replies) / tot
    # Edge 運算成本：每個 VD 每輪上傳的模型大小，對比「把原始訓練資料傳上雲」的大小
    if replies and all("upload-bytes" in r for r in replies):
        out["upload_kb"] = sum(float(r["upload-bytes"]) for r in replies) / len(replies) / 1024
    if replies and all("raw-bytes" in r for r in replies):
        out["raw_kb"] = sum(float(r["raw-bytes"]) for r in replies) / len(replies) / 1024
    if replies and all("train-seconds" in r for r in replies):
        out["train_seconds_max"] = max(float(r["train-seconds"]) for r in replies)
    eps = [float(r["epsilon"]) for r in replies if "epsilon" in r]
    if eps:
        out["epsilon_max"] = max(eps)
        out["epsilon_mean"] = sum(eps) / len(eps)
    return out


def aggregate_eval(replies: list[dict]) -> dict:
    out = {
        "eval_loss": _wmean(replies, "eval-loss"),
        "mae_kmh": _wmean(replies, "mae-kmh"),
        "rmse_kmh": math.sqrt(_wmean(replies, "mse-kmh")),
    }
    if replies and all("naive-mae-kmh" in r for r in replies):
        out["naive_mae_kmh"] = _wmean(replies, "naive-mae-kmh")
    if replies and all("num-jam" in r for r in replies):
        n = sum(float(r["num-jam"]) for r in replies)
        if n:
            out["jam_mae_kmh"] = sum(float(r["num-jam"]) * float(r["jam-mae-kmh"]) for r in replies) / n
            out["jam_bias_kmh"] = sum(float(r["num-jam"]) * float(r["jam-bias-kmh"]) for r in replies) / n
    if replies and all("ft-mae-kmh" in r for r in replies):
        out["ft_mae_kmh"] = _wmean(replies, "ft-mae-kmh")
    ft_eps = [float(r["ft-epsilon"]) for r in replies if "ft-epsilon" in r]
    if ft_eps:
        out["ft_epsilon_max"] = max(ft_eps)
    if any("ft-diverged" in r for r in replies):
        out["n_ft_diverged"] = sum(int(r.get("ft-diverged", 0)) for r in replies)
    return out


def make_run_name(cfg: dict) -> str:
    if cfg.get("run-name"):
        return str(cfg["run-name"])
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.get("use-dp"):
        return f"dp_sigma{cfg.get('noise-multiplier')}_{stamp}"
    return f"nodp_{stamp}"


def _write_csv(path: Path, fields, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows：目的檔被防毒、索引或讀檔的程式短暫開著
            if attempt == 19:
                raise
            time.sleep(0.25)


class RunLogger:
    """每次收到彙整就重寫 CSV，即使中途中斷也留有紀錄。"""

    def __init__(self, out_dir, config: dict, extra: dict | None = None,
                 vdid_map: dict | None = None, expected_clients: int | None = None,
                 expected_train: int | None = None):
        self.dir = Path(out_dir)
        self.vdid_map = {int(k): v for k, v in (vdid_map or {}).items()}
        self.expected = expected_clients
        self.expected_train = expected_train if expected_train is not None else expected_clients
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rounds: dict[int, dict] = {}
        self.clients: dict[tuple[int, int], dict] = {}
        meta = {"started_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_config": dict(config)}
        meta.update(extra or {})
        (self.dir / "config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _client(self, r):
        key = (int(r["round"]), int(r["partition-id"]))
        row = self.clients.setdefault(key, {"round": key[0], "partition": key[1]})
        row["vdid"] = self.vdid_map.get(key[1], "")
        return row

    def _check_count(self, rnd, n, phase):
        expected = self.expected_train if phase == "train" else self.expected
        if expected is not None and n != expected:
            warnings.warn(f"第 {rnd} 輪 {phase} 只收到 {n}/{expected} 個 client 的回覆，"
                          "該輪指標只反映部分 client")

    @staticmethod
    def _complete(replies: list[dict], keys: tuple, phase: str) -> list[dict]:
        """可用性：少欄位或 MSE 為負的回覆不記錄（不然紀錄程式會讓整個伺服器當掉），並發出警告。"""
        def ok(r):
            need = keys + (("jam-mae-kmh", "jam-bias-kmh") if "num-jam" in r else ())
            return all(k in r for k in need) and not float(r.get("mse-kmh", 0)) < 0
        good = [r for r in replies if ok(r)]
        if len(good) < len(replies):
            warnings.warn(f"{phase}：{len(replies) - len(good)} 份回覆欄位不完整，沒有記錄")
        return good

    def on_train(self, replies: list[dict]) -> dict:
        replies = self._complete(replies, ("round", "partition-id", "num-examples", "train-loss"), "train")
        agg = aggregate_train(replies)
        rnd = int(replies[0]["round"]) if replies else 0
        self._check_count(rnd, len(replies), "train")
        self.rounds.setdefault(rnd, {"round": rnd}).update(agg, n_train_clients=len(replies))
        for r in replies:
            row = self._client(r)
            row.update(num_train=int(r["num-examples"]), train_loss=float(r["train-loss"]),
                       epsilon=float(r["epsilon"]) if "epsilon" in r else None,
                       dp_steps=int(r["dp-steps"]) if "dp-steps" in r else None,
                       sample_rate=float(r["sample-rate"]) if "sample-rate" in r else None,
                       local_mae_kmh=float(r["local-mae-kmh"]) if "local-mae-kmh" in r else None,
                       diverged=int(r.get("diverged", 0)),
                       # 只供實驗分析：自適應攻擊者實際用的縮放比例（伺服器的驗證不看它）
                       attack_alpha=float(r["attack-alpha"]) if float(r.get("attack-alpha", -1)) >= 0 else None)
        self._flush()
        return agg

    def on_security(self, rnd: int, stats: dict) -> None:
        """安全閘道的每輪統計：被拒絕（簽章不符）、被竄改（模擬中間人）、斷線的 VD 數。"""
        self.rounds.setdefault(int(rnd), {"round": int(rnd)}).update(stats)
        self._flush()

    def on_audit(self, entries: list[dict]) -> None:
        """稽核紀錄：每份上傳更新一行 JSON（輪次、VD、內容雜湊、簽章、是否採用）。
        用 Ed25519 簽章時，伺服器只有公鑰，事後可以證明某份更新確實是那台 VD 送的（不可否認性）。"""
        with (self.dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def on_eval(self, replies: list[dict]) -> dict:
        replies = self._complete(replies, ("round", "partition-id", "num-examples", "eval-loss", "mae-kmh",
                                           "mse-kmh"), "evaluate")
        agg = aggregate_eval(replies)
        rnd = int(replies[0]["round"]) if replies else 0
        self._check_count(rnd, len(replies), "evaluate")
        self.rounds.setdefault(rnd, {"round": rnd}).update(agg, n_eval_clients=len(replies))
        for r in replies:
            row = self._client(r)
            row.update(num_test=int(r["num-examples"]), eval_loss=float(r["eval-loss"]),
                       mae_kmh=float(r["mae-kmh"]), rmse_kmh=math.sqrt(float(r["mse-kmh"])),
                       naive_mae_kmh=float(r["naive-mae-kmh"]) if "naive-mae-kmh" in r else None,
                       ft_mae_kmh=float(r["ft-mae-kmh"]) if "ft-mae-kmh" in r else None,
                       ft_epsilon=float(r["ft-epsilon"]) if "ft-epsilon" in r else None,
                       ft_diverged=int(r["ft-diverged"]) if "ft-diverged" in r else None)
        self._flush()
        return agg

    def _flush(self):
        _write_csv(self.dir / "metrics.csv", ROUND_FIELDS,
                   [self.rounds[k] for k in sorted(self.rounds)])
        _write_csv(self.dir / "clients.csv", CLIENT_FIELDS,
                   [self.clients[k] for k in sorted(self.clients)])
