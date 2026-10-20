"""實驗紀錄：彙整 client 指標並寫出 metrics.csv / clients.csv / config.json（純 Python）。"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import warnings
from pathlib import Path

ROUND_FIELDS = ["round", "n_train_clients", "n_eval_clients", "train_loss", "eval_loss",
                "mae_kmh", "rmse_kmh", "epsilon_max", "epsilon_mean"]
CLIENT_FIELDS = ["round", "partition", "vdid", "num_train", "train_loss", "epsilon",
                 "dp_steps", "sample_rate",
                 "num_test", "eval_loss", "mae_kmh", "rmse_kmh"]


def _wmean(replies, key):
    tot = sum(float(r["num-examples"]) for r in replies)
    return sum(float(r["num-examples"]) * float(r[key]) for r in replies) / tot if tot else float("nan")


def aggregate_train(replies: list[dict]) -> dict:
    out = {"train_loss": _wmean(replies, "train-loss")}
    eps = [float(r["epsilon"]) for r in replies if "epsilon" in r]
    if eps:
        out["epsilon_max"] = max(eps)
        out["epsilon_mean"] = sum(eps) / len(eps)
    return out


def aggregate_eval(replies: list[dict]) -> dict:
    return {
        "eval_loss": _wmean(replies, "eval-loss"),
        "mae_kmh": _wmean(replies, "mae-kmh"),
        "rmse_kmh": math.sqrt(_wmean(replies, "mse-kmh")),
    }


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
    tmp.replace(path)


class RunLogger:
    """每次收到彙整就重寫 CSV，即使中途中斷也留有紀錄。"""

    def __init__(self, out_dir, config: dict, extra: dict | None = None,
                 vdid_map: dict | None = None, expected_clients: int | None = None):
        self.dir = Path(out_dir)
        self.vdid_map = {int(k): v for k, v in (vdid_map or {}).items()}
        self.expected = expected_clients
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
        if self.expected is not None and n != self.expected:
            warnings.warn(f"第 {rnd} 輪 {phase} 只收到 {n}/{self.expected} 個 client 的回覆，"
                          "該輪指標只反映部分 client")

    def on_train(self, replies: list[dict]) -> dict:
        agg = aggregate_train(replies)
        rnd = int(replies[0]["round"]) if replies else 0
        self._check_count(rnd, len(replies), "train")
        self.rounds.setdefault(rnd, {"round": rnd}).update(agg, n_train_clients=len(replies))
        for r in replies:
            row = self._client(r)
            row.update(num_train=int(r["num-examples"]), train_loss=float(r["train-loss"]),
                       epsilon=float(r["epsilon"]) if "epsilon" in r else None,
                       dp_steps=int(r["dp-steps"]) if "dp-steps" in r else None,
                       sample_rate=float(r["sample-rate"]) if "sample-rate" in r else None)
        self._flush()
        return agg

    def on_eval(self, replies: list[dict]) -> dict:
        agg = aggregate_eval(replies)
        rnd = int(replies[0]["round"]) if replies else 0
        self._check_count(rnd, len(replies), "evaluate")
        self.rounds.setdefault(rnd, {"round": rnd}).update(agg, n_eval_clients=len(replies))
        for r in replies:
            row = self._client(r)
            row.update(num_test=int(r["num-examples"]), eval_loss=float(r["eval-loss"]),
                       mae_kmh=float(r["mae-kmh"]), rmse_kmh=math.sqrt(float(r["mse-kmh"])))
        self._flush()
        return agg

    def _flush(self):
        _write_csv(self.dir / "metrics.csv", ROUND_FIELDS,
                   [self.rounds[k] for k in sorted(self.rounds)])
        _write_csv(self.dir / "clients.csv", CLIENT_FIELDS,
                   [self.clients[k] for k in sorted(self.clients)])
