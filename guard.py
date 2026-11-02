"""伺服器端的安全閘道（包在 FedAvg／DP 策略外層）。

每輪：
1. 下發：替全域模型加上 Ed25519 簽章（sign-models=true），VD 驗證通過才訓練。
2. 收回：剔除斷線（回覆帶錯誤）的 VD，讓其他 VD 照常聚合（可用性）；
   可選擇模擬網路中間人竄改（tamper-rate）；再用各 VD 的金鑰驗 HMAC（verify-updates=true），
   驗不過的更新不進入聚合（完整性）。
3. 把通過的更新交給內層策略（DP 裁剪＋加噪、FedAvg／中位數／截尾平均）。
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from secagg import unmask_sum
from security import sign_model, sign_update, tamper, verify_update

AUTH = "auth"


def arrays_of(record: ArrayRecord) -> dict:
    return {k: v.numpy() for k, v in record.items()}


def _array_record(arrays: dict) -> ArrayRecord:
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()})


def attach_update_tag(content: RecordDict, key: bytes, rnd: int, pid: int) -> None:
    """VD 端：替要上傳的模型參數加上 HMAC。"""
    content[AUTH] = ConfigRecord({"tag": sign_update(key, arrays_of(content["arrays"]), rnd, pid)})


def screen_replies(contents: list[RecordDict], rnd: int, vd_keys: dict, verify: bool,
                   tamper_rate: float, rng: np.random.Generator) -> tuple[list[bool], dict]:
    """回傳每個回覆是否可用於聚合，以及統計（被竄改數、被拒絕數）。會移除 auth 紀錄。"""
    keep, n_tampered, n_rejected = [], 0, 0
    for c in contents:
        if tamper_rate > 0 and rng.random() < tamper_rate:
            c["arrays"] = _array_record(tamper(arrays_of(c["arrays"]), 0.5, rng))
            n_tampered += 1
        tag = str(c[AUTH]["tag"]) if AUTH in c else ""
        if AUTH in c:
            del c[AUTH]
        ok = True
        if verify:
            pid = int(c["metrics"]["partition-id"])
            key = vd_keys.get(pid)  # 驗證表裡沒有的 VD 一律拒絕
            ok = bool(tag) and key is not None and verify_update(key, arrays_of(c["arrays"]), rnd, pid, tag)
        n_rejected += int(not ok)
        keep.append(ok)
    return keep, {"n_tampered": n_tampered, "n_rejected": n_rejected}


def secure_average(contents: list[RecordDict], expected: int) -> ArrayRecord | None:
    """安全聚合：加總遮罩後的更新再除以總樣本數。少了任何 VD，遮罩無法抵銷，回傳 None 放棄這一輪。"""
    if len(contents) != expected or not contents:
        return None
    masked = [arrays_of(c["arrays"]) for c in contents]
    total = sum(float(c["metrics"]["num-examples"]) for c in contents)
    avg = unmask_sum(masked, total)
    return _array_record({k: avg[k].astype(np.float32) for k in masked[0]})


class GuardedStrategy(Strategy):
    def __init__(self, inner: Strategy, signing_key: Ed25519PrivateKey | None, vd_keys: dict | None,
                 tamper_rate: float = 0.0, seed: int = 0,
                 on_round: Callable[[int, dict], None] | None = None, secure_agg_clients: int = 0):
        """signing_key=None：不簽全域模型；vd_keys=None：不驗證上傳更新。"""
        self.inner = inner
        self.vd_keys = vd_keys or {}
        self.verify_updates = vd_keys is not None
        self.tamper_rate = tamper_rate
        self.rng = np.random.default_rng(seed + 7919)
        self.on_round = on_round
        self._priv = signing_key
        # > 0：安全聚合，需要這麼多個 VD 全部回覆遮罩後的更新（基本版不容許斷線）
        self.secure_agg_clients = secure_agg_clients

    def _sign(self, messages: Iterable[Message], arrays: ArrayRecord, rnd: int) -> list[Message]:
        messages = list(messages)
        if self._priv is not None and messages:
            sig = sign_model(self._priv, arrays_of(arrays), rnd)
            for m in messages:  # 同一輪所有訊息共用同一份 RecordDict
                m.content[AUTH] = ConfigRecord({"sig": sig})
        return messages

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                        grid: Grid) -> Iterable[Message]:
        return self._sign(self.inner.configure_train(server_round, arrays, config, grid), arrays, server_round)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                        ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        ok_replies = [m for m in replies if not m.has_error()]
        keep, stats = screen_replies([m.content for m in ok_replies], server_round, self.vd_keys,
                                     self.verify_updates, self.tamper_rate, self.rng)
        stats["n_dropped"] = len(replies) - len(ok_replies)
        kept = [m for m, k in zip(ok_replies, keep) if k]
        if self.on_round:
            self.on_round(server_round, stats)
        if not kept:
            return None, None  # 這一輪沒有可信的更新：維持原本的全域模型
        if self.secure_agg_clients:
            arrays = secure_average([m.content for m in kept], self.secure_agg_clients)
            stats["secagg_failed"] = int(arrays is None)
            if self.on_round:
                self.on_round(server_round, stats)
            # 內層只用來彙整指標（它對遮罩後參數的平均沒有意義，直接丟掉）
            _, metrics = self.inner.aggregate_train(server_round, kept)
            return arrays, metrics
        return self.inner.aggregate_train(server_round, kept)

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                           grid: Grid) -> Iterable[Message]:
        return self._sign(self.inner.configure_evaluate(server_round, arrays, config, grid),
                          arrays, server_round)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        return self.inner.aggregate_evaluate(server_round, [m for m in replies if not m.has_error()])

    def summary(self) -> None:
        self.inner.summary()
