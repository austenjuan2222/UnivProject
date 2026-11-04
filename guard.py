"""伺服器端的安全閘道（包在 FedAvg／DP 策略外層）。

每輪：
1. 下發：替全域模型加上 Ed25519 簽章（sign-models=true），VD 驗證通過才訓練。
2. 收回：剔除斷線（回覆帶錯誤）的 VD，讓其他 VD 照常聚合（可用性）；
   可選擇模擬網路中間人竄改（tamper-rate）；再用各 VD 的金鑰驗 HMAC（verify-updates=true），
   驗不過的更新不進入聚合（完整性）。
3. 把通過的更新交給內層策略（DP 裁剪＋加噪、FedAvg／中位數／截尾平均）。
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from secagg import drop_correction, unmask_sum
from security import sign_model, sign_update, tamper, update_digest, verify_update

AUTH = "auth"


def arrays_of(record: ArrayRecord) -> dict:
    return {k: v.numpy() for k, v in record.items()}


def _array_record(arrays: dict) -> ArrayRecord:
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()})


def attach_update_tag(content: RecordDict, key: bytes, rnd: int, pid: int) -> None:
    """VD 端：替要上傳的模型參數加上 HMAC。"""
    content[AUTH] = ConfigRecord({"tag": sign_update(key, arrays_of(content["arrays"]), rnd, pid)})


def screen_replies(contents: list[RecordDict], rnd: int, vd_keys: dict, verify: bool,
                   tamper_rate: float, rng: np.random.Generator, replay_rate: float = 0.0,
                   last: dict | None = None) -> tuple[list[bool], dict]:
    """回傳每個回覆是否可用於聚合，以及統計（被竄改、被重送、被拒絕數）。會移除 auth 紀錄。
    replay_rate > 0：模擬中間人把該 VD 上一輪的整份訊息（參數＋簽章）再送一次；last 由呼叫端保存跨輪的上一份訊息。"""
    keep, n_tampered, n_replayed, n_rejected, audit = [], 0, 0, 0, []
    for c in contents:
        pid = int(c["metrics"]["partition-id"])
        tag = str(c[AUTH]["tag"]) if AUTH in c else ""
        if last is not None:
            prev, last[pid] = last.get(pid), (c["arrays"], tag)
            if prev is not None and replay_rate > 0 and rng.random() < replay_rate:
                c["arrays"], tag = prev
                n_replayed += 1
        if tamper_rate > 0 and rng.random() < tamper_rate:
            c["arrays"] = _array_record(tamper(arrays_of(c["arrays"]), 0.5, rng))
            n_tampered += 1
        if AUTH in c:
            del c[AUTH]
        ok = True
        if verify:
            key = vd_keys.get(pid)  # 驗證表裡沒有的 VD 一律拒絕
            ok = bool(tag) and key is not None and verify_update(key, arrays_of(c["arrays"]), rnd, pid, tag)
        n_rejected += int(not ok)
        keep.append(ok)
        # 稽核紀錄：哪個 VD、哪一輪、送了什麼（內容雜湊）、附了什麼簽章、有沒有被採用
        audit.append({"round": int(rnd), "pid": pid, "accepted": ok,
                      "digest": update_digest(arrays_of(c["arrays"]), rnd, pid), "tag": tag})
    return keep, {"n_tampered": n_tampered, "n_replayed": n_replayed, "n_rejected": n_rejected, "audit": audit}


def update_delta(local: dict, global_: dict) -> np.ndarray | None:
    """VD 上傳減全域模型，攤平成一條向量；參數名稱、形狀不符或含 NaN／inf 時回傳 None。"""
    if local.keys() != global_.keys() or any(local[k].shape != global_[k].shape for k in local):
        return None
    d = np.concatenate([np.ravel(local[k] - global_[k]) for k in global_])
    return d if np.isfinite(d).all() else None


class CollusionFilter:
    """跨輪偵測串通（概念同 FoolsGold，Fung et al. 2018）：每輪算每個 VD 更新量和其他 VD 的最大餘弦相似度，
    跨輪平均後，明顯高於多數 VD 的（中位數 + z × 1.4826 × MAD）就標記。單輪看不出來的小幅同向偏移，
    因為每輪都一起偏，累積後會浮出來。被標記的 VD 之後的更新不進入聚合。"""

    def __init__(self, z: float = 3.0, min_rounds: int = 2):
        self.z, self.min_rounds = z, min_rounds
        self.total: dict[int, float] = {}
        self.count: dict[int, int] = {}

    def update(self, updates: dict[int, np.ndarray]) -> set[int]:
        pids = list(updates)
        if len(pids) >= 3:
            m = np.stack([np.ravel(updates[p]).astype(np.float64) for p in pids])
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
            sim = m @ m.T
            np.fill_diagonal(sim, -np.inf)
            for p, s in zip(pids, sim.max(axis=1)):
                if not np.isfinite(s):
                    continue
                self.total[p] = self.total.get(p, 0.0) + float(s)
                self.count[p] = self.count.get(p, 0) + 1
        return self.flagged()

    def flagged(self) -> set[int]:
        means = {p: self.total[p] / self.count[p] for p in self.total if self.count[p] >= self.min_rounds}
        if len(means) < 3:
            return set()
        v = np.array(list(means.values()))
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
        return {p for p, s in means.items() if s > med + self.z * mad}


class ValidationFilter:
    """伺服器用自己的參考資料檢查每份更新（概念同 Zeno，Xie et al. 2019；FLTrust，Cao et al. 2021）：
    score(更新後的模型) 越大越差；比這一輪多數 VD 明顯差（中位數 + z × 1.4826 × MAD）的不聚合。
    只拒絕「變差」的一側；分數不是有限值的也拒絕。需要伺服器看得到個別更新，不能和安全聚合一起用。"""

    def __init__(self, score: Callable[[dict], float], z: float = 3.0, min_updates: int = 3):
        self.score, self.z, self.min_updates = score, z, min_updates

    def screen(self, updates: dict[int, dict]) -> set[int]:
        scores = {}
        for p, a in updates.items():
            try:
                scores[p] = float(self.score(a))
            except (RuntimeError, ValueError, KeyError, TypeError):  # 形狀或參數名稱不符
                scores[p] = float("nan")
        bad = {p for p, s in scores.items() if not np.isfinite(s)}
        v = np.array([s for p, s in scores.items() if p not in bad])
        if len(v) < self.min_updates:  # 太少份無法判斷誰是離群值，只拒絕壞掉的
            return bad
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
        return bad | {p for p, s in scores.items() if p not in bad and s > med + self.z * mad}


def secure_average(contents: list[RecordDict], expected: int, revealed: dict | None = None) -> ArrayRecord | None:
    """安全聚合：加總遮罩後的更新再除以總樣本數。少了 VD 時，需要在線 VD 交出的成對種子（revealed）
    才能扣掉沒抵銷的遮罩；拿不到就回傳 None 放棄這一輪。"""
    if not contents or (len(contents) != expected and revealed is None):
        return None
    masked = [arrays_of(c["arrays"]) for c in contents]
    total = sum(float(c["metrics"]["num-examples"]) for c in contents)
    avg = unmask_sum(masked, total, drop_correction(revealed, masked[0]) if revealed else None)
    return _array_record({k: avg[k].astype(np.float32) for k in masked[0]})


class GuardedStrategy(Strategy):
    def __init__(self, inner: Strategy, signing_key: Ed25519PrivateKey | None, vd_keys: dict | None,
                 tamper_rate: float = 0.0, seed: int = 0,
                 on_round: Callable[[int, dict], None] | None = None, secure_agg_clients: int = 0,
                 on_audit: Callable[[list], None] | None = None,
                 collusion_filter: CollusionFilter | None = None, replay_rate: float = 0.0,
                 validation_filter: ValidationFilter | None = None):
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
        self.on_audit = on_audit
        self.collusion_filter = collusion_filter
        self.replay_rate = replay_rate
        self.validation_filter = validation_filter
        self._last: dict = {}  # 每個 VD 上一輪的訊息（模擬重送用）
        self._global: dict = {}
        self._grid: Grid | None = None

    def _sign(self, messages: Iterable[Message], arrays: ArrayRecord, rnd: int) -> list[Message]:
        messages = list(messages)
        if self._priv is not None and messages:
            sig = sign_model(self._priv, arrays_of(arrays), rnd)
            for m in messages:  # 同一輪所有訊息共用同一份 RecordDict
                m.content[AUTH] = ConfigRecord({"sig": sig})
        return messages

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                        grid: Grid) -> Iterable[Message]:
        self._global = arrays_of(arrays)
        self._grid = grid  # 安全聚合斷線復原時要再問一次在線 VD
        return self._sign(self.inner.configure_train(server_round, arrays, config, grid), arrays, server_round)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                        ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        ok_replies = [m for m in replies if not m.has_error()]
        keep, stats = screen_replies([m.content for m in ok_replies], server_round, self.vd_keys,
                                     self.verify_updates, self.tamper_rate, self.rng,
                                     self.replay_rate, self._last if self.replay_rate > 0 else None)
        stats["n_dropped"] = len(replies) - len(ok_replies)
        audit = stats.pop("audit")
        if self.on_audit and self.verify_updates:
            self.on_audit(audit)
        kept = [m for m, k in zip(ok_replies, keep) if k]
        if self.collusion_filter is not None and kept:
            deltas = {id(m): update_delta(arrays_of(m.content["arrays"]), self._global) for m in kept}
            bad = sum(d is None for d in deltas.values())
            stats["n_rejected"] = stats.get("n_rejected", 0) + bad  # 形狀不符或含 NaN／inf：不聚合也不計分
            kept = [m for m in kept if deltas[id(m)] is not None]
            pid = {id(m): int(m.content["metrics"]["partition-id"]) for m in kept}
            flagged = self.collusion_filter.update({pid[id(m)]: deltas[id(m)] for m in kept})
            kept = [m for m in kept if pid[id(m)] not in flagged]
            stats["flagged"] = ";".join(str(p) for p in sorted(flagged))
        if self.validation_filter is not None and kept:
            pid = {id(m): int(m.content["metrics"]["partition-id"]) for m in kept}
            invalid = self.validation_filter.screen({pid[id(m)]: arrays_of(m.content["arrays"]) for m in kept})
            kept = [m for m in kept if pid[id(m)] not in invalid]
            stats["val_rejected"] = ";".join(str(p) for p in sorted(invalid))
        if self.on_round:
            self.on_round(server_round, stats)
        if not kept:
            return None, None  # 這一輪沒有可信的更新：維持原本的全域模型
        if self.secure_agg_clients:
            # 收到了卻被拒絕的 VD（例如 HMAC 不符）不能當成斷線：交出它的種子等於讓伺服器解開它已送達的更新
            rejected = len(ok_replies) > len(kept)
            revealed = (self._recover(server_round, kept)
                        if len(kept) < self.secure_agg_clients and not rejected else None)
            stats["secagg_recovered"] = int(revealed is not None)
            arrays = secure_average([m.content for m in kept], self.secure_agg_clients, revealed)
            stats["secagg_failed"] = int(arrays is None)
            if self.on_round:
                self.on_round(server_round, stats)
            # 內層只用來彙整指標（它對遮罩後參數的平均沒有意義，直接丟掉）
            _, metrics = self.inner.aggregate_train(server_round, kept)
            return arrays, metrics
        return self.inner.aggregate_train(server_round, kept)

    def _recover(self, rnd: int, kept: list[Message]) -> dict | None:
        """向還在線的 VD 要它們和斷線 VD 之間的成對種子；任何一台沒回應就放棄（回傳 None）。"""
        alive = {int(m.content["metrics"]["partition-id"]): m.metadata.src_node_id for m in kept}
        dropped = sorted(set(range(self.secure_agg_clients)) - set(alive))
        if self._grid is None or not dropped:
            return None
        q = RecordDict({"secagg": ConfigRecord({"dropped": dropped, "round": int(rnd)})})
        replies = list(self._grid.send_and_receive(
            [Message(content=q, message_type="query", dst_node_id=n) for n in alive.values()], timeout=120))
        pid_of = {node: pid for pid, node in alive.items()}
        revealed = {}
        for r in replies:
            if r.has_error():
                return None
            try:
                c = r.content["secagg"]
                pid, js, seeds = int(c["pid"]), [int(j) for j in c["dropped"]], [str(x) for x in c["seeds"]]
            except (KeyError, TypeError, ValueError):
                return None
            # 回覆要來自它自稱的 VD、涵蓋整份斷線名單，種子是 32 bytes 的十六進位
            if (pid_of.get(r.metadata.src_node_id) != pid or sorted(js) != dropped or len(seeds) != len(js)
                    or not all(re.fullmatch(r"[0-9a-f]{64}", x) for x in seeds)):
                return None
            revealed[pid] = dict(zip(js, seeds))
        return revealed if set(revealed) == set(alive) else None

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                           grid: Grid) -> Iterable[Message]:
        return self._sign(self.inner.configure_evaluate(server_round, arrays, config, grid),
                          arrays, server_round)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        return self.inner.aggregate_evaluate(server_round, [m for m in replies if not m.has_error()])

    def summary(self) -> None:
        self.inner.summary()
