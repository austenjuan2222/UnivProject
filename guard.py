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

from compress import quantize_update
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from secagg import (neighbors, prune_present, share_holders, share_threshold, drop_correction, open_seeds, open_shares, pair_seed, seed_tag, shamir_combine, shamir_threshold,
                    unmask_sum)
from security import canonical_metrics, sign_model, sign_recover, sign_update, tamper, update_digest, verify_update

AUTH = "auth"


def arrays_of(record: ArrayRecord) -> dict:
    return {k: v.numpy() for k, v in record.items()}


def _array_record(arrays: dict) -> ArrayRecord:
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()})


def attach_update_tag(content: RecordDict, key: bytes, rnd: int, pid: int) -> None:
    """VD 端：替要上傳的模型參數加上 HMAC。"""
    content[AUTH] = ConfigRecord({"tag": sign_update(key, _arrays_or_empty(content), rnd, pid, _meta(content))})


def _arrays_or_empty(c: RecordDict) -> dict:
    return arrays_of(c["arrays"]) if "arrays" in c else {}


def _meta(c: RecordDict) -> str:
    # 訓練和評估分開標示，評估的簽章不能拿去當訓練的用（反之亦然）
    return ("train|" if "arrays" in c else "eval|") + canonical_metrics(dict(c["metrics"]))


def _tag(c: RecordDict) -> str:
    # 沒驗過簽章的內容不能讓伺服器當掉：auth 少了 tag 或型別不對都當成沒簽
    try:
        return str(c[AUTH]["tag"]) if AUTH in c else ""
    except (KeyError, TypeError):
        return ""


def _has_records(c: RecordDict, *names: str) -> bool:
    # 只接受預期的紀錄：多出來的紀錄沒簽到，還會讓 Flower 的一致性檢查當掉
    return set(c.keys()) - {AUTH} == set(names)


def verify_reply(c: RecordDict, rnd: int, vd_keys: dict) -> bool:
    """驗證並移除 auth 紀錄；驗證表沒有的 VD、沒附簽章、或參數／指標被改過都回傳 False。"""
    tag = _tag(c)
    c.pop(AUTH, None)
    pid = int(c["metrics"]["partition-id"])
    key = vd_keys.get(pid)
    return bool(tag) and key is not None and verify_update(key, _arrays_or_empty(c), rnd, pid, tag, _meta(c))


def _well_formed_metrics(c: RecordDict) -> bool:
    """編號與 num-examples（FedAvg 的加權）都要是正整數。"""
    try:
        pid, n = float(c["metrics"]["partition-id"]), float(c["metrics"]["num-examples"])
    except (KeyError, TypeError, ValueError):
        return False
    # 絕對上限：只有兩三份更新時沒有中位數可比，自報天文數字的筆數也不能蓋過別人或讓加權總和溢位
    return pid.is_integer() and pid >= 0 and n.is_integer() and 0 < n <= 2**31


def _well_formed(c: RecordDict, kinds_ok: frozenset = frozenset("f")) -> bool:
    """可用性：指標合格、參數都是浮點數（安全聚合時是 mod 2^32 的 uint32）。不合格的連 HMAC 都不算（字串陣列會讓驗證本身當掉）。"""
    try:
        kinds = {v.numpy().dtype.kind for v in c["arrays"].values()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return _has_records(c, "arrays", "metrics") and _well_formed_metrics(c) and kinds <= kinds_ok


def screen_replies(contents: list[RecordDict], rnd: int, vd_keys: dict, verify: bool,
                   tamper_rate: float, rng: np.random.Generator, replay_rate: float = 0.0,
                   last: dict | None = None, kinds_ok: frozenset = frozenset("f")) -> tuple[list[bool], dict]:
    """回傳每個回覆是否可用於聚合，以及統計（被竄改、被重送、被拒絕數）。會移除 auth 紀錄。
    replay_rate > 0：模擬中間人把該 VD 上一輪的整份訊息（參數＋簽章）再送一次；last 由呼叫端保存跨輪的上一份訊息。"""
    keep, n_tampered, n_replayed, n_rejected, audit = [], 0, 0, 0, []
    for c in contents:
        if not _well_formed(c, kinds_ok):
            c.pop(AUTH, None)
            n_rejected += 1
            keep.append(False)
            pid = int(c["metrics"]["partition-id"]) if _well_formed_metrics(c) else -1
            audit.append({"round": int(rnd), "pid": pid, "accepted": False, "digest": "", "tag": ""})
            continue
        pid = int(c["metrics"]["partition-id"])
        tag = _tag(c)
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
            ok = bool(tag) and key is not None and verify_update(key, arrays_of(c["arrays"]), rnd, pid, tag, _meta(c))
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
    # 先轉成聚合用的 float32 再看：float64 的 1e300 本身有限，轉過去就變 inf
    ok = np.isfinite(d).all() and all(np.isfinite(local[k].astype(np.float32)).all() for k in local)
    return d if ok else None


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
    只拒絕「變差」的一側；分數不是有限值的也拒絕。需要伺服器看得到個別更新，不能和安全聚合一起用。
    margin > 0：另外要求至少比中位數差 margin（比例）才拒絕，避免多數更新擠在一起、MAD 很小時誤擋只差一點的 VD。
    group > 1：每輪把更新隨機分成每組 group 份，只看各組的平均（模擬「組內安全聚合、伺服器只看得到小組加總」），
    整組一起採用或拒絕；最後一組不足 group 份時併進前一組。
    memory > 0：記住每個 VD 每輪的可疑程度（(分數 − 中位數) / (1.4826 × MAD)，截在 0～z，好的輪次不能抵銷壞的），
    看過至少 3 輪、平均超過 memory 的也拒絕，用來抓每輪都只差一點、單輪門檻抓不到的攻擊者（只用在 group = 1）。
    歷史要用穩定的 VD 身分記：screen 的 ids 把每份訊息對應到它的 VD（伺服器驗過 HMAC 的編號）；同一 VD 同輪多份取最可疑的。
    strata：VD 編號 → 路段類別（例如依交通單位的歷史塞車比例分成「常塞車／順暢」），每類各自算中位數與門檻，
    避免幾乎不塞車的誠實路段只因為本地沒看過塞車就被當成可疑。需要 ids，只用在 group = 1。"""

    def __init__(self, score: Callable[[dict], float], z: float = 3.0, min_updates: int = 3, margin: float = 0.0,
                 group: int = 1, seed: int = 0, memory: float = 0.0, strata: dict | None = None):
        self.score, self.z, self.min_updates, self.margin, self.group = score, z, min_updates, margin, group
        self.memory, self.history, self.strata = memory, {}, strata
        if strata and group > 1:
            raise ValueError("validation-strata 只能搭配 validation-group = 1")
        if memory > 0 and group > 1:
            raise ValueError("validation-memory 只能搭配 validation-group = 1（分組時看不到個別 VD）")
        self.rng = np.random.default_rng(seed + 31337)
        self.last_scores: dict = {}

    def _score(self, a: dict) -> float:
        try:
            return float(self.score(a))
        except (RuntimeError, ValueError, KeyError, TypeError):  # 形狀或參數名稱不符
            return float("nan")

    def screen(self, updates: dict[int, dict], ids: dict | None = None) -> set[int]:
        if self.group > 1 and len(updates) >= 2 * self.group:
            keys = [list(updates)[i] for i in self.rng.permutation(len(updates))]
            groups = [keys[i:i + self.group] for i in range(0, len(keys), self.group)]
            if len(groups[-1]) < self.group:
                last = groups.pop()  # 先 pop 再合併：groups[-2] += groups.pop() 會寫錯位置、吃掉一整組
                groups[-1] += last
            gs = {}
            for g, members in enumerate(groups):
                try:
                    avg = {k: np.mean([np.asarray(updates[m][k], np.float64) for m in members], axis=0)
                           for k in updates[members[0]]}
                except (ValueError, KeyError):  # 組內形狀或參數名稱不一致
                    avg = None
                gs[g] = self._score(avg) if avg is not None else float("nan")
            self.last_scores = {m: gs[g] for g, members in enumerate(groups) for m in members}
            return {m for g in self._outliers(gs) for m in groups[g]}
        self.last_scores = {p: self._score(a) for p, a in updates.items()}
        mem_ids = ids if self.memory > 0 else None
        if not self.strata:
            return self._outliers(self.last_scores, mem_ids)
        if ids is None:
            raise ValueError("分層檢查需要每份訊息的 VD 編號（ids）")
        out, pooled = set(), None
        for label in {self.strata.get(ids[p]) for p in self.last_scores}:  # 不在表上的 VD 自成一類
            sub = {p: s for p, s in self.last_scores.items() if self.strata.get(ids[p]) == label}
            if sum(np.isfinite(v) for v in sub.values()) >= self.min_updates:
                out |= self._outliers(sub, mem_ids)
            else:  # 這類這輪太少份、自己判斷不了：併回全部一起比，不能就此不檢查（這幾份不記跨輪歷史）
                pooled = self._outliers(self.last_scores) if pooled is None else pooled
                out |= pooled & set(sub)
        return out

    def _outliers(self, scores: dict, ids: dict | None = None) -> set:
        bad = {p for p, s in scores.items() if not np.isfinite(s)}
        v = np.array([s for p, s in scores.items() if p not in bad])
        if len(v) < self.min_updates:  # 太少份無法判斷誰是離群值，只拒絕壞掉的
            return bad
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
        limit = med + max(self.z * mad, self.margin * abs(med))
        out = bad | {p for p, s in scores.items() if p not in bad and s > limit}
        if ids is not None:
            now: dict = {}
            for p, s in scores.items():
                zp = self.z if p in bad else min(max((s - med) / mad, 0.0), self.z)
                now[ids[p]] = max(now.get(ids[p], 0.0), zp)
            for vd, zp in now.items():
                self.history.setdefault(vd, []).append(zp)
            out |= {p for p in scores
                    if len(self.history[ids[p]]) >= 3 and np.mean(self.history[ids[p]]) > self.memory}
            # 同一個 VD 編號這輪送了不只一份：視為異常，全部拒絕（歷史照樣記最可疑的那份）
            dup = {vd for vd in now if sum(ids[p] == vd for p in scores) > 1}
            out |= {p for p in scores if ids[p] in dup}
        return out


def _hex(x) -> str:
    return x.hex() if isinstance(x, bytes) else str(x)


def secure_average(contents: list[RecordDict], expected: int, revealed: dict | None = None,
                   base: dict | None = None, frac_bits: int = 12, bits: int | None = None) -> ArrayRecord | None:
    """安全聚合：加總遮罩後的更新再除以總樣本數。少了 VD 時，需要在線 VD 交出的成對種子（revealed）
    才能扣掉沒抵銷的遮罩；拿不到就回傳 None 放棄這一輪。"""
    if not contents or (len(contents) != expected and revealed is None):
        return None
    masked = [arrays_of(c["arrays"]) for c in contents]
    total = sum(float(c["metrics"]["num-examples"]) for c in contents)
    try:
        if bits is not None and any(np.asarray(v).dtype.kind != "u" or np.asarray(v).dtype.itemsize * 8 != bits
                                    for m in masked for v in m.values()):
            return None  # 先擋寬度，drop_correction 才不會照攻擊者給的型別算
        avg = unmask_sum(masked, total, drop_correction(revealed, masked[0]) if revealed else None, frac_bits, bits)
    except ValueError:
        return None
    if base:  # VD 上傳的是相對於全域模型的差值
        avg = {k: avg[k] + np.asarray(base[k], dtype=np.float64) for k in avg}
    return _array_record({k: avg[k].astype(np.float32) for k in masked[0]})


class GuardedStrategy(Strategy):
    def __init__(self, inner: Strategy, signing_key: Ed25519PrivateKey | None, vd_keys: dict | None,
                 tamper_rate: float = 0.0, seed: int = 0,
                 on_round: Callable[[int, dict], None] | None = None, secure_agg_clients: int = 0,
                 on_audit: Callable[[list], None] | None = None,
                 collusion_filter: CollusionFilter | None = None, replay_rate: float = 0.0,
                 validation_filter: ValidationFilter | None = None, download_bits: int = 32,
                 norm_cap: float = 0.0, secagg_keys: tuple | None = None,
                 secagg_self_mask: bool = False, secagg_bits: int = 32, secagg_frac_bits: int = 12,
                 secagg_shamir: bool = False, secagg_neighbors: int = 0):
        """signing_key=None：不簽全域模型；vd_keys=None：不驗證上傳更新。
        norm_cap > 0：更新長度超過這輪中位數 norm_cap 倍的，縮回 norm_cap 倍。"""
        if norm_cap and (norm_cap < 1 or secure_agg_clients):
            raise ValueError("norm-cap 要 ≥ 1（小於 1 連中位數的誠實 VD 都會被縮），且不能和 secure-agg 一起用（伺服器看不到個別更新）")
        self.norm_cap = norm_cap
        self.inner = inner
        self.vd_keys = vd_keys or {}
        self.verify_updates = vd_keys is not None
        self.tamper_rate = tamper_rate
        self.rng = np.random.default_rng(seed + 7919)
        self.on_round = on_round
        self.secagg_keys = secagg_keys
        self.secagg_self_mask = secagg_self_mask
        self.secagg_ring = (int(secagg_bits), int(secagg_frac_bits))
        if secagg_shamir and not secagg_self_mask:
            raise ValueError("secagg-shamir 要搭配 secagg-self-mask=true（分享的是自我遮罩種子）")
        self.secagg_shamir = secagg_shamir
        self.secagg_k = int(secagg_neighbors) if secagg_shamir else 0
        if self.secagg_k and secure_agg_clients:
            neighbors(0, secure_agg_clients, self.secagg_k)  # 不合法的 k 先丟 ValueError
        self._shares: dict = {}  # 這輪各 VD 的自我遮罩種子加密份：_shares[擁有者][持有者] = 密文
        self._self_hash: dict = {}  # 各 VD 上傳時承諾的種子雜湊（HMAC 涵蓋的指標）
        self._epubs: dict = {}  # 這輪各 VD 公告的臨時公鑰（十六進位）
        self._eshares: dict = {}  # 臨時私鑰的加密份：_eshares[擁有者][持有者] = 密文
        self._priv = signing_key
        # > 0：安全聚合，需要這麼多個 VD 全部回覆遮罩後的更新（基本版不容許斷線）
        self.secure_agg_clients = secure_agg_clients
        self.on_audit = on_audit
        self.collusion_filter = collusion_filter
        self.replay_rate = replay_rate
        self.validation_filter = validation_filter
        self.validation_scores: list = []  # 每輪 (輪次, [(VD 編號, 驗證分數)])，事後分析誤擋用
        # 下行壓縮：只廣播「新全域模型 − VD 手上那一版」的量化差值；第一次（VD 手上沒有模型）送完整 float32
        self.download_bits = download_bits
        self.download_bytes: list[int] = []  # 每次實際廣播的位元組數（每個 VD 收到一樣的）
        self._bc_src: dict | None = None  # 上次廣播對應的真正全域模型
        self._bc: ArrayRecord | None = None  # 上次廣播出去（VD 手上）的版本
        self._bc_rng = np.random.default_rng(seed + 104729)
        self._last: dict = {}  # 每個 VD 上一輪的訊息（模擬重送用）
        # VD 編號 ↔ 送出它的節點：第一次通過 HMAC 時綁定（TOFU），之後別的節點拿同一編號（偷來的金鑰）或同節點換編號都拒絕
        self._node_of_pid: dict = {}
        self._pid_of_node: dict = {}
        self._global: dict = {}
        self._grid: Grid | None = None

    def _sign(self, messages: Iterable[Message], arrays: ArrayRecord, rnd: int) -> list[Message]:
        messages = list(messages)
        if self._priv is not None and messages:
            sig = sign_model(self._priv, arrays_of(arrays), rnd)
            for m in messages:  # 同一輪所有訊息共用同一份 RecordDict
                m.content[AUTH] = ConfigRecord({"sig": sig})
        return messages

    def _broadcast(self, arrays: ArrayRecord) -> ArrayRecord:
        if self.download_bits >= 32:
            return arrays
        a = arrays_of(arrays)
        if self._bc_src is not None and all(np.array_equal(a[k], self._bc_src[k]) for k in a):
            return self._bc  # 同一份全域模型（本輪評估 → 下一輪訓練）不重送
        if self._bc is None:
            q, n = quantize_update(a, a, 32, self._bc_rng)
        else:
            q, n = quantize_update(arrays_of(self._bc), a, self.download_bits, self._bc_rng)
        self._bc_src, self._bc = a, _array_record(q)
        self.download_bytes.append(n)
        return self._bc

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                        grid: Grid) -> Iterable[Message]:
        arrays = self._broadcast(arrays)
        self._global = arrays_of(arrays)
        self._grid = grid  # 安全聚合斷線復原時要再問一次在線 VD
        msgs = self.inner.configure_train(server_round, arrays, config, grid)
        if self.secure_agg_clients and self.secagg_shamir:
            msgs = self._advertise(server_round, msgs)
        return self._sign(msgs, arrays, server_round)

    def _advertise(self, rnd: int, messages) -> list[Message]:
        """公告回合：收集每台 VD 這輪的臨時公鑰、給別台的 MAC、臨時私鑰的加密份，再把公鑰和 MAC 附進各自的訓練訊息。"""
        messages, n = list(messages), self.secure_agg_clients
        q = RecordDict({"secagg": ConfigRecord({"round": int(rnd), "phase": "advertise"})})
        self._epubs, self._eshares, macs, pid_of = {}, {}, {}, {}
        for r in self._grid.send_and_receive(
                [Message(content=q, message_type="query", dst_node_id=m.metadata.dst_node_id) for m in messages],
                timeout=120):
            try:
                c = r.content["secagg"]
                if not all(isinstance(x, bytes) for x in [c["epub"], *c["epub-macs"], *c["eshares"]]):
                    continue  # 只收二進位（之後要 bytes.fromhex 轉回去，字串可能不是合法十六進位）
                pid, epub = int(c["pid"]), _hex(c["epub"])
                m0, e0 = [_hex(x) for x in c["epub-macs"]], [_hex(x) for x in c["eshares"]]
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if 0 <= pid < n and pid not in pid_of.values() and len(m0) == n and len(e0) == n \
                    and re.fullmatch(r"[0-9a-f]{64}", epub) \
                    and {j for j, e in enumerate(e0) if e} == set(share_holders(pid, n, self.secagg_k, rnd, self._run_id())) \
                    and {j for j, x in enumerate(m0) if x} == set(neighbors(pid, n, self.secagg_k, rnd, self._run_id())):
                self._epubs[pid], macs[pid], pid_of[r.metadata.src_node_id] = epub, m0, pid
                self._eshares[pid] = {j: e for j, e in enumerate(e0) if e}
        # 只在公告過的 VD 之間遮罩：沒公告的 VD 收不到公鑰（會拒絕遮罩），這輪當作不在場；
        # 每台 VD 自己檢查在場的持有者夠不夠門檻（伺服器謊稱別人沒公告來削弱它時會拒絕）
        # 先剔除在場持有者不夠門檻的 VD（反覆到穩定）：它反正會拒絕遮罩，留著會讓整輪的份湊不滿
        adv = prune_present(self._epubs, n, self.secagg_k, rnd, self._run_id())
        for i in set(self._epubs) - adv:
            del self._epubs[i], self._eshares[i]
            pid_of = {node: p for node, p in pid_of.items() if p != i}
        out = []
        for m in messages:
            pid = pid_of.get(m.metadata.dst_node_id)
            if pid is None:
                out.append(m)
                continue
            content = RecordDict(dict(m.content.items()))  # 同一輪的訊息共用 RecordDict，每台要自己的一份
            nb = set(neighbors(pid, n, self.secagg_k, rnd, self._run_id())) & adv  # 只轉交在場鄰居的公鑰
            content["secagg-eph"] = ConfigRecord({
                "epubs": [bytes.fromhex(self._epubs[i]) if i in nb or i == pid else b"" for i in range(n)],
                "macs": [bytes.fromhex(macs[i][pid]) if i in nb else b"" for i in range(n)],
                "present": sorted(adv)})  # VD 核對公鑰和名單一致，並把名單綁進一致性證明
            m.content = content
            out.append(m)
        return out

    def revoke(self, pid: int) -> None:
        """應用層撤銷：移除該 VD 的驗證金鑰，之後它（或偷到金鑰的人）送的更新一律拒絕；
        傳輸層的撤銷（flwr supernode unregister）只切斷節點，金鑰仍有效，兩者要一起做。"""
        if not self.verify_updates:
            raise ValueError("revoke 需要 verify-updates=true（沒有驗證時撤銷不會有任何效果）")
        self.vd_keys.pop(pid, None)
        # 只放開「編號 → 節點」（重新配發金鑰後可由新節點綁定）；被撤銷的節點仍綁在這個已無金鑰的編號上，不能改用別的編號
        self._node_of_pid.pop(pid, None)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                        ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        ok_replies = [m for m in replies if not m.has_error()]
        # 秘密分享的加密份不在 HMAC 範圍（每份有 AEAD），先拿出來，其他檢查照舊只看參數＋指標
        shares_of = {id(m): m.content.pop("secagg-shares") for m in ok_replies
                     if "secagg-shares" in m.content.keys()}
        keep, stats = screen_replies([m.content for m in ok_replies], server_round, self.vd_keys,
                                     self.verify_updates, self.tamper_rate, self.rng,
                                     self.replay_rate, self._last if self.replay_rate > 0 else None,
                                     # 安全聚合上傳的是 mod 2^32 的遮罩整數，一般模式只收浮點數
                                     frozenset("u") if self.secure_agg_clients else frozenset("f"))
        stats["n_dropped"] = len(replies) - len(ok_replies)
        audit = stats.pop("audit")
        if self.verify_updates:
            claims = [(i, int(m.content["metrics"]["partition-id"]), m.metadata.src_node_id)
                      for i, m in enumerate(ok_replies) if keep[i]]
            nodes_of: dict = {}
            first: dict = {}  # 每個節點每輪只算第一份；同一節點多送的不聚合（避免同一份更新被算兩次）
            for i, pid, node in claims:
                first.setdefault(node, i)
            for _, pid, node in claims:
                if self._pid_of_node.get(node, pid) == pid:  # 已綁別的編號的節點不算數，不能拿來擋別人
                    nodes_of.setdefault(pid, set()).add(node)
            for i, pid, node in claims:
                bn, bp = self._node_of_pid.get(pid), self._pid_of_node.get(node)
                # 先檢查兩邊再寫入：被拒絕的訊息不能留下綁定（否則可以搶先綁走別人的編號）；
                # 還沒綁的編號同一輪有多個節點搶時全部拒絕，不讓先到的贏；已綁的只認綁定的節點
                if (first[node] != i or (bn is not None and bn != node) or (bp is not None and bp != pid)
                        or (bn is None and len(nodes_of[pid]) > 1)):
                    keep[i] = audit[i]["accepted"] = False
                    stats["n_rejected"] += 1
                else:
                    self._node_of_pid[pid], self._pid_of_node[node] = node, pid
        kept = [m for m, k in zip(ok_replies, keep) if k]
        if kept and self._global:
            # 可用性：合法金鑰送來的 NaN／inf 或形狀不符的更新會讓全域模型變 NaN 或聚合當掉，一律不聚合
            deltas = {id(m): update_delta(arrays_of(m.content["arrays"]), self._global) for m in kept}
            stats["n_rejected"] += sum(d is None for d in deltas.values())
            kept = [m for m in kept if deltas[id(m)] is not None]
        if len(kept) >= 3:
            # 自報的資料筆數超過這輪中位數 10 倍：FedAvg 下等於一台 VD 蓋過其他所有人
            n = {id(m): float(m.content["metrics"]["num-examples"]) for m in kept}
            cap = 10 * float(np.median(list(n.values())))
            stats["n_rejected"] += sum(v > cap for v in n.values())
            kept = [m for m in kept if n[id(m)] <= cap]
        if self.collusion_filter is not None and kept:
            pid = {id(m): int(m.content["metrics"]["partition-id"]) for m in kept}
            flagged = self.collusion_filter.update({pid[id(m)]: deltas[id(m)] for m in kept})
            kept = [m for m in kept if pid[id(m)] not in flagged]
            stats["flagged"] = ";".join(str(p) for p in sorted(flagged))
        if self.validation_filter is not None and kept:
            # 內容完全相同的重複訊息（例如網路上被重送一份）只留一份，避免被當成「同一編號送兩份」而連帶拒絕誠實 VD
            seen, uniq = set(), []
            for m in kept:
                pid_m = int(m.content["metrics"]["partition-id"])
                d = update_digest(arrays_of(m.content["arrays"]), server_round, pid_m)
                if (pid_m, d) not in seen:
                    seen.add((pid_m, d))
                    uniq.append(m)
            kept = uniq
            # 以每份訊息為單位評分（不用 VD 自報的編號當鍵），同一編號送兩份也會各自被檢查
            invalid = self.validation_filter.screen({i: arrays_of(m.content["arrays"]) for i, m in enumerate(kept)},
                                                    {i: int(m.content["metrics"]["partition-id"]) for i, m in enumerate(kept)})
            stats["val_rejected"] = ";".join(sorted(str(int(kept[i].content["metrics"]["partition-id"]))
                                                    for i in invalid))
            self.validation_scores.append((server_round, [(int(kept[i].content["metrics"]["partition-id"]), v)
                                                          for i, v in self.validation_filter.last_scores.items()]))
            kept = [m for i, m in enumerate(kept) if i not in invalid]
        if self.norm_cap and self._global:
            # 放在驗證檢查之後：先用原始更新評分，放大的惡意更新整份被擋；縮小只套在通過的更新上
            # 送回全域模型（長度 0）不算進中位數，不然攻擊者可以一直送 0 把中位數往下拉、讓誠實 VD 被縮
            norms = {id(m): float(np.linalg.norm(deltas[id(m)])) for m in kept}
            ref = [v for v in norms.values() if v > 0]
            stats["n_clipped"] = 0
            # shortcut: 少於 3 份非零更新時不縮（沒有可信的中位數）；要更穩可改用前幾輪中位數的移動平均
            limit = self.norm_cap * float(np.median(ref)) if len(ref) >= 3 else float("inf")
            for m in kept:
                if norms[id(m)] > limit:
                    s = limit / norms[id(m)]
                    local = arrays_of(m.content["arrays"])
                    m.content["arrays"] = _array_record(
                        {k: (self._global[k] + s * (local[k] - self._global[k])).astype(np.float32) for k in local})
                    stats["n_clipped"] += 1
            stats["clipped"] = ";".join(sorted(str(int(m.content["metrics"]["partition-id"]))
                                               for m in kept if norms[id(m)] > limit))
        if self.secure_agg_clients and self.secagg_shamir:
            # 沒附完整份數的 VD 在復原時無法替它還原自我遮罩，當成被拒絕（雙重遮罩下等於斷線）
            def ids(m):
                pid = int(m.content["metrics"]["partition-id"])
                return {str(j) for j in share_holders(pid, self.secure_agg_clients, self.secagg_k, server_round,
                                                         self._run_id()) if j in self._epubs}
            good = {id(m) for m in kept if id(m) in shares_of and set(shares_of[id(m)].keys()) == ids(m)}
            stats["n_rejected"] += len(kept) - len(good)
            kept = [m for m in kept if id(m) in good]
            self._shares = {int(m.content["metrics"]["partition-id"]): {int(j): v.hex() if isinstance(v, bytes) else str(v)
                                                                        for j, v in shares_of[id(m)].items()} for m in kept}
            self._self_hash = {int(m.content["metrics"]["partition-id"]):
                               int(float(m.content["metrics"].get("secagg-self-tag", -1))) for m in kept}
        if self.on_audit and self.verify_updates:
            # 所有檢查跑完才寫：HMAC 對、但被格式／筆數／串通／驗證檢查擋掉的，也要記成沒採用
            final = {id(m) for m in kept}
            for i, m in enumerate(ok_replies):
                audit[i]["accepted"] = audit[i]["accepted"] and id(m) in final
            self.on_audit(audit)
        if self.on_round:
            self.on_round(server_round, stats)
        if not kept:
            return None, None  # 這一輪沒有可信的更新：維持原本的全域模型
        if self.secure_agg_clients:
            # 收到了卻被拒絕的 VD（例如 HMAC 不符）不能當成斷線：交出它的種子等於讓伺服器解開它已送達的更新
            rejected = len(ok_replies) > len(kept)
            # 雙重遮罩時每輪都要復原（收自我遮罩種子），沒有就不能還原加總
            # VD 自報有幾個值超出定點範圍被截（指標有 HMAC 保護）
            stats["secagg_clipped"] = int(sum(float(m.content["metrics"].get("secagg-clipped", 0)) for m in kept))
            need = len(kept) < self.secure_agg_clients or self.secagg_self_mask
            # 雙重遮罩時被拒絕的 VD 可以當成斷線：它的自我遮罩種子不會交出，交出成對種子也解不開它的更新
            ok_to_reveal = self.secagg_self_mask or not rejected
            # VD 和伺服器的雙重遮罩設定不一致時，扣遮罩會默默算錯：整輪放棄
            # VD 數（num-partitions）也要一致：不一致時成對遮罩對不上、定點範圍也可能溢位
            same = all(bool(float(m.content["metrics"].get("secagg-self-mask", 0))) == self.secagg_self_mask
                       and int(float(m.content["metrics"].get("secagg-num-vd", -1))) == self.secure_agg_clients
                       and (int(float(m.content["metrics"].get("secagg-bits", -1))),
                            int(float(m.content["metrics"].get("secagg-frac-bits", -99)))) == self.secagg_ring
                       for m in kept)
            revealed = self._recover(server_round, kept) if need and ok_to_reveal and same else None
            stats["secagg_recovered"] = int(revealed is not None and len(kept) < self.secure_agg_clients)
            arrays = (None if not same or (self.secagg_self_mask and revealed is None)
                      else secure_average([m.content for m in kept], self.secure_agg_clients, revealed,
                                          base=self._global, frac_bits=self.secagg_ring[1],
                                          bits=self.secagg_ring[0]))
            stats["secagg_failed"] = int(arrays is None)
            if self.on_round:
                self.on_round(server_round, stats)
            # 內層只用來彙整指標（它對遮罩後參數的平均沒有意義，直接丟掉）
            _, metrics = self.inner.aggregate_train(server_round, kept)
            return arrays, metrics
        return self.inner.aggregate_train(server_round, kept)

    def _seal_pair(self, pid: int):
        spriv, pubs = self.secagg_keys
        return spriv, pubs[pid]

    def _run_id(self) -> int:
        return int(getattr(getattr(self._grid, "run", None), "run_id", 0))

    def _ask(self, rnd: int, dropped: list) -> dict:
        ask = {"dropped": dropped, "round": int(rnd)}
        if self._priv is not None:  # 沒簽的話，沒有 TLS 時中間人可以假造「某台斷線」來解開它已送達的遮罩
            ask["sig"] = sign_recover(self._priv, rnd, dropped, self._run_id())
        return ask

    def _consist(self, rnd: int, dropped: list, alive: dict) -> dict:
        """一致性回合：每台替收到的名單向別台做 MAC，伺服器只負責轉交（改不了、偽造不了）。"""
        macs = {}
        q0 = RecordDict({"secagg": ConfigRecord({**self._ask(rnd, dropped), "phase": "consist"})})
        for r in self._grid.send_and_receive(
                [Message(content=q0, message_type="query", dst_node_id=n) for n in alive.values()], timeout=120):
            try:
                pid0 = int(r.content["secagg"]["pid"])
                m0 = [str(x) for x in r.content["secagg"]["macs"]]
            except (KeyError, TypeError, ValueError, AttributeError):
                continue  # 沒回或回錯：少一份證明，門檻由 VD 自己檢查
            if alive.get(pid0) == r.metadata.src_node_id and len(m0) == self.secure_agg_clients:
                macs[pid0] = m0
        return macs

    def _recover_query(self, rnd: int, dropped: list, owners: list, macs: dict):
        ask = self._ask(rnd, dropped)

        def query(j):
            c = dict(ask)
            if self.secagg_shamir:  # 每台只拿到「別人要給它」的那份密文，加上別人替名單做的證明
                mine = [i for i in owners if j in self._shares[i]]  # 只有 j 持有份的那些
                c["share-owners"] = mine
                c["share-cts"] = [self._shares[i][j] for i in mine]
                emine = [d for d in dropped if j in self._eshares[d]]  # 斷線 VD 的臨時私鑰份
                c["eshare-owners"] = emine
                c["eshare-cts"] = [self._eshares[d][j] for d in emine]
                frm = sorted(i for i in macs if i != j)
                c["attest-from"], c["attest-macs"] = frm, [macs[i][j] for i in frm]
            return RecordDict({"secagg": ConfigRecord(c)})
        return query

    def _recover(self, rnd: int, kept: list[Message]) -> dict | None:
        """向還在線的 VD 要它們和斷線 VD 之間的成對種子；任何一台沒回應就放棄（回傳 None）。"""
        alive = {int(m.content["metrics"]["partition-id"]): m.metadata.src_node_id for m in kept}
        everyone = set(self._epubs) if self.secagg_shamir and self._epubs else set(range(self.secure_agg_clients))
        if not set(alive) <= everyone:
            return None  # 沒公告的 VD 不該有遮罩過的上傳
        dropped = sorted(everyone - set(alive))  # 公告過、卻沒交上傳的；沒公告的 VD 這輪本來就不在
        if self._grid is None or not (dropped or self.secagg_self_mask):
            return None
        owners = sorted(alive)
        if self.secagg_shamir and not all(d in self._eshares for d in dropped):
            return None  # 斷線 VD 沒公告過臨時金鑰（不該發生：沒公告的 VD 收不到可遮罩的訓練訊息）
        macs = self._consist(rnd, dropped, alive) if self.secagg_shamir else {}
        query = self._recover_query(rnd, dropped, owners, macs)
        replies = list(self._grid.send_and_receive(
            [Message(content=query(pid), message_type="query", dst_node_id=n) for pid, n in alive.items()],
            timeout=120))
        pid_of = {node: pid for pid, node in alive.items()}
        revealed, shares, eshares = {}, {i: {} for i in owners}, {d: {} for d in dropped}
        for r in replies:
            if r.has_error():
                if self.secagg_shamir:
                    continue  # 秘密分享：復原時沉默的 VD 由別人的份還原
                return None
            try:
                c = r.content["secagg"]
                pid, js, seeds = int(c["pid"]), [int(j) for j in c["dropped"]], [str(x) for x in c["seeds"]]
            except (KeyError, TypeError, ValueError):
                return None
            # 回覆要來自它自稱的 VD、涵蓋整份斷線名單；加密的種子要解得開，明文的要是 32 bytes 的十六進位
            # 秘密分享模式不交成對種子（改交斷線 VD 的臨時私鑰份）
            want = [] if self.secagg_shamir else dropped
            if pid_of.get(r.metadata.src_node_id) != pid or sorted(js) != want or len(seeds) != len(js):
                return None
            got = dict(zip(js, seeds))
            if self.secagg_keys is not None:
                spriv, pubs = self.secagg_keys
                got = open_seeds(got, pid, spriv, pubs[pid], rnd, self._run_id()) if pid in pubs else None
                if got is None:
                    return None
            elif not all(re.fullmatch(r"[0-9a-f]{64}", x) for x in seeds):
                return None
            if self.secagg_shamir:
                try:
                    got_owners = [int(i) for i in c["share-owners"]]
                    sealed_list = [str(x) for x in c["shares"]]
                    if len(sealed_list) != len(got_owners):
                        return None
                    sealed = dict(zip(got_owners, sealed_list))
                except (KeyError, TypeError, ValueError):
                    return None
                opened = open_shares(sealed, pid, *self._seal_pair(pid), rnd, self._run_id())
                if sorted(got_owners) != [i for i in owners if pid in self._shares[i]] or opened is None:
                    return None
                for i, y in opened.items():
                    shares[i][pid] = y
                try:
                    eo, el = [int(i) for i in c["eshare-owners"]], [str(x) for x in c["eshares"]]
                except (KeyError, TypeError, ValueError):
                    return None
                eopened = (open_shares(dict(zip(eo, el)), pid, *self._seal_pair(pid), rnd, self._run_id(), kind="eshare")
                           if len(eo) == len(el) and sorted(eo) == [d for d in dropped if pid in self._eshares[d]]
                           else None)
                if eopened is None:
                    return None
                for d, y in eopened.items():
                    eshares[d][pid] = y
            elif self.secagg_self_mask:  # 雙重遮罩：每台在線 VD 都要交出自己的自我遮罩種子
                own = (open_seeds({pid: c["self"]}, pid, *self._seal_pair(pid), rnd, self._run_id())
                       if "self" in c and self.secagg_keys is not None else None)
                if own is None:
                    return None
                got = {**got, **own}
            revealed[pid] = got
        if self.secagg_shamir:
            t = share_threshold(self.secure_agg_clients, self.secagg_k)
            if any(len(shares[i]) < t for i in owners) or any(len(eshares[d]) < t for d in dropped):
                return None  # 某台的份不夠門檻
            for d in dropped:
                # 還原斷線 VD 這輪的臨時私鑰，自己算它和每台在線 VD（含沉默的）之間的成對種子
                e = shamir_combine(eshares[d])
                try:
                    ek = X25519PrivateKey.from_private_bytes(e) if e is not None else None
                except ValueError:
                    ek = None
                if ek is None or ek.public_key().public_bytes_raw().hex() != self._epubs.get(d):
                    return None  # 有人交了錯的份：和它公告的公鑰對不上
                for i in owners:
                    if d not in neighbors(i, self.secure_agg_clients, self.secagg_k, rnd, self._run_id()):
                        continue  # 不是鄰居就沒有成對遮罩
                    pub = X25519PublicKey.from_public_bytes(bytes.fromhex(self._epubs[i]))
                    revealed.setdefault(i, {})[d] = pair_seed(ek, pub, d, i, rnd, self._run_id()).hex()
            for i in owners:
                b = shamir_combine(shares[i])
                # 有人交了錯的份：還原出的種子和它上傳時承諾的雜湊對不上，整輪放棄（不默默算錯）
                if b is None or seed_tag(b) != self._self_hash.get(i):
                    return None
                revealed.setdefault(i, {})[i] = b.hex()
            return revealed
        return revealed if set(revealed) == set(alive) else None

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                           grid: Grid) -> Iterable[Message]:
        arrays = self._broadcast(arrays)
        return self._sign(self.inner.configure_evaluate(server_round, arrays, config, grid),
                          arrays, server_round)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        # 評估的 num-examples 是紀錄加權用：NaN 會讓紀錄程式當掉。誤差本身可以是 NaN（模型發散時要照實記錄）
        replies = [m for m in replies if not m.has_error()]
        ok = [m for m in replies if _has_records(m.content, "metrics") and _well_formed_metrics(m.content)]
        if self.verify_updates:
            # 評估指標也要驗：不然中間人改誤差就能讓紀錄（和根據紀錄下的結論）失真。
            # 編號和節點要雙向符合訓練時的綁定，每個節點只算第一份（重送不能讓它多算幾次）
            seen, good = set(), []
            for m in ok:
                node = m.metadata.src_node_id
                if not verify_reply(m.content, server_round, self.vd_keys) or node in seen:
                    continue
                pid = int(m.content["metrics"]["partition-id"])
                if self._node_of_pid.get(pid, node) == node and self._pid_of_node.get(node, pid) == pid:
                    seen.add(node)
                    good.append(m)
            ok = good
        if self.on_round:
            self.on_round(server_round, {"n_eval_rejected": len(replies) - len(ok)})
        return self.inner.aggregate_evaluate(server_round, ok)

    def summary(self) -> None:
        self.inner.summary()
