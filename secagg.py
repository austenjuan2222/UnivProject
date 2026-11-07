"""安全聚合（Bonawitz et al. 2017 的成對遮罩，基本版）：伺服器只看得到加總，看不到單一 VD 的更新。

對應講義「機密性」：即使伺服器（或偷看伺服器的人）不可信，也拿不到個別 VD 的模型更新。
做法：每對 VD (i, j) 用 X25519 金鑰交換得到只有兩人知道的種子，各自產生同一份遮罩 r_ij；
i < j 時 i 加上 r_ij、j 減去 r_ij，加總時全部抵銷。每輪遮罩不同（種子含輪次）。

限制（如實記錄）：
- 斷線復原：在線 VD 交出和斷線 VD 之間的成對種子，伺服器扣掉這些遮罩（reveal_seeds、drop_correction）；種子加密給伺服器（seal_seeds），中間人讀不到。
  這是簡化版：沒有雙重遮罩與秘密分享，若斷線 VD 的更新事後又送到，伺服器就看得到它（完整的 SecAgg+ 解決這點）。
- 伺服器看不到個別更新，所以不能和「伺服器端裁剪的 client-level DP」或中位數聚合一起用；
  可以和 record-level DP（在 VD 本地加噪）一起用。
- 用 float64 加遮罩示範；正式實作在有限體上做整數運算，抵銷才會完全精確。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

_RAW = serialization.Encoding.Raw
MASK_RANGE = 1e3  # 遮罩的均勻分布範圍 ±1000，遠大於模型參數的尺度


def _new_x25519(path: Path) -> X25519PrivateKey:
    if not path.exists():
        raw = X25519PrivateKey.generate().private_bytes(_RAW, serialization.PrivateFormat.Raw,
                                                        serialization.NoEncryption())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw.hex())
    return X25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))


def provision_agreement_keys(key_dir, num_clients: int) -> None:
    """每個 VD 一把 X25519 私鑰（vd-<編號>.x25519），公鑰表 x25519_pub.json 給所有 VD 看。
    伺服器也有一把（server/agree.x25519，公鑰 x25519_server.pub），VD 交出的復原種子加密給它。"""
    key_dir = Path(key_dir)
    (key_dir / "server").mkdir(parents=True, exist_ok=True)
    spub = _new_x25519(key_dir / "server" / "agree.x25519").public_key()
    (key_dir / "x25519_server.pub").write_text(spub.public_bytes(_RAW, serialization.PublicFormat.Raw).hex(),
                                               encoding="utf-8")
    pub_path = key_dir / "x25519_pub.json"
    pubs = json.loads(pub_path.read_text(encoding="utf-8")) if pub_path.exists() else {}
    for pid in range(num_clients):
        priv = _new_x25519(key_dir / f"vd-{pid}.x25519")
        pubs[str(pid)] = priv.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex()
    pub_path.write_text(json.dumps(pubs), encoding="utf-8")


def load_agreement_keys(key_dir, pid: int) -> tuple[X25519PrivateKey, dict]:
    key_dir = Path(key_dir)
    priv = X25519PrivateKey.from_private_bytes(bytes.fromhex((key_dir / f"vd-{pid}.x25519").read_text().strip()))
    return priv, load_agreement_pubs(key_dir)


def load_agreement_pubs(key_dir) -> dict:
    """公鑰表（VD 編號 → 公鑰，另有 "server"）；伺服器只需要這個，不讀任何 VD 私鑰。"""
    key_dir = Path(key_dir)
    pubs = json.loads((key_dir / "x25519_pub.json").read_text(encoding="utf-8"))
    out = {int(k): X25519PublicKey.from_public_bytes(bytes.fromhex(v)) for k, v in pubs.items()}
    out["server"] = X25519PublicKey.from_public_bytes(
        bytes.fromhex((key_dir / "x25519_server.pub").read_text(encoding="utf-8").strip()))
    return out


def load_server_agreement_key(key_dir) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(
        bytes.fromhex((Path(key_dir) / "server" / "agree.x25519").read_text().strip()))


def seal_seeds(seeds: dict, me: int, my_priv: X25519PrivateKey, their_pub: X25519PublicKey, rnd: int,
               run: int = 0) -> dict[int, str]:
    """VD 用和伺服器的 X25519 共享金鑰把種子 XOR 加密（伺服器用同一函式、換成自己的私鑰和 VD 公鑰解開）。
    金鑰流綁執行、輪次、這對 VD，每個只用一次。沒有 TLS 時中間人讀回覆也拿不到種子。"""
    shared = my_priv.exchange(their_pub)
    def pad(j):
        return hashlib.sha256(shared + f"secagg-seal|{int(run)}|{int(rnd)}|{int(me)}|{int(j)}".encode()).digest()
    return {int(j): bytes(a ^ b for a, b in zip(bytes.fromhex(x), pad(j))).hex() for j, x in seeds.items()}


def pair_seed(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, me: int, other: int, rnd: int,
              run: int = 0) -> bytes:
    # 綁執行編號：金鑰跨次執行保留，不綁的話上一次執行交出的種子能解開這次同一輪的遮罩
    shared = my_priv.exchange(their_pub)
    a, b = sorted((me, other))
    return hashlib.sha256(shared + f"secagg|{int(run)}|{a}|{b}|{rnd}".encode()).digest()


def _mask(seed: bytes, shape) -> np.ndarray:
    rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint32))
    return rng.uniform(-MASK_RANGE, MASK_RANGE, size=shape)


def mask_update(update: dict, n: int, me: int, pids, my_priv, pubs: dict, rnd: int, run: int = 0) -> dict:
    """回傳 n × 更新 ＋ 成對遮罩（float64）。伺服器把所有 VD 的結果加總後除以總樣本數即得加權平均。"""
    out = {k: n * np.asarray(v, dtype=np.float64) for k, v in update.items()}
    for other in pids:
        if other == me:
            continue
        seed = pair_seed(my_priv, pubs[other], me, other, rnd, run)
        sign = 1.0 if me < other else -1.0
        for i, k in enumerate(sorted(out)):
            out[k] += sign * _mask(seed + i.to_bytes(4, "little"), out[k].shape)
    return out


def reveal_seeds(me: int, dropped, my_priv, pubs: dict, rnd: int, run: int = 0) -> dict[int, str]:
    """斷線復原：還在線的 VD 交出自己和每個斷線 VD 之間的成對種子（只有這幾對，伺服器還是解不開在線 VD 的遮罩）。"""
    return {int(j): pair_seed(my_priv, pubs[int(j)], me, int(j), rnd, run).hex() for j in dropped if int(j) != me}


def drop_correction(revealed: dict, like: dict) -> dict:
    """revealed[i][j] = VD i 交出的 (i, j) 種子；回傳在線 VD 為了斷線 VD 加上、卻沒被抵銷的遮罩總和。"""
    out = {k: np.zeros_like(v, dtype=np.float64) for k, v in like.items()}
    for i, seeds in revealed.items():
        for j, seed in seeds.items():
            sign = 1.0 if int(i) < int(j) else -1.0
            for idx, k in enumerate(sorted(out)):
                out[k] += sign * _mask(bytes.fromhex(seed) + idx.to_bytes(4, "little"), out[k].shape)
    return out


def unmask_sum(masked: list[dict], total_n: float, correction: dict | None = None) -> dict:
    return {k: (sum(m[k] for m in masked) - (correction[k] if correction else 0)) / total_n for k in masked[0]}
