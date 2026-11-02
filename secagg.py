"""安全聚合（Bonawitz et al. 2017 的成對遮罩，基本版）：伺服器只看得到加總，看不到單一 VD 的更新。

對應講義「機密性」：即使伺服器（或偷看伺服器的人）不可信，也拿不到個別 VD 的模型更新。
做法：每對 VD (i, j) 用 X25519 金鑰交換得到只有兩人知道的種子，各自產生同一份遮罩 r_ij；
i < j 時 i 加上 r_ij、j 減去 r_ij，加總時全部抵銷。每輪遮罩不同（種子含輪次）。

限制（如實記錄）：
- 這是基本版：有 VD 在遮罩後斷線，遮罩就無法抵銷（完整的 SecAgg+ 用秘密分享處理）。
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


def provision_agreement_keys(key_dir, num_clients: int) -> None:
    """每個 VD 一把 X25519 私鑰（vd-<編號>.x25519），公鑰表 x25519_pub.json 給所有 VD 看。"""
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    pub_path = key_dir / "x25519_pub.json"
    pubs = json.loads(pub_path.read_text(encoding="utf-8")) if pub_path.exists() else {}
    for pid in range(num_clients):
        path = key_dir / f"vd-{pid}.x25519"
        if not path.exists():
            raw = X25519PrivateKey.generate().private_bytes(
                _RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption())
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(raw.hex())
        priv = X25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))
        pubs[str(pid)] = priv.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex()
    pub_path.write_text(json.dumps(pubs), encoding="utf-8")


def load_agreement_keys(key_dir, pid: int) -> tuple[X25519PrivateKey, dict]:
    key_dir = Path(key_dir)
    priv = X25519PrivateKey.from_private_bytes(bytes.fromhex((key_dir / f"vd-{pid}.x25519").read_text().strip()))
    pubs = json.loads((key_dir / "x25519_pub.json").read_text(encoding="utf-8"))
    return priv, {int(k): X25519PublicKey.from_public_bytes(bytes.fromhex(v)) for k, v in pubs.items()}


def pair_seed(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, me: int, other: int, rnd: int) -> bytes:
    shared = my_priv.exchange(their_pub)
    a, b = sorted((me, other))
    return hashlib.sha256(shared + f"secagg|{a}|{b}|{rnd}".encode()).digest()


def _mask(seed: bytes, shape) -> np.ndarray:
    rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint32))
    return rng.uniform(-MASK_RANGE, MASK_RANGE, size=shape)


def mask_update(update: dict, n: int, me: int, pids, my_priv, pubs: dict, rnd: int) -> dict:
    """回傳 n × 更新 ＋ 成對遮罩（float64）。伺服器把所有 VD 的結果加總後除以總樣本數即得加權平均。"""
    out = {k: n * np.asarray(v, dtype=np.float64) for k, v in update.items()}
    for other in pids:
        if other == me:
            continue
        seed = pair_seed(my_priv, pubs[other], me, other, rnd)
        sign = 1.0 if me < other else -1.0
        for i, k in enumerate(sorted(out)):
            out[k] += sign * _mask(seed + i.to_bytes(4, "little"), out[k].shape)
    return out


def unmask_sum(masked: list[dict], total_n: float) -> dict:
    return {k: sum(m[k] for m in masked) / total_n for k in masked[0]}
