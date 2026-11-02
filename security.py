"""通訊安全與攻擊模擬（對應課程講義「物聯網安全」的 CIA 三要素）。

- 完整性／鑑別性（上傳）：每個 VD 用自己的金鑰對模型更新算 HMAC-SHA256（講義的 MAC = H(K||M)；
  實作用 HMAC，避免 H(K||M) 的長度延伸攻擊）。訊息內含輪次與 VD 編號，舊輪次的更新重送（replay）會被拒絕。
- 完整性（下發）：伺服器用 Ed25519 數位簽章簽全域模型，VD 驗證通過才訓練，
  對應講義「更新未簽章、被騙安裝假更新」的案例（一對多下發用公鑰簽章，VD 不需要伺服器的私鑰）。
- 攻擊模擬：被入侵的 VD（有合法金鑰，MAC 擋不住）送出惡意更新；網路上的中間人竄改更新（MAC 擋得住）。

金鑰：模擬時由 run config 的 auth-secret 推導；實際部署應在 VD 安裝時個別配發（例如 PKI／TPM），
不可共用同一把主金鑰。
"""
from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def vd_key(secret: str, pid: int) -> bytes:
    """VD 專屬金鑰（由主密鑰推導；各 VD 不同，拿到一台 VD 的金鑰也偽造不了別台的更新）。"""
    return hmac.new(secret.encode(), f"vd-key|{pid}".encode(), hashlib.sha256).digest()


def _payload(arrays: dict, header: str) -> bytes:
    """固定順序的序列化：標頭＋各陣列名稱、形狀、float32 位元組。"""
    h = hashlib.sha256(header.encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float32))
        h.update(name.encode())
        h.update(struct.pack(f"<{a.ndim + 1}q", a.ndim, *a.shape))
        h.update(a.tobytes())
    return h.digest()


def sign_update(key: bytes, arrays: dict, rnd: int, pid: int) -> str:
    return hmac.new(key, _payload(arrays, f"update|{rnd}|{pid}"), hashlib.sha256).hexdigest()


def verify_update(key: bytes, arrays: dict, rnd: int, pid: int, tag: str) -> bool:
    return hmac.compare_digest(sign_update(key, arrays, rnd, pid), str(tag))


def server_private_key(secret: str) -> Ed25519PrivateKey:
    seed = hashlib.sha256(f"server-signing|{secret}".encode()).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def server_public_key(secret: str) -> Ed25519PublicKey:
    return server_private_key(secret).public_key()


def sign_model(priv: Ed25519PrivateKey, arrays: dict, rnd: int) -> str:
    return priv.sign(_payload(arrays, f"global|{rnd}")).hex()


def verify_model(pub: Ed25519PublicKey, arrays: dict, rnd: int, sig: str) -> bool:
    try:
        pub.verify(bytes.fromhex(sig), _payload(arrays, f"global|{rnd}"))
        return True
    except (InvalidSignature, ValueError):
        return False


def poison_update(global_arrays: dict, local_arrays: dict, kind: str, scale: float,
                  rng: np.random.Generator) -> dict:
    """被入侵 VD 的惡意更新。flip：反向並放大自己的更新量；noise：送出大幅隨機雜訊。"""
    out = {}
    for k, g in global_arrays.items():
        g = np.asarray(g, dtype=np.float32)
        if kind == "flip":
            out[k] = (g - scale * (np.asarray(local_arrays[k], dtype=np.float32) - g)).astype(np.float32)
        elif kind == "noise":
            out[k] = (g + scale * rng.standard_normal(g.shape)).astype(np.float32)
        else:
            raise ValueError(f"未知的攻擊類型：{kind}（可用 flip、noise）")
    return out


def tamper(arrays: dict, magnitude: float, rng: np.random.Generator) -> dict:
    """網路中間人竄改：在傳輸中替參數加上偏移（不改動原本的陣列）。"""
    return {k: (np.asarray(v, dtype=np.float32) + magnitude * rng.standard_normal(np.shape(v)))
            .astype(np.float32) for k, v in arrays.items()}


def is_attacker(pid: int, num_clients: int, num_attackers: int) -> bool:
    """實驗設定：最後 num_attackers 個 VD 被入侵（它們仍持有合法金鑰）。"""
    return num_attackers > 0 and pid >= num_clients - num_attackers


def drops_out(seed: int, pid: int, rnd: int, rate: float) -> bool:
    """可重現的斷線模擬：同一組（seed, VD, 輪次）每次結果相同。"""
    if rate <= 0:
        return False
    return bool(np.random.default_rng([seed, pid, rnd, 99]).random() < rate)
