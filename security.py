"""通訊安全與攻擊模擬（對應課程講義「物聯網安全」的 CIA 三要素）。

- 完整性／鑑別性（上傳）：每個 VD 用自己的金鑰對模型更新算 HMAC-SHA256（講義的 MAC = H(K||M)；
  實作用 HMAC，避免 H(K||M) 的長度延伸攻擊）。訊息內含輪次與 VD 編號，舊輪次的更新重送（replay）會被拒絕。
- 完整性（下發）：伺服器用 Ed25519 數位簽章簽全域模型，VD 驗證通過才訓練，
  對應講義「更新未簽章、被騙安裝假更新」的案例（一對多下發用公鑰簽章，VD 不需要伺服器的私鑰）。
- 攻擊模擬：被入侵的 VD（有合法金鑰，MAC 擋不住）送出惡意更新；網路上的中間人竄改更新（MAC 擋得住）。

金鑰配發（provision_keys）：每個 VD 一把隨機 32 位元組金鑰（vd-<編號>.key），加上伺服器公鑰 server.pub；
伺服器的 Ed25519 私鑰與「VD 編號 → 金鑰」驗證表放在 server/ 子資料夾，VD 端程式只讀自己的檔案。
沒有共用的主密鑰，所以一台 VD 被入侵，也算不出別台 VD 的金鑰或伺服器的簽章私鑰。
模擬時所有檔案在同一台電腦；實際部署時 vd-*.key 在 VD 安裝時寫入該 VD（例如 TPM），server/ 只留在伺服器。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_RAW = serialization.Encoding.Raw


def _write_secret(path: Path, data: str) -> None:
    """只有擁有者可讀寫（0600）；先建檔再寫入，避免短暫以預設權限存在。"""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)


def provision_keys(key_dir, num_clients: int) -> None:
    """產生（或補齊）金鑰；已存在的金鑰不會被換掉。"""
    key_dir = Path(key_dir)
    server = key_dir / "server"
    server.mkdir(parents=True, exist_ok=True)
    os.chmod(server, 0o700)
    priv_path = server / "ed25519.key"
    if not priv_path.exists():
        _write_secret(priv_path, Ed25519PrivateKey.generate().private_bytes(
            _RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex())
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_path.read_text().strip()))
    (key_dir / "server.pub").write_text(priv.public_key().public_bytes(
        _RAW, serialization.PublicFormat.Raw).hex(), encoding="utf-8")
    # 不可否認性用：每個 VD 一把 Ed25519 簽章私鑰，公鑰表 vd_sign_pub.json 公開
    pub_table_path = key_dir / "vd_sign_pub.json"
    sign_pubs = json.loads(pub_table_path.read_text(encoding="utf-8")) if pub_table_path.exists() else {}
    for pid in range(num_clients):
        p = key_dir / f"vd-{pid}.ed25519"
        if not p.exists():
            _write_secret(p, Ed25519PrivateKey.generate().private_bytes(
                _RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex())
        k = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(p.read_text().strip()))
        sign_pubs[str(pid)] = k.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex()
    pub_table_path.write_text(json.dumps(sign_pubs), encoding="utf-8")
    table_path = server / "vd_keys.json"
    table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else {}
    for pid in range(num_clients):
        if str(pid) not in table:
            table[str(pid)] = secrets.token_hex(32)
        if not (key_dir / f"vd-{pid}.key").exists():
            _write_secret(key_dir / f"vd-{pid}.key", table[str(pid)])
    _write_secret(table_path, json.dumps(table))


def load_vd_key(key_dir, pid: int) -> bytes:
    """VD 端：只讀自己的金鑰。"""
    return bytes.fromhex((Path(key_dir) / f"vd-{pid}.key").read_text().strip())


def load_vd_signing_key(key_dir, pid: int) -> Ed25519PrivateKey:
    """VD 端：自己的 Ed25519 簽章私鑰（只有這台 VD 有）。"""
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex((Path(key_dir) / f"vd-{pid}.ed25519").read_text().strip()))


def load_vd_verify_keys(key_dir) -> dict:
    """伺服器端：各 VD 的簽章公鑰（只能驗證，不能簽）。"""
    table = json.loads((Path(key_dir) / "vd_sign_pub.json").read_text(encoding="utf-8"))
    return {int(k): Ed25519PublicKey.from_public_bytes(bytes.fromhex(v)) for k, v in table.items()}


def load_server_public_key(key_dir) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex((Path(key_dir) / "server.pub").read_text().strip()))


def load_server_secrets(key_dir) -> tuple[Ed25519PrivateKey, dict]:
    """伺服器端：簽章私鑰與 VD 金鑰驗證表。"""
    server = Path(key_dir) / "server"
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex((server / "ed25519.key").read_text().strip()))
    table = json.loads((server / "vd_keys.json").read_text(encoding="utf-8"))
    return priv, {int(k): bytes.fromhex(v) for k, v in table.items()}


def _payload(arrays: dict, header: str) -> bytes:
    """固定順序的序列化：標頭＋各陣列名稱、形狀、float32 位元組。"""
    h = hashlib.sha256(header.encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float32))
        h.update(name.encode())
        h.update(struct.pack(f"<{a.ndim + 1}q", a.ndim, *a.shape))
        h.update(a.tobytes())
    return h.digest()


def update_digest(arrays: dict, rnd: int, pid: int) -> str:
    """更新內容的 SHA-256（稽核紀錄用；不含金鑰）。"""
    return _payload(arrays, f"update|{rnd}|{pid}").hex()


def sign_update(key, arrays: dict, rnd: int, pid: int) -> str:
    """key 是 bytes：HMAC-SHA256（完整性、鑑別性）；是 Ed25519 私鑰：數位簽章（再加上不可否認性）。"""
    msg = _payload(arrays, f"update|{rnd}|{pid}")
    if isinstance(key, Ed25519PrivateKey):
        return key.sign(msg).hex()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_update(key, arrays: dict, rnd: int, pid: int, tag: str) -> bool:
    """key 是 bytes：比對 HMAC；是 Ed25519 公鑰：驗簽章。"""
    if isinstance(key, Ed25519PublicKey):
        try:
            key.verify(bytes.fromhex(str(tag)), _payload(arrays, f"update|{rnd}|{pid}"))
            return True
        except (InvalidSignature, ValueError):
            return False
    return hmac.compare_digest(sign_update(key, arrays, rnd, pid), str(tag))


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
