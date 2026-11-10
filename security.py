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


def _revoked(key_dir) -> set[int]:
    p = Path(key_dir) / "server" / "revoked.json"
    return {int(x) for x in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else set()  # 讀不到就報錯（fail closed）


def revoke_vd(key_dir, pid: int) -> None:
    """伺服器端撤銷（寫進 server/revoked.json，重開後仍有效）：之後載入的驗證表與公鑰表都不含這台 VD，
    它（或偷到它金鑰的人）送的更新 HMAC／簽章即使正確也會被拒絕。舊金鑰同時刪除；要重新加入需從清單移除，再執行 provision_keys 配發新金鑰。"""
    key_dir, pid = Path(key_dir), int(pid)
    _write_secret(key_dir / "server" / "revoked.json", json.dumps(sorted(_revoked(key_dir) | {pid})))
    # 舊金鑰可能已外洩：一併刪掉，之後就算從清單移除，provision_keys 也只會配發新的
    for table_path, secret in ((key_dir / "server" / "vd_keys.json", True), (key_dir / "vd_sign_pub.json", False)):
        if table_path.exists():
            table = json.loads(table_path.read_text(encoding="utf-8"))
            table.pop(str(pid), None)
            (_write_secret(table_path, json.dumps(table)) if secret
             else table_path.write_text(json.dumps(table), encoding="utf-8"))
    for f in (key_dir / f"vd-{pid}.key", key_dir / f"vd-{pid}.ed25519"):
        f.unlink(missing_ok=True)


def load_vd_key(key_dir, pid: int) -> bytes:
    """VD 端：只讀自己的金鑰。"""
    return bytes.fromhex((Path(key_dir) / f"vd-{pid}.key").read_text().strip())


def load_vd_signing_key(key_dir, pid: int) -> Ed25519PrivateKey:
    """VD 端：自己的 Ed25519 簽章私鑰（只有這台 VD 有）。"""
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex((Path(key_dir) / f"vd-{pid}.ed25519").read_text().strip()))


def load_vd_verify_keys(key_dir) -> dict:
    """伺服器端：各 VD 的簽章公鑰（只能驗證，不能簽）。"""
    table = json.loads((Path(key_dir) / "vd_sign_pub.json").read_text(encoding="utf-8"))
    revoked = _revoked(key_dir)
    return {int(k): Ed25519PublicKey.from_public_bytes(bytes.fromhex(v)) for k, v in table.items() if int(k) not in revoked}


def load_server_public_key(key_dir) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex((Path(key_dir) / "server.pub").read_text().strip()))


def load_server_secrets(key_dir) -> tuple[Ed25519PrivateKey, dict]:
    """伺服器端：簽章私鑰與 VD 金鑰驗證表。"""
    server = Path(key_dir) / "server"
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex((server / "ed25519.key").read_text().strip()))
    table = json.loads((server / "vd_keys.json").read_text(encoding="utf-8"))
    revoked = _revoked(key_dir)
    return priv, {int(k): bytes.fromhex(v) for k, v in table.items() if int(k) not in revoked}


def _payload(arrays: dict, header: str) -> bytes:
    """固定順序的序列化：標頭＋各陣列名稱、形狀、float32 位元組。
    整數陣列（安全聚合的定點數）改涵蓋型別＋原始位元組：轉 float32 會丟掉 uint32 的低位元，
    也分不出 uint16 和 uint32，中間人就能改寬度或低位元而不破壞標籤。"""
    h = hashlib.sha256(header.encode())
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        if a.dtype.kind in "ui":
            a = np.ascontiguousarray(a)
            h.update(b"int:" + a.dtype.str.encode())
        else:
            a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(name.encode())
        h.update(struct.pack(f"<{a.ndim + 1}q", a.ndim, *a.shape))
        h.update(a.tobytes())
    return h.digest()


def update_digest(arrays: dict, rnd: int, pid: int) -> str:
    """更新內容的 SHA-256（稽核紀錄用；不含金鑰）。"""
    return _payload(arrays, f"update|{rnd}|{pid}").hex()


def canonical_metrics(metrics: dict) -> str:
    """VD 回報指標的固定序列化（整數與浮點一律轉 float，傳輸時型別改變也對得上）。
    用 JSON：欄位名稱會加引號跳脫，中間人不能把兩個欄位併成一個而序列化剛好一樣。"""
    def f(v):
        return [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)
    return json.dumps({str(k): f(v) for k, v in metrics.items()}, sort_keys=True, separators=(",", ":"))


def _update_header(rnd: int, pid: int, meta, run: int = 0) -> str:
    # meta 是指標的序列化（含 FedAvg 的加權 num-examples）：不簽進去的話，中間人改它就能讓模型變 NaN、
    # 讓一台 VD 蓋過全部，或改 train-loss／誤差讓紀錄失真
    # run：執行編號，上一次執行同一輪的更新不能重送進這次（金鑰跨執行沿用）
    return f"update|{int(run)}|{rnd}|{pid}" + ("" if meta is None else "|" + str(meta))


def sign_update(key, arrays: dict, rnd: int, pid: int, meta=None, run: int = 0) -> str:
    """key 是 bytes：HMAC-SHA256（完整性、鑑別性）；是 Ed25519 私鑰：數位簽章（再加上不可否認性）。"""
    msg = _payload(arrays, _update_header(rnd, pid, meta, run))
    if isinstance(key, Ed25519PrivateKey):
        return key.sign(msg).hex()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_update(key, arrays: dict, rnd: int, pid: int, tag: str, meta=None, run: int = 0) -> bool:
    """key 是 bytes：比對 HMAC；是 Ed25519 公鑰：驗簽章。"""
    if isinstance(key, Ed25519PublicKey):
        try:
            key.verify(bytes.fromhex(str(tag)), _payload(arrays, _update_header(rnd, pid, meta, run)))
            return True
        except (InvalidSignature, ValueError):
            return False
    return hmac.compare_digest(sign_update(key, arrays, rnd, pid, meta, run), str(tag))


def sign_model(priv: Ed25519PrivateKey, arrays: dict, rnd: int, run: int = 0) -> str:
    return priv.sign(_payload(arrays, f"global|{int(run)}|{rnd}")).hex()


def verify_model(pub: Ed25519PublicKey, arrays: dict, rnd: int, sig: str, run: int = 0) -> bool:
    try:
        pub.verify(bytes.fromhex(sig), _payload(arrays, f"global|{int(run)}|{rnd}"))
        return True
    except (InvalidSignature, ValueError):
        return False


def _recover_header(rnd: int, dropped, run: int) -> bytes:
    return f"secagg-recover|{int(run)}|{int(rnd)}|{sorted({int(j) for j in dropped})}".encode()


def sign_recover(priv: Ed25519PrivateKey, rnd: int, dropped, run: int = 0) -> str:
    """安全聚合斷線復原請求的伺服器簽章：涵蓋執行編號、輪次與斷線名單（別次執行的請求不能重送）。"""
    return priv.sign(_recover_header(rnd, dropped, run)).hex()


def verify_recover(pub: Ed25519PublicKey, rnd: int, dropped, sig: str, run: int = 0) -> bool:
    try:
        pub.verify(bytes.fromhex(sig), _recover_header(rnd, dropped, run))
        return True
    except (InvalidSignature, ValueError):
        return False


def poison_update(global_arrays: dict, local_arrays: dict, kind: str, scale: float,
                  rng: np.random.Generator) -> dict:
    """被入侵 VD 的惡意更新。flip：反向並放大自己的更新量；noise：送出大幅隨機雜訊；
    collude：多台被入侵的 VD 串通，每個參數都往同一個方向偏移 scale 倍自己的更新幅度
    （偏移量和正常更新同一個量級，比較難被中位數當成離群值剔除；rng 要用同一輪共用的種子）。"""
    out = {}
    for k, g in global_arrays.items():
        g = np.asarray(g, dtype=np.float32)
        if kind == "flip":
            out[k] = (g - scale * (np.asarray(local_arrays[k], dtype=np.float32) - g)).astype(np.float32)
        elif kind == "noise":
            out[k] = (g + scale * rng.standard_normal(g.shape)).astype(np.float32)
        elif kind == "hidejam":
            # 本地已用竄改過的標籤訓練（hide_congestion），再把更新放大 scale 倍，蓋過誠實 VD 的平均
            out[k] = (g + scale * (np.asarray(local_arrays[k], dtype=np.float32) - g)).astype(np.float32)
        elif kind == "collude":
            d = np.asarray(local_arrays[k], dtype=np.float32) - g
            direction = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=g.shape)
            out[k] = (g + d + scale * np.abs(d) * direction).astype(np.float32)
        else:
            raise ValueError(f"未知的攻擊類型：{kind}（可用 flip、noise、collude、hidejam）")
    return out


def adaptive_shrink(global_arrays: dict, bad_arrays: dict, score, tol: float,
                    alphas=(1.0, 0.5, 0.25, 0.125)) -> tuple[dict, float]:
    """知道伺服器驗證方式的攻擊者（最壞情況：拿得到參考 VD 和評分函式）：把惡意更新縮成 g + α(bad − g)，
    取分數不超過「全域模型本身分數 ×(1+tol)」的最大 α；都不行就送全域模型（α=0，等於不攻擊）。"""
    g = {k: np.asarray(v, np.float32) for k, v in global_arrays.items()}
    limit = score(g) * (1 + tol)
    for a in alphas:
        cand = {k: (g[k] + a * (np.asarray(bad_arrays[k], np.float32) - g[k])).astype(np.float32) for k in g}
        if score(cand) <= limit:
            return cand, a
    return g, 0.0


def hide_congestion(y: np.ndarray, mean: float, std: float, below_kmh: float = 60.0,
                    shift_kmh: float = 30.0) -> np.ndarray:
    """針對性攻擊的標籤竄改：實際車速低於 below_kmh（塞車）的樣本，標籤改高 shift_kmh，
    讓模型學到「塞車時也預測順暢」。y 是標準化後的車速。"""
    kmh = y * std + mean
    return np.where(kmh < below_kmh, y + shift_kmh / std, y).astype(np.float32)


def tamper(arrays: dict, magnitude: float, rng: np.random.Generator) -> dict:
    """網路中間人竄改：在傳輸中替參數加上偏移（不改動原本的陣列）。
    安全聚合的 uint32 定點數：偏移換算成定點單位（2^12），結果仍是 uint32（mod 2^32）。"""
    # shortcut: 一律用預設 12 位小數換算；secagg-frac-bits 改小時竄改幅度會被放大，要精確比較時再傳入
    if any(np.asarray(v).dtype.kind == "u" for v in arrays.values()):
        from secagg import FRAC_BITS
        return {k: (np.asarray(v)
                    + np.rint(magnitude * 2 ** FRAC_BITS * rng.standard_normal(np.shape(v))).astype(np.int64)
                    .astype(np.asarray(v).dtype)) for k, v in arrays.items()}
    return {k: (np.asarray(v, dtype=np.float32) + magnitude * rng.standard_normal(np.shape(v)))
            .astype(np.float32) for k, v in arrays.items()}


def is_attacker(pid: int, num_clients: int, num_attackers: int, ids: tuple = ()) -> bool:
    """實驗設定：最後 num_attackers 個 VD 被入侵（它們仍持有合法金鑰）；給了 ids 就改成這些編號。"""
    if ids:
        return pid in ids
    return num_attackers > 0 and pid >= num_clients - num_attackers


def drops_out(seed: int, pid: int, rnd: int, rate: float) -> bool:
    """可重現的斷線模擬：同一組（seed, VD, 輪次）每次結果相同。"""
    if rate <= 0:
        return False
    return bool(np.random.default_rng([seed, pid, rnd, 99]).random() < rate)
