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
import math
import os
import secrets
import struct
import time
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_RAW = serialization.Encoding.Raw


def _write_secret(path: Path, data: str) -> None:
    """只有擁有者可讀寫（0600）；先建檔再寫入，避免短暫以預設權限存在。"""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, 0o600)  # 0o600 只在新建時套用；覆寫原本權限較寬的檔案時也要收緊
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)


def _write_atomic(path: Path, data: str, secret: bool = True) -> None:
    """先寫暫存檔再 os.replace：中途當掉時舊表還在，不會留下寫一半、讓伺服器起不來的 JSON。"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")  # 每次不同：多個行程同時寫也不會互相蓋到暫存檔
    if secret:
        _write_secret(tmp, data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


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
    if not (key_dir / "server.pub").exists():  # VD 安裝時釘住的公鑰；輪替後靠 server_chain.json 跟上，不在這裡改寫
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
    issued_path = server / "key_issued.json"
    issued = json.loads(issued_path.read_text(encoding="utf-8")) if issued_path.exists() else {}
    now = time.time()
    for pid in range(num_clients):
        if str(pid) not in table:
            table[str(pid)] = secrets.token_hex(32)
            issued[str(pid)] = now
        vd_key = key_dir / f"vd-{pid}.key"
        if str(pid) not in issued:  # 舊金鑰沒有紀錄：用金鑰檔的寫入時間，不能當成剛簽發（不然每次重開都延期）
            issued[str(pid)] = vd_key.stat().st_mtime if vd_key.exists() else 0
        if not vd_key.exists():
            _write_secret(vd_key, table[str(pid)])
    _write_secret(table_path, json.dumps(table))
    _write_atomic(issued_path, json.dumps(issued))


def _fresh(key_dir, max_age_days: float, now) -> callable:
    """金鑰有效期限（像 TLS 憑證的到期日）：回傳判斷某台 VD 金鑰還有效的函式；max_age_days=0 不檢查。
    沒有紀錄、不是數字、或簽發時間在未來超過一天的，一律當過期（fail closed）。"""
    if not max_age_days:
        return lambda pid: True
    p = Path(key_dir) / "server" / "key_issued.json"
    issued = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    now = time.time() if now is None else now

    def ok(pid) -> bool:
        t = issued.get(str(pid))
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            return False
        try:
            t = float(t)
        except OverflowError:
            return False
        if not math.isfinite(t):
            return False
        return now - max_age_days * 86400 <= t <= now + 86400
    return ok


def key_expiry(key_dir, max_age_days: float) -> dict:
    """給伺服器每輪檢查用：{VD 編號: 到期時間}。已經過期的也列（第一輪就移除），不然啟動瞬間剛好跨過期限的會載入了卻永遠不過期；
    紀錄壞掉的不列（載入時已經被 _fresh 擋掉，不在驗證表裡）。"""
    if not max_age_days:
        return {}
    p = Path(key_dir) / "server" / "key_issued.json"
    issued = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    out = {}
    for k, t in issued.items():
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            continue
        try:
            t = float(t)
        except OverflowError:
            continue
        if math.isfinite(t):
            out[int(k)] = t + max_age_days * 86400
    return out


def _revoked(key_dir) -> set[int]:
    p = Path(key_dir) / "server" / "revoked.json"
    return {int(x) for x in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else set()  # 讀不到就報錯（fail closed）


def revoke_vd(key_dir, pid: int) -> None:
    """伺服器端撤銷（寫進 server/revoked.json，重開後仍有效）：之後載入的驗證表與公鑰表都不含這台 VD，
    它（或偷到它金鑰的人）送的更新 HMAC／簽章即使正確也會被拒絕。舊金鑰同時刪除；要重新加入需從清單移除，再執行 provision_keys 配發新金鑰。"""
    key_dir, pid = Path(key_dir), int(pid)
    _write_secret(key_dir / "server" / "revoked.json", json.dumps(sorted(_revoked(key_dir) | {pid})))
    # 舊金鑰可能已外洩：一併刪掉，之後就算從清單移除，provision_keys 也只會配發新的
    for table_path, secret in ((key_dir / "server" / "vd_keys.json", True), (key_dir / "vd_sign_pub.json", False),
                               (key_dir / "server" / "key_issued.json", True)):
        if table_path.exists():
            table = json.loads(table_path.read_text(encoding="utf-8"))
            table.pop(str(pid), None)
            (_write_secret(table_path, json.dumps(table)) if secret
             else table_path.write_text(json.dumps(table), encoding="utf-8"))
    for f in (key_dir / f"vd-{pid}.key", key_dir / f"vd-{pid}.ed25519"):
        f.unlink(missing_ok=True)


def rotate_vd(key_dir, pid: int, now=None) -> None:
    """定期換新一台 VD 的 HMAC 金鑰和 Ed25519 簽章金鑰（不撤銷）：舊金鑰就算外洩，下次伺服器啟動後也無效。
    先改伺服器的表、再寫 VD 的檔：中途失敗時兩邊不一致只會讓這台 VD 被拒絕（fail closed），重跑即可。"""
    key_dir, pid = Path(key_dir), int(pid)
    if pid in _revoked(key_dir):
        raise ValueError(f"VD {pid} 已撤銷；要重新加入請先從撤銷清單移除再配發")
    pubs_path = key_dir / "vd_sign_pub.json"
    pubs = json.loads(pubs_path.read_text(encoding="utf-8"))
    table_path = key_dir / "server" / "vd_keys.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if pid < 0 or str(pid) not in table or str(pid) not in pubs:
        raise ValueError(f"VD {pid} 沒有配發過金鑰，不能輪替（避免打錯編號多出一個有效身分）")
    sign = Ed25519PrivateKey.generate()
    pubs[str(pid)] = sign.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex()
    _write_atomic(pubs_path, json.dumps(pubs), secret=False)
    # 安全聚合的 X25519 金鑰放在同一個資料夾，資料夾外洩時一起外洩，所以一併換新
    agree_path = key_dir / "x25519_pub.json"
    if agree_path.exists():
        from secagg import _new_x25519  # 延遲載入：secagg 也會用到這個模組
        agree = json.loads(agree_path.read_text(encoding="utf-8"))
        (key_dir / f"vd-{pid}.x25519").unlink(missing_ok=True)
        x = _new_x25519(key_dir / f"vd-{pid}.x25519")
        agree[str(pid)] = x.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex()
        _write_atomic(agree_path, json.dumps(agree), secret=False)
    table[str(pid)] = secrets.token_hex(32)
    _write_atomic(table_path, json.dumps(table))
    issued_path = key_dir / "server" / "key_issued.json"
    issued = json.loads(issued_path.read_text(encoding="utf-8")) if issued_path.exists() else {}
    issued[str(pid)] = time.time() if now is None else now
    _write_atomic(issued_path, json.dumps(issued))
    _write_secret(key_dir / f"vd-{pid}.ed25519",
                  sign.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex())
    _write_secret(key_dir / f"vd-{pid}.key", table[str(pid)])


def load_vd_key(key_dir, pid: int) -> bytes:
    """VD 端：只讀自己的金鑰。"""
    return bytes.fromhex((Path(key_dir) / f"vd-{pid}.key").read_text().strip())


def load_vd_signing_key(key_dir, pid: int) -> Ed25519PrivateKey:
    """VD 端：自己的 Ed25519 簽章私鑰（只有這台 VD 有）。"""
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex((Path(key_dir) / f"vd-{pid}.ed25519").read_text().strip()))


def load_vd_verify_keys(key_dir, max_age_days: float = 0, now=None) -> dict:
    """伺服器端：各 VD 的簽章公鑰（只能驗證，不能簽）。"""
    table = json.loads((Path(key_dir) / "vd_sign_pub.json").read_text(encoding="utf-8"))
    revoked, fresh = _revoked(key_dir), _fresh(key_dir, max_age_days, now)
    return {int(k): Ed25519PublicKey.from_public_bytes(bytes.fromhex(v)) for k, v in table.items()
            if int(k) not in revoked and fresh(k)}


def _rollover_msg(new_pub: bytes) -> bytes:
    return b"its-fl server rollover v1\x00" + new_pub


def rotate_server(key_dir) -> None:
    """換新伺服器的 Ed25519 簽章金鑰：用舊私鑰簽新公鑰，接在 server_chain.json 後面（像憑證鏈），
    VD 不必重新安裝 server.pub，從自己釘住的公鑰一路驗到最新的。
    順序：新私鑰先存成 ed25519.key.next → 寫鏈 → 換上新私鑰。中途當掉時重跑會接續（鏈最後一把就是 .next）或重來（.next 還沒公告）。"""
    key_dir = Path(key_dir)
    priv_path = key_dir / "server" / "ed25519.key"
    next_path = priv_path.with_name("ed25519.key.next")
    chain_path = key_dir / "server_chain.json"
    chain = json.loads(chain_path.read_text(encoding="utf-8")) if chain_path.exists() else []
    if next_path.exists():
        pending = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(next_path.read_text().strip()))
        if chain and chain[-1].get("pub") == pending.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw).hex():
            os.replace(next_path, priv_path)  # 上次已公告但沒換上：接續完成
            return
        next_path.unlink()  # 上次還沒公告：丟掉重來
    old = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_path.read_text().strip()))
    new = Ed25519PrivateKey.generate()
    new_pub = new.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw)
    _write_secret(next_path, new.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex())
    chain.append({"pub": new_pub.hex(), "sig": old.sign(_rollover_msg(new_pub)).hex()})
    _write_atomic(chain_path, json.dumps(chain), secret=False)
    os.replace(next_path, priv_path)


def load_server_public_key(key_dir) -> Ed25519PublicKey:
    """VD 端：從釘住的 server.pub 沿著 server_chain.json 驗到最新的伺服器公鑰；任何一段驗不過就報錯（fail closed）。
    跟上之後把最新公鑰寫回 server.pub，之後偷到舊私鑰的人就不能從舊金鑰另外接一條鏈。"""
    key_dir = Path(key_dir)
    pin_path = key_dir / "server.pub"
    for attempt in range(20):
        try:
            pin = bytes.fromhex(pin_path.read_text().strip())
            break
        except PermissionError:  # Windows：別台 VD 正在 os.replace 這個檔（模擬時共用資料夾），稍等再讀
            if attempt == 19:
                raise
            time.sleep(0.05)
    chain_path = key_dir / "server_chain.json"
    chain = json.loads(chain_path.read_text(encoding="utf-8")) if chain_path.exists() else []
    try:
        entries = [(bytes.fromhex(e["pub"]), bytes.fromhex(e["sig"])) for e in chain]
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError(f"server_chain.json 格式錯誤：{e}") from None
    start = next((i + 1 for i, (pub, _) in enumerate(entries) if pub == pin), 0)
    current = pin
    for pub, sig in entries[start:]:
        try:
            Ed25519PublicKey.from_public_bytes(current).verify(sig, _rollover_msg(pub))
        except (InvalidSignature, ValueError):
            raise ValueError("伺服器金鑰鏈驗不過：可能被竄改，拒絕使用") from None
        current = pub
    if current != pin:  # 一定要存下來才用：不然 VD 一直釘在舊公鑰，偷到舊私鑰的人可以另接一條鏈
        for attempt in range(20):
            try:
                _write_atomic(pin_path, current.hex(), secret=False)
                break
            except PermissionError:  # Windows：模擬時別台 VD 正在換同一個檔，稍等再試
                if attempt == 19:
                    raise
                time.sleep(0.05)
    return Ed25519PublicKey.from_public_bytes(current)


def load_server_secrets(key_dir, max_age_days: float = 0, now=None) -> tuple[Ed25519PrivateKey, dict]:
    """伺服器端：簽章私鑰與 VD 金鑰驗證表。"""
    server = Path(key_dir) / "server"
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex((server / "ed25519.key").read_text().strip()))
    table = json.loads((server / "vd_keys.json").read_text(encoding="utf-8"))
    revoked, fresh = _revoked(key_dir), _fresh(key_dir, max_age_days, now)
    return priv, {int(k): bytes.fromhex(v) for k, v in table.items() if int(k) not in revoked and fresh(k)}


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
