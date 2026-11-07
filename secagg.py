"""安全聚合（Bonawitz et al. 2017 的成對遮罩，基本版）：伺服器只看得到加總，看不到單一 VD 的更新。

對應講義「機密性」：即使伺服器（或偷看伺服器的人）不可信，也拿不到個別 VD 的模型更新。
做法：每對 VD (i, j) 用 X25519 金鑰交換得到只有兩人知道的種子，各自產生同一份遮罩 r_ij；
i < j 時 i 加上 r_ij、j 減去 r_ij，加總時全部抵銷。每輪遮罩不同（種子含輪次）。

限制（如實記錄）：
- 斷線復原：在線 VD 交出和斷線 VD 之間的成對種子，伺服器扣掉這些遮罩（reveal_seeds、drop_correction）；種子加密給伺服器（seal_seeds），中間人讀不到。
  這是簡化版：沒有雙重遮罩與秘密分享，若斷線 VD 的更新事後又送到，伺服器就看得到它（完整的 SecAgg+ 解決這點）。
- 伺服器看不到個別更新，所以不能和「伺服器端裁剪的 client-level DP」或中位數聚合一起用；
  可以和 record-level DP（在 VD 本地加噪）一起用。
- 在有限體 Z_{2^32} 上做整數遮罩（定點數，小數 12 位元），抵銷完全精確，上傳和 float32 一樣大。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

_RAW = serialization.Encoding.Raw
# 有限體 Z_{2^32} 上的定點數：小數 12 位元。遮罩是 [0, 2^32) 的均勻整數，加總後完全抵銷（沒有浮點誤差），
# 上傳和 float32 一樣大。每台 VD 把 n×值限制在 ±2^31 / 2^12 / VD 數以內，加總才不會溢位。
FRAC_BITS = 12


def _limit(num_vd: int) -> int:
    return (2 ** 31 - 1) // max(int(num_vd), 1)


def _encode(x: np.ndarray, num_vd: int) -> np.ndarray:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float64) * 2 ** FRAC_BITS), -_limit(num_vd), _limit(num_vd))
    return q.astype(np.int64).astype(np.uint32)  # 負數取 2 的補數（mod 2^32）


def count_clipped(update: dict, n: int, num_vd: int) -> int:
    """這份更新有多少個值超出定點範圍、會被截到上限（VD 自己回報，伺服器看不到個別更新）。"""
    lim = _limit(num_vd)
    return int(sum(np.sum(np.abs(np.rint(n * np.asarray(v, dtype=np.float64) * 2 ** FRAC_BITS)) > lim)
                   for v in update.values()))


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
    os.chmod(key_dir / "server", 0o700)
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
    server = key_dir / "x25519_server.pub"
    if server.exists():  # 只有斷線復原要用；VD 訓練不因少這個檔案而失敗
        out["server"] = X25519PublicKey.from_public_bytes(bytes.fromhex(server.read_text(encoding="utf-8").strip()))
    return out


def load_server_agreement_key(key_dir) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(
        bytes.fromhex((Path(key_dir) / "server" / "agree.x25519").read_text().strip()))


def _seal_aead(my_priv: X25519PrivateKey, their_pub: X25519PublicKey) -> ChaCha20Poly1305:
    shared = my_priv.exchange(their_pub)
    return ChaCha20Poly1305(HKDF(hashes.SHA256(), 32, None, b"secagg-seal").derive(shared))


def _seal_aad(me: int, j: int, rnd: int, run: int) -> bytes:
    return f"secagg-seal|{int(run)}|{int(rnd)}|{int(me)}|{int(j)}".encode()


def seal_seeds(seeds: dict, me: int, my_priv: X25519PrivateKey, server_pub: X25519PublicKey, rnd: int,
               run: int = 0) -> dict[int, str]:
    """VD 把復原種子加密給伺服器（X25519 共享金鑰 → ChaCha20-Poly1305）：沒有 TLS 時中間人讀不到，
    改過也會被發現（不然改一個位元就會默默算錯遮罩、毀掉全域模型）。執行、輪次、這對 VD 放在 AAD。"""
    aead = _seal_aead(my_priv, server_pub)
    # 每把金鑰對每個 (執行, 輪次, VD 對) 只加密同一份明文；用 AAD 綁定，nonce 取 AAD 的雜湊前 12 bytes
    return {int(j): aead.encrypt(_nonce(me, j, rnd, run), bytes.fromhex(x), _seal_aad(me, j, rnd, run)).hex()
            for j, x in seeds.items()}


def _nonce(me, j, rnd, run) -> bytes:
    return hashlib.sha256(_seal_aad(me, j, rnd, run)).digest()[:12]


def open_seeds(sealed: dict, me: int, server_priv: X25519PrivateKey, vd_pub: X25519PublicKey, rnd: int,
               run: int = 0) -> dict[int, str] | None:
    """伺服器端解開；任何一個密文格式錯或驗不過就回傳 None（整輪復原放棄）。"""
    aead, out = _seal_aead(server_priv, vd_pub), {}
    for j, x in sealed.items():
        try:
            seed = aead.decrypt(_nonce(me, j, rnd, run), bytes.fromhex(str(x)), _seal_aad(me, j, rnd, run))
        except (InvalidTag, ValueError, TypeError):
            return None
        if len(seed) != 32:
            return None
        out[int(j)] = seed.hex()
    return out


def pair_seed(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, me: int, other: int, rnd: int,
              run: int = 0) -> bytes:
    # 綁執行編號：金鑰跨次執行保留，不綁的話上一次執行交出的種子能解開這次同一輪的遮罩
    shared = my_priv.exchange(their_pub)
    a, b = sorted((me, other))
    return hashlib.sha256(shared + f"secagg|{int(run)}|{a}|{b}|{rnd}".encode()).digest()


def _mask(seed: bytes, shape) -> np.ndarray:
    # SHAKE-256 當密碼學 PRG（numpy 的 PCG64 不是），每個值是均勻的 uint32
    size = int(np.prod(shape))
    return np.frombuffer(hashlib.shake_256(seed).digest(4 * size), "<u4").reshape(shape).astype(np.uint32)


def self_seed(my_priv: X25519PrivateKey, me: int, rnd: int, run: int = 0) -> bytes:
    """雙重遮罩的自我遮罩種子：只有這台 VD 算得出來（從自己的私鑰導出），每輪、每次執行不同。"""
    raw = my_priv.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return HKDF(hashes.SHA256(), 32, None, f"secagg-self|{int(run)}|{int(me)}|{int(rnd)}".encode()).derive(raw)


def mask_update(update: dict, n: int, me: int, pids, my_priv, pubs: dict, rnd: int, run: int = 0,
                self_mask: bool = False) -> dict:
    """回傳 n × 更新（定點數，mod 2^32 的 uint32）＋ 成對遮罩。伺服器把所有 VD 的結果加總後除以總樣本數即得加權平均。
    self_mask：再加一層只有自己知道的遮罩（雙重遮罩），在線時才交出它的種子；伺服器宣布某台斷線、
    拿到所有和它配對的種子後，就算它的更新遲到送達也解不開。"""
    pids = list(pids)
    out = {k: _encode(n * np.asarray(v, dtype=np.float64), len(pids)) for k, v in update.items()}
    if self_mask:
        seed = self_seed(my_priv, me, rnd, run)
        for i, k in enumerate(sorted(out)):
            out[k] += _mask(seed + i.to_bytes(4, "little"), out[k].shape)
    for other in pids:
        if other == me:
            continue
        seed = pair_seed(my_priv, pubs[other], me, other, rnd, run)
        for i, k in enumerate(sorted(out)):
            m = _mask(seed + i.to_bytes(4, "little"), out[k].shape)
            out[k] = out[k] + m if me < other else out[k] - m  # uint32 運算自動 mod 2^32
    return out


def reveal_seeds(me: int, dropped, my_priv, pubs: dict, rnd: int, run: int = 0) -> dict[int, str]:
    """斷線復原：還在線的 VD 交出自己和每個斷線 VD 之間的成對種子（只有這幾對，伺服器還是解不開在線 VD 的遮罩）。"""
    return {int(j): pair_seed(my_priv, pubs[int(j)], me, int(j), rnd, run).hex() for j in dropped if int(j) != me}


def drop_correction(revealed: dict, like: dict) -> dict:
    """revealed[i][j] = VD i 交出的 (i, j) 種子；回傳在線 VD 為了斷線 VD 加上、卻沒被抵銷的遮罩總和。
    revealed[i][i] 是 VD i 自己的自我遮罩種子（雙重遮罩），一律是加上去的。"""
    out = {k: np.zeros(np.shape(v), dtype=np.uint32) for k, v in like.items()}
    for i, seeds in revealed.items():
        for j, seed in seeds.items():
            for idx, k in enumerate(sorted(out)):
                m = _mask(bytes.fromhex(seed) + idx.to_bytes(4, "little"), out[k].shape)
                out[k] = out[k] + m if int(i) <= int(j) else out[k] - m
    return out


def unmask_sum(masked: list[dict], total_n: float, correction: dict | None = None) -> dict:
    """加總（mod 2^32）、扣掉沒抵銷的遮罩，再解回浮點數並除以總樣本數。"""
    out = {}
    for k in masked[0]:
        acc = np.zeros(np.shape(masked[0][k]), dtype=np.uint32)
        for m in masked:
            if np.asarray(m[k]).dtype != np.uint32:  # 浮點數硬轉 uint32 會默默截斷（甚至未定義）
                raise ValueError(f"安全聚合只收 uint32，{k} 是 {np.asarray(m[k]).dtype}")
            acc = acc + np.asarray(m[k])
        if correction:
            acc = acc - correction[k]
        out[k] = acc.view(np.int32).astype(np.float64) / 2 ** FRAC_BITS / total_n
    return out