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
# LPWAN 可改用 Z_{2^16}（bits=16）：上傳再減半，但範圍只有 ±2^15 / VD 數，小數位數要跟著調低（可以是負的）。
FRAC_BITS = 12
_RING = {32: (np.uint32, np.int32), 16: (np.uint16, np.int16)}


def _limit(num_vd: int, bits: int = 32) -> int:
    return (2 ** (bits - 1) - 1) // max(int(num_vd), 1)


def _encode(x: np.ndarray, num_vd: int, bits: int = 32, frac_bits: int = FRAC_BITS) -> np.ndarray:
    lim = _limit(num_vd, bits)
    q = np.clip(np.rint(np.asarray(x, dtype=np.float64) * 2.0 ** frac_bits), -lim, lim)
    return q.astype(np.int64).astype(_RING[bits][0])  # 負數取 2 的補數（mod 2^bits）


def count_clipped(update: dict, n: int, num_vd: int, bits: int = 32, frac_bits: int = FRAC_BITS) -> int:
    """這份更新有多少個值超出定點範圍、會被截到上限（VD 自己回報，伺服器看不到個別更新）。"""
    lim = _limit(num_vd, bits)
    return int(sum(np.sum(np.abs(np.rint(n * np.asarray(v, dtype=np.float64) * 2.0 ** frac_bits)) > lim)
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


def _seal_aad(me: int, j: int, rnd: int, run: int, kind: str = "seal") -> bytes:
    return f"secagg-{kind}|{int(run)}|{int(rnd)}|{int(me)}|{int(j)}".encode()


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


def _mask(seed: bytes, shape, dtype=np.uint32) -> np.ndarray:
    # SHAKE-256 當密碼學 PRG（numpy 的 PCG64 不是），每個值是均勻的 uint32（或 uint16）
    dt = np.dtype(dtype).newbyteorder("<")
    size = int(np.prod(shape))
    return np.frombuffer(hashlib.shake_256(seed).digest(dt.itemsize * size), dt).reshape(shape).astype(dtype)


def self_seed(my_priv: X25519PrivateKey, me: int, rnd: int, run: int = 0) -> bytes:
    """雙重遮罩的自我遮罩種子：只有這台 VD 算得出來（從自己的私鑰導出），每輪、每次執行不同。"""
    raw = my_priv.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return HKDF(hashes.SHA256(), 32, None, f"secagg-self|{int(run)}|{int(me)}|{int(rnd)}".encode()).derive(raw)


def mask_update(update: dict, n: int, me: int, pids, my_priv, pubs: dict, rnd: int, run: int = 0,
                self_mask: bool = False, bits: int = 32, frac_bits: int = FRAC_BITS,
                pair_priv: X25519PrivateKey | None = None, pair_pubs: dict | None = None,
                num_vd: int | None = None) -> dict:
    """回傳 n × 更新（定點數，mod 2^32 的 uint32）＋ 成對遮罩。伺服器把所有 VD 的結果加總後除以總樣本數即得加權平均。
    self_mask：再加一層只有自己知道的遮罩（雙重遮罩），在線時才交出它的種子；伺服器宣布某台斷線、
    拿到所有和它配對的種子後，就算它的更新遲到送達也解不開。"""
    pids = list(pids)
    # 定點數範圍照「加總的 VD 數」算（只和鄰居遮罩時 pids 比總數少）
    nv = int(num_vd) if num_vd is not None else len(pids)
    out = {k: _encode(n * np.asarray(v, dtype=np.float64), nv, bits, frac_bits) for k, v in update.items()}
    dt = _RING[bits][0]
    if self_mask:
        seed = self_seed(my_priv, me, rnd, run)
        for i, k in enumerate(sorted(out)):
            out[k] += _mask(seed + i.to_bytes(4, "little"), out[k].shape, dt)
    for other in pids:
        if other == me:
            continue
        # 自我遮罩一律用長期私鑰導出；成對遮罩可改用這輪的臨時金鑰（pair_priv／pair_pubs）
        seed = (pair_seed(pair_priv, pair_pubs[other], me, other, rnd, run) if pair_priv is not None
                else pair_seed(my_priv, pubs[other], me, other, rnd, run))
        for i, k in enumerate(sorted(out)):
            m = _mask(seed + i.to_bytes(4, "little"), out[k].shape, dt)
            out[k] = out[k] + m if me < other else out[k] - m  # 無號整數運算自動 mod 2^bits
    return out


def reveal_seeds(me: int, dropped, my_priv, pubs: dict, rnd: int, run: int = 0) -> dict[int, str]:
    """斷線復原：還在線的 VD 交出自己和每個斷線 VD 之間的成對種子（只有這幾對，伺服器還是解不開在線 VD 的遮罩）。"""
    return {int(j): pair_seed(my_priv, pubs[int(j)], me, int(j), rnd, run).hex() for j in dropped if int(j) != me}


def drop_correction(revealed: dict, like: dict) -> dict:
    """revealed[i][j] = VD i 交出的 (i, j) 種子；回傳在線 VD 為了斷線 VD 加上、卻沒被抵銷的遮罩總和。
    revealed[i][i] 是 VD i 自己的自我遮罩種子（雙重遮罩），一律是加上去的。"""
    out = {k: np.zeros(np.shape(v), dtype=np.asarray(v).dtype) for k, v in like.items()}
    for i, seeds in revealed.items():
        for j, seed in seeds.items():
            for idx, k in enumerate(sorted(out)):
                m = _mask(bytes.fromhex(seed) + idx.to_bytes(4, "little"), out[k].shape, out[k].dtype)
                out[k] = out[k] + m if int(i) <= int(j) else out[k] - m
    return out


def unmask_sum(masked: list[dict], total_n: float, correction: dict | None = None,
               frac_bits: int = FRAC_BITS, bits: int | None = None) -> dict:
    """加總（mod 2^bits）、扣掉沒抵銷的遮罩，再解回浮點數並除以總樣本數。
    bits：伺服器設定的環寬度；給了就只收那個寬度（不能讓上傳自己決定）。"""
    ring = {np.dtype(u): s for u, s in _RING.values()}
    for m in masked:  # 少了、多了陣列或形狀不同（會被廣播）一律拒收
        if set(m) != set(masked[0]) or any(np.shape(m[k]) != np.shape(masked[0][k]) for k in m):
            raise ValueError("安全聚合的上傳陣列名稱或形狀不一致")
    out = {}
    for k in masked[0]:
        dt = np.asarray(masked[0][k]).dtype
        if dt not in ring or (bits is not None and dt != np.dtype(_RING[bits][0])):
            raise ValueError(f"安全聚合只收 {'uint%d' % bits if bits else 'uint32／uint16'}，{k} 是 {dt}")
        acc = np.zeros(np.shape(masked[0][k]), dtype=dt)
        for m in masked:
            if np.asarray(m[k]).dtype != dt:  # 浮點數或不同寬度硬轉會默默截斷
                raise ValueError(f"安全聚合的 {k} 寬度不一致：{dt} 和 {np.asarray(m[k]).dtype}")
            acc = acc + np.asarray(m[k])
        if correction:
            acc = acc - correction[k]
        out[k] = acc.view(ring[dt]).astype(np.float64) / 2.0 ** frac_bits / total_n
    return out


# ---- Shamir 秘密分享自我遮罩種子（SecAgg 的做法）----
# VD 上傳時把自我遮罩種子切成 n 份、各用和持有者之間的金鑰加密，經伺服器轉交。復原時 VD 不直接交出自己的種子，
# 改由其他 VD 交出「名單上在線」的那些 VD 的份。每台 VD 對同一台只會交出成對種子（說它斷線）或份（說它在線）其中一種，
# 惡意伺服器對不同 VD 送不同名單也拿不到同一台的兩種種子；在線卻在復原時沉默的 VD，伺服器湊滿門檻份數就能還原。
SHAMIR_P = 2 ** 521 - 1  # 梅森質數，比 256 位元的種子大
_SHARE_LEN = 66


def shamir_threshold(n: int) -> int:
    """門檻：過半數。少於這麼多份的 VD（和伺服器）合謀，還原不出任何一台的自我遮罩種子。"""
    return int(n) // 2 + 1


def neighbors(i: int, n: int, k: int = 0) -> list[int]:
    """SecAgg+ 的通訊圖：環上前後各 k/2 台（k=0 表示所有其他 VD）。對稱，成對遮罩才抵銷得掉。"""
    if not k:
        return [j for j in range(n) if j != i]
    if k % 2 or not 2 <= k < n:
        raise ValueError("secagg-neighbors 要是偶數，且 2 ≤ k < VD 數")
    # shortcut: 固定的環狀圖，攻擊者事先知道誰是誰的鄰居；SecAgg+ 用每輪隨機圖讓攻擊者無法挑鄰居，要更強時再換
    h = k // 2
    return sorted({(i + d) % n for d in range(1, h + 1)} | {(i - d) % n for d in range(1, h + 1)})


def share_holders(i: int, n: int, k: int = 0) -> list[int]:
    """誰持有 VD i 的份：k=0 是所有 VD（含自己），否則是它的 k 個鄰居。"""
    return list(range(n)) if not k else neighbors(i, n, k)


def share_threshold(n: int, k: int = 0) -> int:
    return shamir_threshold(n) if not k else k // 2 + 1


def shamir_split(secret: bytes, ids, t: int) -> dict[int, int]:
    # 係數由種子導出（HKDF）：同一顆種子重算得到同樣的份，加密時就不會用同一個 nonce 加密不同明文
    stream = HKDF(hashes.SHA512(), 66 * max(int(t) - 1, 1), None, b"secagg-shamir-coeffs").derive(secret)
    coeffs = [int.from_bytes(secret, "big")] + [int.from_bytes(stream[66 * i:66 * (i + 1)], "big") % SHAMIR_P
                                                for i in range(int(t) - 1)]
    out = {}
    for j in ids:
        x, y = int(j) + 1, 0
        for c in reversed(coeffs):
            y = (y * x + c) % SHAMIR_P
        out[int(j)] = y
    return out


def seed_tag(seed: bytes) -> int:
    """種子的承諾（SHA-256 前 6 bytes：指標會經過 float，2^53 以內才不失真）：伺服器用它檢查還原出的種子對不對（被交錯的份就對不上）。"""
    return int.from_bytes(hashlib.sha256(b"secagg-self-tag" + seed).digest()[:6], "big")


def shamir_combine(shares: dict) -> bytes | None:
    """Lagrange 內插求 x=0 的值；份數不足或份被改過時幾乎一定得到 ≥ 2^256 的數，回傳 None。"""
    if not shares:
        return None
    xs = {int(j): int(j) + 1 for j in shares}
    s = 0
    for j, y in shares.items():
        num = den = 1
        for k, xk in xs.items():
            if k != int(j):
                num = num * (-xk) % SHAMIR_P
                den = den * (xs[int(j)] - xk) % SHAMIR_P
        s = (s + int(y) * num * pow(den, -1, SHAMIR_P)) % SHAMIR_P
    return s.to_bytes(32, "big") if s < 2 ** 256 else None


def consistency_threshold(n: int) -> int:
    """名單一致性回合要湊滿的相同證明數（含自己）。"""
    return 2 * int(n) // 3 + 1


def ephemeral_key(my_priv: X25519PrivateKey, me: int, rnd: int, run: int = 0) -> X25519PrivateKey:
    """每輪的臨時金鑰（成對遮罩用）：從長期私鑰導出，只有自己算得出；還原它只會洩漏這一輪的成對種子。"""
    raw = my_priv.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return X25519PrivateKey.from_private_bytes(
        HKDF(hashes.SHA256(), 32, None, f"secagg-eph|{int(run)}|{int(me)}|{int(rnd)}".encode()).derive(raw))


def eph_bytes(key: X25519PrivateKey) -> bytes:
    return key.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption())


def _pair_hmac(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, info: bytes, msg: str) -> str:
    import hmac
    key = HKDF(hashes.SHA256(), 32, None, info).derive(my_priv.exchange(their_pub))
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


def epub_mac(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, sender: int, receiver: int, rnd: int, run: int,
             epub: str) -> str:
    """sender 向 receiver 證明「這是我這輪的臨時公鑰」：伺服器轉交時換成自己的公鑰會被發現。"""
    return _pair_hmac(my_priv, their_pub, b"secagg-epub",
                      f"secagg-epub|{int(run)}|{int(rnd)}|{int(sender)}|{int(receiver)}|{epub}")


def consist_mac(my_priv: X25519PrivateKey, their_pub: X25519PublicKey, sender: int, receiver: int, rnd: int,
                run: int, dropped) -> str:
    """sender 向 receiver 證明「我收到的斷線名單是 dropped」：用兩兩之間的金鑰，伺服器轉交時改不了也偽造不了。"""
    return _pair_hmac(my_priv, their_pub, b"secagg-consist",
                      f"secagg-consist|{int(run)}|{int(rnd)}|{int(sender)}|{int(receiver)}|{sorted(int(j) for j in dropped)}")


def _share_aead(my_priv: X25519PrivateKey, their_pub: X25519PublicKey) -> ChaCha20Poly1305:
    return ChaCha20Poly1305(HKDF(hashes.SHA256(), 32, None, b"secagg-share").derive(my_priv.exchange(their_pub)))


def _share_aad(owner: int, holder: int, rnd: int, run: int, kind: str = "share") -> bytes:
    # kind 區分自我遮罩種子的份（share）和臨時私鑰的份（eshare）：伺服器不能拿其中一種冒充另一種
    return f"secagg-{kind}|{int(run)}|{int(rnd)}|{int(owner)}|{int(holder)}".encode()


def encrypt_shares(secret: bytes, me: int, pids, my_priv: X25519PrivateKey, pubs: dict, rnd: int, run: int = 0,
                   t: int | None = None, kind: str = "share") -> dict[int, str]:
    """切份並各自加密給持有者（只有持有者解得開；AAD 綁執行、輪次、擁有者、持有者）。"""
    pids = [int(j) for j in pids]
    shares = shamir_split(secret, pids, t or shamir_threshold(len(pids)))
    out = {}
    for j, y in shares.items():
        aad = _share_aad(me, j, rnd, run, kind)
        out[j] = _share_aead(my_priv, pubs[j]).encrypt(hashlib.sha256(aad).digest()[:12],
                                                       y.to_bytes(_SHARE_LEN, "big"), aad).hex()
    return out


def open_share(ct: str, owner: int, holder: int, holder_priv: X25519PrivateKey, pubs: dict, rnd: int,
               run: int = 0, kind: str = "share") -> int | None:
    aad = _share_aad(owner, holder, rnd, run, kind)
    try:
        y = _share_aead(holder_priv, pubs[int(owner)]).decrypt(hashlib.sha256(aad).digest()[:12],
                                                               bytes.fromhex(str(ct)), aad)
    except (InvalidTag, ValueError, TypeError, KeyError):
        return None
    return int.from_bytes(y, "big") if len(y) == _SHARE_LEN else None


def seal_shares(shares: dict, me: int, my_priv: X25519PrivateKey, server_pub: X25519PublicKey, rnd: int,
                run: int = 0, kind: str = "share") -> dict[int, str]:
    """VD 把解開的份再加密給伺服器（和種子用同一把金鑰，AAD 標明是份）。"""
    aead = _seal_aead(my_priv, server_pub)
    out = {}
    for i, y in shares.items():
        aad = _seal_aad(me, i, rnd, run, kind)
        out[int(i)] = aead.encrypt(hashlib.sha256(aad).digest()[:12], int(y).to_bytes(_SHARE_LEN, "big"), aad).hex()
    return out


def open_shares(sealed: dict, me: int, server_priv: X25519PrivateKey, vd_pub: X25519PublicKey, rnd: int,
                run: int = 0, kind: str = "share") -> dict[int, int] | None:
    aead, out = _seal_aead(server_priv, vd_pub), {}
    for i, x in sealed.items():
        aad = _seal_aad(me, i, rnd, run, kind)
        try:
            y = aead.decrypt(hashlib.sha256(aad).digest()[:12], bytes.fromhex(str(x)), aad)
        except (InvalidTag, ValueError, TypeError):
            return None
        if len(y) != _SHARE_LEN:
            return None
        out[int(i)] = int.from_bytes(y, "big")
    return out
