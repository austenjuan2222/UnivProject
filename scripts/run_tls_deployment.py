#!/usr/bin/env python3
"""用真的 TLS 連線＋VD 裝置認證跑一次聯邦訓練（對應課程講義 TLS／RFID 認證）。

模擬模式（flwr run . local）所有 VD 都在同一個行程裡，沒有網路，也就沒有 TLS。
這支腳本改用 Flower 的部署模式：一個 SuperLink（伺服器）＋ N 個 SuperNode（各代表一個 VD），
彼此走 gRPC over TLS，並開啟 SuperNode 認證，只有事先註冊過公鑰的 VD 能連上。最後再試兩種冒充：

  1. 沒註冊的假 VD（講義 RFID 的偽造標籤）：拿自己產生的金鑰連線，應被拒絕
  2. 不用 TLS 的明文連線：應連不上

  python scripts/run_tls_deployment.py --data-dir data/processed_28d_32 --num-vds 8 --rounds 3

憑證與金鑰寫在 keys/tls/（不進 git），每次重新產生。
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[1]
FLEET, CONTROL = 9092, 9093


def _write_key(key, path: Path, fmt=serialization.PrivateFormat.PKCS8) -> None:
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, fmt, serialization.NoEncryption()))


def make_tls(out: Path, days: int = 2) -> dict[str, Path]:
    """自建 CA，簽一張給 127.0.0.1／localhost 的伺服器憑證（ECDSA P-256）。"""
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)

    def cert(subject, issuer, pub, signer, ca: bool, san=None):
        b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(pub)
             .serial_number(x509.random_serial_number())
             .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
             .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
        if san:
            b = b.add_extension(x509.SubjectAlternativeName(san), critical=False)
        return b.sign(signer, hashes.SHA256())

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ITS-FL demo CA")])
    ca_cert = cert(ca_name, ca_name, ca_key.public_key(), ca_key, True)
    srv_key = ec.generate_private_key(ec.SECP256R1())
    srv_cert = cert(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]), ca_name,
                    srv_key.public_key(), ca_key, False,
                    [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))])
    paths = {"ca": out / "ca.crt", "cert": out / "server.pem", "key": out / "server.key"}
    paths["ca"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    paths["cert"].write_bytes(srv_cert.public_bytes(serialization.Encoding.PEM))
    _write_key(srv_key, paths["key"])
    return paths


def make_node_key(out: Path, name: str) -> tuple[Path, Path]:
    """SuperNode 認證用的 P-384 金鑰對（Flower 要求 NIST EC 曲線）。"""
    key = ec.generate_private_key(ec.SECP384R1())
    priv, pub = out / f"{name}.key", out / f"{name}.pub"
    _write_key(key, priv, serialization.PrivateFormat.OpenSSH)  # flower-supernode 只讀 OpenSSH 格式
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                                  serialization.PublicFormat.OpenSSH))
    return priv, pub


def wait_port(port: int, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise SystemExit(f"等不到 port {port}")


def node_cmd(exe: Path, tls: dict, key: Path | None, pid: int, n: int, port: int, secure=True) -> list[str]:
    cmd = [str(exe / "flower-supernode"), "--superlink", f"127.0.0.1:{FLEET}",
           "--clientappio-api-address", f"127.0.0.1:{port}", "--max-retries", "2", "--max-wait-time", "20",
           "--node-config", f"partition-id={pid} num-partitions={n}"]
    cmd += ["--root-certificates", str(tls["ca"])] if secure else ["--insecure"]
    if key:
        cmd += ["--auth-supernode-private-key", str(key)]
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--num-vds", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "tls"))
    a = p.parse_args(argv)

    exe = Path(sys.executable).parent
    out = Path(a.out) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    tls_dir = ROOT / "keys" / "tls"
    tls = make_tls(tls_dir)
    home = out / "flwr_home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[superlink]\ndefault = "tls"\n\n[superlink.tls]\n'
        f'address = "127.0.0.1:{CONTROL}"\nroot-certificates = "{tls["ca"].as_posix()}"\n', encoding="utf-8")
    env = dict(os.environ, FLWR_HOME=str(home), PYTHONUTF8="1",
               PATH=str(exe) + os.pathsep + os.environ.get("PATH", ""))
    procs, logs = [], []

    def start(cmd, name):
        f = (out / f"{name}.log").open("w", encoding="utf-8")
        logs.append(f)
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
        return procs[-1]

    try:
        start([str(exe / "flower-superlink"), "--ssl-ca-certfile", str(tls["ca"]),
               "--ssl-certfile", str(tls["cert"]), "--ssl-keyfile", str(tls["key"]),
               # 不給 --database：狀態放記憶體；Flower 1.26 在 Windows 上把檔案路徑轉 sqlite URL 會出錯
               "--enable-supernode-auth"], "superlink")
        wait_port(FLEET)
        wait_port(CONTROL)
        flwr = str(exe / "flwr")
        for i in range(a.num_vds):
            priv, pub = make_node_key(tls_dir, f"vd-{i}")
            reg = subprocess.run([flwr, "supernode", "register", str(pub), "tls"], env=env, cwd=ROOT,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            if reg.returncode:
                raise SystemExit(f"註冊 vd-{i} 失敗：{reg.stdout}{reg.stderr}")
            start(node_cmd(exe, tls, priv, i, a.num_vds, 9200 + i), f"vd-{i}")

        cfg = (f"num-clients={a.num_vds} num-server-rounds={a.rounds} "
               f"data-dir='{Path(a.data_dir).resolve().as_posix()}' results-dir='{out.as_posix()}' "
               "run-name='tls_auth'")
        t0 = time.time()
        r = subprocess.run([flwr, "run", str(ROOT), "tls", "--run-config", cfg, "--stream"],
                           env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (out / "run.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        metrics = out / "tls_auth" / "metrics.csv"
        ok = metrics.exists() and len(metrics.read_text(encoding="utf-8").splitlines()) == a.rounds + 1
        print(f"訓練{'完成' if ok else '失敗'}（{time.time() - t0:.0f} 秒），紀錄：{out / 'run.log'}")
        if not ok:
            return 1

        # 冒充測試：沒註冊的金鑰、不用 TLS
        fake_priv, _ = make_node_key(tls_dir, "fake-vd")
        results = {}
        for name, cmd in {
            "fake_unregistered": node_cmd(exe, tls, fake_priv, 0, a.num_vds, 9300),
            "plaintext_insecure": node_cmd(exe, tls, None, 0, a.num_vds, 9301, secure=False),
        }.items():
            pr = start(cmd, name)
            try:
                pr.wait(timeout=90)
            except subprocess.TimeoutExpired:
                pr.kill()
            text = (out / f"{name}.log").read_text(encoding="utf-8", errors="replace")
            results[name] = (pr.returncode, text.strip().splitlines()[-3:])
            print(f"{name}: exit={pr.returncode}  最後幾行：{results[name][1]}")
        return r.returncode
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pr.kill()
        for f in logs:
            f.close()


if __name__ == "__main__":
    raise SystemExit(main())
