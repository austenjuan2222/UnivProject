"""把桌機 Cloudflare 快速通道的網址發佈到 GitHub（web-vercel/tunnel.json），給 Vercel 雲端版轉抓。

每 interval 秒讀一次 cloudflared 的 log；網址和 tunnel.json 不同時才 commit、push（只有通道重開才會發生）。
用法：python scripts/publish_tunnel_url.py --log C:/Users/TESTER/tools/cloudflared/tunnel.log [--once]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = "web-vercel/tunnel.json"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def latest_url(log_text: str) -> str | None:
    found = URL_RE.findall(log_text)
    return found[-1] if found else None


def current(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("url")
    except (OSError, ValueError):
        return None


def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True, timeout=120)


def publish_once(log: Path) -> str:
    url = latest_url(log.read_text(encoding="utf-8", errors="replace"))
    path = REPO / TARGET
    if not url:
        return "log 裡沒有通道網址"
    if url == current(path):
        return "未變更"
    git("pull", "--rebase", "--autostash")  # 先跟上遠端，其他人推過也不會 push 失敗
    if url == current(path):
        return "未變更"
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps({"url": url, "updated_at": now}, ensure_ascii=False) + "\n", encoding="utf-8")
    git("add", TARGET)
    git("commit", "-m", "通道網址更新（publish_tunnel_url.py）", "--", TARGET)
    git("push")
    return f"已發佈 {url}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    while True:
        try:
            msg = publish_once(a.log)
        except Exception as e:  # noqa: BLE001 - 網路或 git 失敗就下一輪再試
            msg = f"失敗：{type(e).__name__} {getattr(e, 'stderr', '') or e}".strip()
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
