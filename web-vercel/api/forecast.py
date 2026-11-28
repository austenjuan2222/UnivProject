"""Vercel 函式：GET /api/forecast → 轉抓桌機版的即時預測（JSON 原樣回傳）。

高公局開放資料擋海外連線，Vercel 抓不到，所以改由桌機計算：
桌機的 Cloudflare 快速通道網址寫在 GitHub 上的 tunnel.json（scripts/publish_tunnel_url.py 維護），
這裡先讀網址、再轉抓桌機的 /api/forecast。CDN 快取 60 秒，桌機實際收到的請求很少。
"""
import json
import re
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler

TUNNEL_JSON = "https://raw.githubusercontent.com/austenjuan2222/UnivProject/claude/project-thread-flkrik/web-vercel/tunnel.json"
ALLOWED = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
MAX_BYTES = 4 * 2**20
OFFLINE = {"error": "桌機資料來源離線，請稍後再試"}


def fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "univproject-forecast-relay"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # 非 2xx 會丟 HTTPError
        return r.read(MAX_BYTES + 1)


def tunnel_url(raw: bytes) -> str:
    url = json.loads(raw).get("url")
    if not isinstance(url, str) or not ALLOWED.fullmatch(url):
        raise ValueError("tunnel.json 的網址不是 https://*.trycloudflare.com")
    return url


def relay(fetch=fetch) -> bytes:
    base = tunnel_url(fetch(TUNNEL_JSON, 10.0))
    body = fetch(base + "/api/forecast", 15.0)
    if len(body) > MAX_BYTES or not isinstance(json.loads(body), dict):
        raise ValueError("桌機回傳的不是預測 JSON")
    return body


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel 規定的名稱
    def do_GET(self):  # noqa: N802
        try:
            data, cache = relay(), "public, s-maxage=60, stale-while-revalidate=120"
        except Exception:  # noqa: BLE001 - 任何失敗都當桌機離線
            traceback.print_exc()
            data, cache = json.dumps(OFFLINE, ensure_ascii=False).encode("utf-8"), "no-store"
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)
