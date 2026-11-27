#!/usr/bin/env python3
"""即時車速預測網頁：每 5 分鐘抓一次 VD 資料、預測每台未來 15 分鐘，瀏覽器打開就能看。

  python scripts/forecast_web.py --run-dir <訓練結果>/dp_sigma1.1_seed1 --data-dir ../UnivProject-run/data/processed_28d_32
  然後打開 http://127.0.0.1:8000

預設只聽本機（127.0.0.1）；要給同一個網路的人看才加 --host 0.0.0.0（沒有登入機制，只提供公開路況資料的預測）。
只用標準函式庫的 http.server，適合展示與小規模使用，不適合直接放到網際網路上。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_live as fl  # noqa: E402

# 網頁本身在 forecast_page.html（只用 textContent／createElement 放資料，不用 innerHTML）
PAGE = (Path(__file__).resolve().parent / "forecast_page.html").read_text(encoding="utf-8")


class Forecaster:
    """背景執行緒每 interval 秒更新一次；網頁只讀最新結果，不會每個請求都去抓資料。"""

    def __init__(self, run_dir: Path, data_dir: Path, interval: int = 300):
        self.model, self.cfg = fl.load_model(run_dir)
        self.stats = fl.vd_stats(data_dir, self.cfg["train-frac"])
        self.interval, self.lock = interval, threading.Lock()
        self.latest: dict = {"error": "第一次預測還在進行（約 1 分鐘）"}

    def update(self) -> None:
        try:
            r = fl.predict_all(self.model, self.cfg, self.stats, sorted(self.stats), fl.taipei_now())
        except Exception as e:  # noqa: BLE001 - 網路或資料問題時保留上一次的結果
            traceback.print_exc()
            with self.lock:
                self.latest = {**self.latest, "error": f"{type(e).__name__}，顯示上一次的結果"}
            return
        with self.lock:
            self.latest = r

    def loop(self) -> None:
        while True:
            t0 = time.monotonic()
            self.update()
            time.sleep(max(10.0, self.interval - (time.monotonic() - t0)))

    def json(self) -> bytes:
        with self.lock:
            return json.dumps(self.latest, ensure_ascii=False).encode("utf-8")


class RateLimit:
    """每個來源每 window 秒最多 limit 個請求；公開網址被狂刷時回 429，不拖慢其他人。"""

    def __init__(self, limit: int = 120, window: float = 60.0):
        self.limit, self.window, self.lock, self.hits = limit, window, threading.Lock(), {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            if len(self.hits) > 10000:  # 來源太多時丟掉過期的，記憶體不會無限長
                self.hits = {k: v for k, v in self.hits.items() if now - v[0] < self.window}
            start, n = self.hits.get(key, (now, 0))
            if now - start >= self.window:
                start, n = now, 0
            self.hits[key] = (start, n + 1)
            return n < self.limit


def make_handler(fc: Forecaster, limiter: RateLimit | None = None, timeout: float = 10.0):
    conn_timeout = timeout
    limiter = limiter or RateLimit()

    class Handler(BaseHTTPRequestHandler):
        timeout = conn_timeout  # 標頭送一半就停的連線，逾時後關掉，不會一直佔著執行緒
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _client(self) -> str:
            # 經過 Cloudflare 通道時連線都來自本機，真正的來源在 CF-Connecting-IP；直接連進來的不信任這個標頭
            ip = self.client_address[0]
            return self.headers.get("CF-Connecting-IP", ip) if ip in ("127.0.0.1", "::1") else ip

        def do_GET(self):  # noqa: N802 - http.server 的命名
            if not limiter.allow(self._client()):
                self._send(429, "請求太頻繁，請稍後再試".encode("utf-8"), "text/plain; charset=utf-8")
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/forecast":
                self._send(200, fc.json(), "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, *_):  # 不把每個請求印出來
            pass
    return Handler


class BoundedServer(ThreadingHTTPServer):
    """同時最多 max_conn 個連線；滿了就直接關掉新連線，不再開執行緒（限流在 do_GET 裡才算，擋不住慢速連線）。"""
    daemon_threads = True

    def __init__(self, addr, handler, max_conn: int = 64):
        self.slots = threading.BoundedSemaphore(max_conn)
        super().__init__(addr, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def make_server(host: str, port: int, fc: Forecaster, timeout: float = 10.0, max_conn: int = 64) -> BoundedServer:
    return BoundedServer((host, port), make_handler(fc, timeout=timeout), max_conn)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--interval", type=int, default=300, help="幾秒更新一次（預設 5 分鐘，配合資料的時間單位）")
    a = p.parse_args(argv)
    fc = Forecaster(a.run_dir, a.data_dir, a.interval)
    threading.Thread(target=fc.loop, daemon=True).start()
    print(f"打開 http://{a.host}:{a.port}", flush=True)
    make_server(a.host, a.port, fc).serve_forever()


if __name__ == "__main__":
    main()
