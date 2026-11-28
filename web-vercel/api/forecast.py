"""Vercel 函式：GET /api/forecast → 32 台 VD 的即時預測（JSON）。

CDN 快取 120 秒，多人同時打開也只會實際抓一次高公局資料。
"""
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_lib"))
import forecast_core as fc  # noqa: E402

MODEL, CFG, STATS, BUNDLE = fc.load_bundle(HERE / "_lib" / "model_bundle.json")
VDS = sorted(STATS)


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel 規定的名稱
    def do_GET(self):  # noqa: N802
        try:
            body, cache = fc.predict_all(MODEL, CFG, STATS, VDS, fc.taipei_now()), "public, s-maxage=120, stale-while-revalidate=300"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            body, cache = {"error": f"{type(e).__name__}，請稍後再試"}, "no-store"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)
