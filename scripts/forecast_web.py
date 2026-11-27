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

PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>國道車速預測</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--ok:#1f7a4d;--slow:#a86400;--jam:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9c9b94;--card:#201f1d;--line:#34332f;--ok:#5cc58f;--slow:#e3a44a;--jam:#f08074}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans TC","Microsoft JhengHei",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px;font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-size:12px;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}.ok{color:var(--ok)}.slow{color:var(--slow)}.jam{color:var(--jam);font-weight:600}
.na{color:var(--muted)}.t{display:block;font-size:11px;color:var(--muted)}.err{color:var(--jam)}
.legend{font-size:12px;color:var(--muted);margin:12px 0 0}
.wrap{overflow-x:auto}
@media (max-width:520px){th,td{padding:7px 6px;font-size:14px}h1{font-size:19px}}
</style></head><body><main>
<h1>國道 1 號北向：未來 15 分鐘車速預測</h1>
<p class="sub" id="status">載入中…</p>
<div class="wrap"><table><thead><tr><th scope="col">位置</th><th scope="col">最近</th>
<th scope="col">+5 分</th><th scope="col">+10 分</th><th scope="col">+15 分</th></tr></thead>
<tbody id="rows"></tbody></table></div>
<p class="legend">單位 km/h，每格是那個時刻開始的 5 分鐘平均車速。<span class="ok">80 以上順暢</span>、<span class="slow">40～80 車多</span>、<span class="jam">40 以下壅塞</span>。
模型以聯邦學習＋差分隱私訓練（各 VD 的原始資料不離開本地）；塞車時預測的回升通常比實際快，僅供參考。</p>
</main><script>
const cls = v => v == null ? "na" : v >= 80 ? "ok" : v >= 40 ? "slow" : "jam";
function place(id){const m=/^VD-N(\\d+)-([NS])-([\\d.]+)-/.exec(id);return m?`${m[2]=="N"?"北":"南"} ${(+m[3]).toFixed(1)}K`:id}
function cell(v,t){const td=document.createElement("td");td.className=cls(v);td.textContent=v==null?"–":v.toFixed(1);
 if(t){const s=document.createElement("span");s.className="t";s.textContent=t;td.appendChild(s)}return td}
async function load(){
 const st=document.getElementById("status");
 try{const r=await fetch("api/forecast",{cache:"no-store"});const d=await r.json();
  if(d.error){st.textContent="更新失敗："+d.error;st.className="sub err";if(!d.vds)return}
  else{st.className="sub";st.textContent=`預測時間點 ${d.end}（台灣時間），抓到 ${d.minutes} 分鐘資料；每 5 分鐘自動更新`}
  const tb=document.getElementById("rows");tb.replaceChildren();
  d.vds.sort((a,b)=>parseFloat(a.vdid.split("-")[3])-parseFloat(b.vdid.split("-")[3]));
  for(const v of d.vds){const tr=document.createElement("tr");const th=document.createElement("td");th.textContent=place(v.vdid);th.title=v.vdid;tr.appendChild(th);
   tr.appendChild(cell(v.last_kmh,v.last_time));
   for(let i=0;i<3;i++){const p=v.preds[i];tr.appendChild(p?cell(p.kmh,p.time):cell(null,i==0?"資料不足":""))}
   tb.appendChild(tr)}
 }catch(e){st.textContent="連不到伺服器";st.className="sub err"}}
load();setInterval(load,60000);
</script></body></html>"""


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


def make_handler(fc: Forecaster):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server 的命名
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
    ThreadingHTTPServer((a.host, a.port), make_handler(fc)).serve_forever()


if __name__ == "__main__":
    main()
