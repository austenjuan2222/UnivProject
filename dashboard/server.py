#!/usr/bin/env python3
"""UnivProject 即時儀表板伺服器（只用 Python 標準函式庫，不呼叫任何 LLM、不花 token）。

  python dashboard/server.py                              # 監看 results/live/events.jsonl + Claude Code 紀錄
  python dashboard/server.py --replay results/<批次>/dp_sigma1.1   # 重播跑完的實驗
  python dashboard/server.py --replay ... --speed 4       # 4 倍速重播

然後開瀏覽器到 http://127.0.0.1:8765 。另開終端機跑：
  python scripts/run_experiments.py --dashboard

資料流：訓練行程寫事件檔 → 這支程式 tail → Server-Sent Events 推給瀏覽器。
Claude Code 的 token／工具呼叫來自 ~/.claude/projects/<本專案>/*.jsonl（本機檔案，零 token）。
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dashboard_events import parse_claude_line, read_events, replay_schedule  # noqa: E402

HISTORY_MAX = 50_000


class Hub:
    """保存事件歷史（新開的分頁會先收到目前這次實驗的所有事件），並廣播給所有連線。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.fl_history: list[dict] = []   # 從最近一次 run_start 起的訓練事件
        self.cc_history: list[dict] = []   # Claude Code 事件
        self.subs: list[queue.Queue] = []
        self.last_live = 0.0               # 最近一次從即時事件檔讀到新事件的時間

    def publish(self, ev: dict) -> None:
        with self.lock:
            if ev.get("kind") == "run_start":
                self.fl_history = []
            hist = self.cc_history if str(ev.get("kind", "")).startswith("cc") else self.fl_history
            hist.append(ev)
            if len(hist) > HISTORY_MAX:
                del hist[: len(hist) - HISTORY_MAX]
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass  # 瀏覽器跟不上就丟，下次重新整理會拿到完整歷史

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue(maxsize=20_000)
        with self.lock:
            self.subs.append(q)
            snapshot = self.cc_history + self.fl_history
        return q, snapshot

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


def tail_events(hub: Hub, path: Path, from_start: bool, poll: float = 0.15) -> None:
    """持續讀事件檔。檔案被清空或換掉（變小）時從頭開始。"""
    offset = 0
    if not from_start and path.exists():
        # 只回放最後一次 run_start 之後的內容，避免載入很久以前的實驗
        evs, end = read_events(path, 0)
        last = max((i for i, e in enumerate(evs) if e.get("kind") == "run_start"), default=None)
        for e in evs[last:] if last is not None else []:
            hub.publish(e)
        offset = end
    while True:
        try:
            if path.exists() and path.stat().st_size < offset:
                offset = 0
        except OSError:
            pass
        evs, offset = read_events(path, offset)
        if evs:
            hub.last_live = time.time()
        for e in evs:
            hub.publish(e)
        time.sleep(poll)


def latest_run_events(results: Path) -> Path | None:
    """results/ 底下最新一份 run 的 events.jsonl（不含 live/ 那份即時檔）。"""
    runs = [p for p in results.glob("**/events.jsonl") if p.parent.name != "live" and p.stat().st_size]
    return max(runs, key=lambda p: p.stat().st_mtime, default=None)


def idle_replay(hub: Hub, results: Path, idle: float, speed: float) -> None:
    """沒有實驗在跑時（idle 秒沒有新事件）循環重播最近一次 run，畫面不會停住；新實驗一開始就讓位。"""
    def busy():
        return time.time() - hub.last_live < idle
    while True:
        time.sleep(1)
        src = None if busy() else latest_run_events(results)
        if src is None:
            continue
        evs, _ = read_events(src, 0)
        for e, d in zip(evs, replay_schedule(evs, speed=speed, max_gap=1.0)):
            time.sleep(d)
            if busy():
                break
            hub.publish(dict(e, t=time.time(), replay=True))
        time.sleep(2)


def replay(hub: Hub, path: Path, speed: float, loop: bool) -> None:
    evs, _ = read_events(path, 0)
    if not evs:
        print(f"⚠ {path} 沒有事件可重播")
        return
    while True:
        for e, d in zip(evs, replay_schedule(evs, speed=speed, max_gap=1.5)):
            time.sleep(d)
            hub.publish(dict(e, t=time.time()))
        if not loop:
            return
        time.sleep(3)


def claude_project_dirs(project_root: Path, base: Path | None = None) -> list[Path]:
    """找出這個專案在 ~/.claude/projects/ 下的紀錄資料夾。

    Claude Code 把專案路徑中非英數字元換成 '-' 當資料夾名，例如
    C:\\Users\\TESTER\\univproject → C--Users-TESTER-univproject。
    同一專案常分成 git 工作副本、worktree、venv 資料夾（UnivProject-git／-run／-dashboard），
    所以也收「名稱第一段相同」的兄弟資料夾。
    """
    base = base or Path.home() / ".claude" / "projects"
    if not base.is_dir():
        return []
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project_root)).lower()
    stem = re.sub(r"[^A-Za-z0-9]", "-", project_root.name.split("-")[0]).lower()
    sib = re.compile(rf"-{re.escape(stem)}(-|$)")
    return sorted(d for d in base.iterdir()
                  if d.is_dir() and (d.name.lower() == slug or sib.search(d.name.lower())))


def tail_claude(hub: Hub, dirs: list[Path], since_hours: float, poll: float = 1.0) -> None:
    """讀 Claude Code 紀錄：啟動時載入最近 since_hours 小時有更新的檔案，之後持續追新內容。"""
    offsets: dict[Path, int] = {}
    seen: set = set()
    cutoff = time.time() - since_hours * 3600
    first = True
    while True:
        files = [f for d in dirs for f in d.glob("**/*.jsonl")]
        for f in files:
            try:
                st = f.stat()
            except OSError:
                continue
            if f not in offsets:
                if first and st.st_mtime < cutoff:
                    offsets[f] = st.st_size  # 太舊的對話不載入
                    continue
                offsets[f] = 0
            if st.st_size < offsets[f]:
                offsets[f] = 0
            if st.st_size == offsets[f]:
                continue
            with f.open("rb") as fh:
                fh.seek(offsets[f])
                chunk = fh.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                continue
            offsets[f] += end + 1
            for raw in chunk[:end].split(b"\n"):
                ev = parse_claude_line(raw.decode("utf-8", "replace"), seen)
                if ev and (ev.get("kind") != "cc" or ev["in"] or ev["out"] or ev["cache_read"]
                           or ev["cache_write"] or ev["tools"]):
                    ev["session"] = f.stem[:8]
                    ev["t"] = time.time()
                    hub.publish(ev)
        first = False
        time.sleep(poll)


def make_handler(hub: Hub, prices: dict):
    index = HERE / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 安靜一點
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, index.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/prices.json":
                self._send(200, json.dumps(prices).encode(), "application/json")
            elif self.path == "/events":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q, snapshot = hub.subscribe()
            try:
                self.wfile.write(b"event: snapshot\ndata: " + json.dumps(snapshot).encode() + b"\n\n")
                self.wfile.flush()
                while True:
                    batch = []
                    try:
                        batch.append(q.get(timeout=10))
                        while len(batch) < 500:
                            batch.append(q.get_nowait())
                    except queue.Empty:
                        pass
                    if batch:
                        self.wfile.write(b"data: " + json.dumps(batch).encode() + b"\n\n")
                    else:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", default=str(ROOT / "results" / "live" / "events.jsonl"),
                   help="要監看的事件檔（run_experiments.py --dashboard 會寫到這裡）")
    p.add_argument("--replay", default=None, help="重播：events.jsonl 或含 events.jsonl 的實驗資料夾")
    p.add_argument("--speed", type=float, default=1.0, help="重播倍速")
    p.add_argument("--loop", action="store_true", help="重播完自動重來")
    p.add_argument("--claude-dir", default=None,
                   help="Claude Code 紀錄資料夾（預設自動找 ~/.claude/projects/<本專案>）")
    p.add_argument("--project-root", default=str(ROOT), help="用來找 Claude Code 紀錄的專案路徑")
    p.add_argument("--claude-hours", type=float, default=12.0, help="啟動時載入最近幾小時的 Claude Code 紀錄")
    p.add_argument("--no-claude", action="store_true")
    p.add_argument("--idle-replay", type=float, default=20.0,
                   help="幾秒沒有新事件就循環重播最近一次 run（0 = 關閉）")
    p.add_argument("--idle-speed", type=float, default=3.0, help="閒置重播的倍速")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)

    prices = json.loads((HERE / "prices.json").read_text(encoding="utf-8"))
    hub = Hub()
    if a.replay:
        rp = Path(a.replay)
        rp = rp / "events.jsonl" if rp.is_dir() else rp
        threading.Thread(target=replay, args=(hub, rp, a.speed, a.loop), daemon=True).start()
        print(f"重播 {rp}（{a.speed}x）")
    else:
        ep = Path(a.events)
        threading.Thread(target=tail_events, args=(hub, ep, False), daemon=True).start()
        print(f"監看 {ep}")
        if a.idle_replay > 0:
            threading.Thread(target=idle_replay, args=(hub, ep.parent.parent, a.idle_replay, a.idle_speed),
                             daemon=True).start()
    if not a.no_claude:
        dirs = [Path(a.claude_dir)] if a.claude_dir else claude_project_dirs(Path(a.project_root).resolve())
        if dirs:
            print("Claude Code 紀錄：" + ", ".join(str(d) for d in dirs))
            threading.Thread(target=tail_claude, args=(hub, dirs, a.claude_hours), daemon=True).start()
        else:
            print("⚠ 找不到這個專案的 Claude Code 紀錄（可用 --claude-dir 指定）；下半部先空著")

    srv = ThreadingHTTPServer((a.host, a.port), make_handler(hub, prices))
    srv.daemon_threads = True
    srv.handle_error = lambda *_: None  # 瀏覽器關分頁造成的連線中斷不用印 traceback
    url = f"http://{a.host}:{a.port}"
    print(f"儀表板：{url}（Ctrl+C 結束）")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
