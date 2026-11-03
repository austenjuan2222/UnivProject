"""即時儀表板的事件匯流排（純 Python，不依賴 Flower／PyTorch）。

Flower 模擬時每個 client 跑在獨立行程，因此所有事件都「附加」寫進同一個 JSONL 檔，
儀表板伺服器（dashboard/server.py）再即時讀取。路徑空字串 = 關閉，完全不影響實驗。

每行一個事件：{"t": <unix 秒>, "kind": "<種類>", ...}
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


def _safe(v):
    """NaN／inf 轉成 null，numpy 純量轉成 Python 型別，確保輸出是合法 JSON。"""
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    return v


class Emitter:
    """寫事件。多個行程同時附加寫入：每個事件一次 os.write 寫完整一行。"""

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def emit(self, kind: str, **data) -> None:
        if self.path is None:
            return
        rec = {"t": round(time.time(), 3), "kind": kind}
        rec.update({k: _safe(v) for k, v in data.items()})
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass  # 儀表板只是觀察用，寫不進去也不能中斷訓練

    def size(self) -> int:
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0


class Throttle:
    """進度事件節流：一個 epoch 大約回報 parts 次，最後一步一定回報。"""

    def __init__(self, parts: int = 20):
        self.parts = parts

    def due(self, step: int, total: int) -> bool:
        every = max(1, total // self.parts)
        return step == total or step % every == 0


def read_events(path, offset: int = 0) -> tuple[list[dict], int]:
    """從 offset 開始讀完整的行；回傳（事件, 新 offset）。尚未寫完的最後一行留到下次。"""
    p = Path(path)
    try:
        with p.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        return [], offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    out = []
    for raw in chunk[:end].split(b"\n"):
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out, offset + end + 1


def copy_since(src, offset: int, dst) -> None:
    """把 src 從 offset 之後的內容存成 dst（每個實驗結束時留一份，可重播）。"""
    try:
        with Path(src).open("rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(data)


def parse_claude_line(line: str, seen_ids: set) -> dict | None:
    """把 Claude Code 對話紀錄（~/.claude/projects/<專案>/*.jsonl）的一行轉成儀表板事件。

    同一則 assistant 訊息會分成多行（每個 content 區塊一行）且帶相同 usage，
    用 message.id 去重，避免 token 重複計算。
    """
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    msg = obj.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    blocks = content if isinstance(content, list) else []
    if obj.get("type") == "assistant" and isinstance(msg, dict):
        mid = msg.get("id")
        first = mid not in seen_ids
        if mid:
            seen_ids.add(mid)
        u = msg.get("usage") or {}
        g = (lambda k: int(u.get(k) or 0)) if first else (lambda k: 0)
        return {"kind": "cc", "ts": obj.get("timestamp"), "model": msg.get("model") or "?",
                "sub": bool(obj.get("isSidechain")),
                "in": g("input_tokens"), "out": g("output_tokens"),
                "cache_read": g("cache_read_input_tokens"),
                "cache_write": g("cache_creation_input_tokens"),
                "tools": [b.get("name", "?") for b in blocks
                          if isinstance(b, dict) and b.get("type") == "tool_use"]}
    if obj.get("type") == "user":
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                return {"kind": "cc_error", "ts": obj.get("timestamp")}
    return None


def replay_schedule(events: list[dict], speed: float = 1.0, max_gap: float = 2.0) -> list[float]:
    """重播時每個事件前要等幾秒：依原始時間差 / speed，過長的空檔壓到 max_gap。"""
    out, prev = [], None
    for e in events:
        t = e.get("t")
        if t is None or prev is None:
            out.append(0.0)
        else:
            out.append(min(max(0.0, (t - prev) / speed), max_gap))
        if t is not None:
            prev = t
    return out
