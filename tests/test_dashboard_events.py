"""即時儀表板：事件檔寫入／讀取、Claude Code 紀錄解析、重播時間軸。"""
import json

import pytest

from dashboard_events import (Emitter, Throttle, copy_since, parse_claude_line, read_events,
                              replay_schedule)


def test_emitter_noop_when_path_empty(tmp_path):
    e = Emitter("")
    e.emit("x", a=1)  # 不會出錯、不寫檔
    assert not e.enabled
    assert list(tmp_path.iterdir()) == []


def test_emit_and_read_with_offset(tmp_path):
    p = tmp_path / "live" / "events.jsonl"
    e = Emitter(str(p))
    e.emit("run_start", run="r1", rounds=3)
    e.emit("progress", pid=0, step=5, total=10)
    evs, off = read_events(p, 0)
    assert [x["kind"] for x in evs] == ["run_start", "progress"]
    assert evs[1]["step"] == 5 and "t" in evs[0]
    e.emit("train_end", pid=0)
    evs2, off2 = read_events(p, off)
    assert [x["kind"] for x in evs2] == ["train_end"] and off2 > off


def test_read_events_skips_partial_and_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"kind":"a","t":1}\nnot json\n{"kind":"b"')  # 最後一行還沒寫完
    evs, off = read_events(p, 0)
    assert [x["kind"] for x in evs] == ["a"]
    with p.open("ab") as f:
        f.write(b',"t":2}\n')
    evs, _ = read_events(p, off)
    assert [x["kind"] for x in evs] == ["b"]


def _burst(path, tag, n):
    e = Emitter(path)
    for i in range(n):
        e.emit("progress", tag=tag, i=i, pad="x" * 200)


def test_concurrent_processes_do_not_interleave_lines(tmp_path):
    # Windows 的 O_APPEND 是「先移到檔尾再寫」，多行程同時寫會互相截斷（實測約 1.6% 的行壞掉）
    import multiprocessing as mp
    p = tmp_path / "e.jsonl"
    procs = [mp.Process(target=_burst, args=(str(p), k, 300)) for k in range(4)]
    for x in procs:
        x.start()
    for x in procs:
        x.join()
    lines = p.read_bytes().split(b"\n")[:-1]
    assert len(lines) == 1200
    assert all(json.loads(x)["kind"] == "progress" for x in lines)


def test_latest_run_events_skips_live_and_empty(tmp_path):
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
    from server import latest_run_events
    assert latest_run_events(tmp_path) is None
    old, new, live, empty = (tmp_path / d / "events.jsonl" for d in ("b1/dp", "b2/dp", "live", "b3/dp"))
    for p, body in ((old, "{}\n"), (new, "{}\n"), (live, "{}\n"), (empty, "")):
        p.parent.mkdir(parents=True)
        p.write_text(body)
    os.utime(old, (1, 1))
    os.utime(live, (2e9, 2e9))
    os.utime(empty, (2e9, 2e9))
    assert latest_run_events(tmp_path) == new


def test_read_events_missing_file(tmp_path):
    assert read_events(tmp_path / "nope.jsonl", 0) == ([], 0)


def test_emit_values_are_json_safe(tmp_path):
    p = tmp_path / "e.jsonl"
    Emitter(str(p)).emit("x", loss=float("nan"), eps=float("inf"))
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["loss"] is None and rec["eps"] is None


def test_copy_since(tmp_path):
    src, dst = tmp_path / "live.jsonl", tmp_path / "run" / "events.jsonl"
    e = Emitter(str(src))
    e.emit("old")
    off = src.stat().st_size
    e.emit("new1")
    e.emit("new2")
    copy_since(src, off, dst)
    assert [x["kind"] for x in read_events(dst, 0)[0]] == ["new1", "new2"]


def test_throttle_emits_about_n_times_plus_last():
    th = Throttle(parts=10)
    hits = [s for s in range(1, 101) if th.due(s, 100)]
    assert len(hits) == 10 and hits[-1] == 100
    th2 = Throttle(parts=10)
    assert [s for s in range(1, 4) if th2.due(s, 3)] == [1, 2, 3]


def test_parse_claude_line_usage_tools_and_dedupe():
    seen = set()
    line = json.dumps({"type": "assistant", "timestamp": "2026-11-03T10:00:00Z", "isSidechain": True,
                       "message": {"id": "m1", "model": "claude-opus-x",
                                   "usage": {"input_tokens": 10, "output_tokens": 20,
                                             "cache_read_input_tokens": 300,
                                             "cache_creation_input_tokens": 40},
                                   "content": [{"type": "tool_use", "name": "Bash"}]}})
    ev = parse_claude_line(line, seen)
    assert ev["kind"] == "cc" and ev["model"] == "claude-opus-x" and ev["sub"] is True
    assert (ev["in"], ev["out"], ev["cache_read"], ev["cache_write"]) == (10, 20, 300, 40)
    assert ev["tools"] == ["Bash"]
    # 同一則訊息的第二個 content 區塊：token 不重複計算，但工具照算
    line2 = line.replace('"Bash"', '"Read"')
    ev2 = parse_claude_line(line2, seen)
    assert ev2["in"] == 0 and ev2["out"] == 0 and ev2["tools"] == ["Read"]


def test_parse_claude_line_ignores_other_lines():
    seen = set()
    assert parse_claude_line("garbage", seen) is None
    assert parse_claude_line(json.dumps({"type": "summary"}), seen) is None
    err = json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "is_error": True, "content": "boom"}]}})
    assert parse_claude_line(err, seen)["kind"] == "cc_error"


def test_replay_schedule_compresses_gaps():
    evs = [{"t": 100.0}, {"t": 100.5}, {"t": 160.0}, {"kind": "no-time"}]
    delays = replay_schedule(evs, speed=2.0, max_gap=1.0)
    assert delays == pytest.approx([0.0, 0.25, 1.0, 0.0])


def test_claude_project_dirs_matches_slug(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
    from server import claude_project_dirs
    base = tmp_path / "projects"
    (base / "C--Users-TESTER-univproject").mkdir(parents=True)
    (base / "other").mkdir()
    assert [d.name for d in claude_project_dirs(Path("/x/y/univproject"), base)] == \
        ["C--Users-TESTER-univproject"]
    assert claude_project_dirs(Path("/x/nothing"), base) == []
    # worktree／venv 兄弟資料夾也要收進來，但不能誤收 univprojectx 這種
    for n in ("C--Users-TESTER-Documents-UnivProject-git", "C--Users-TESTER-Documents-UnivProject-run",
              "C--Users-TESTER-UnivProjectX"):
        (base / n).mkdir()
    got = [d.name for d in claude_project_dirs(Path("/d/UnivProject-dashboard"), base)]
    assert got == ["C--Users-TESTER-Documents-UnivProject-git", "C--Users-TESTER-Documents-UnivProject-run",
                   "C--Users-TESTER-univproject"]
    assert claude_project_dirs(Path("/x"), tmp_path / "missing") == []
