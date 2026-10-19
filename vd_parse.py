"""解析高公局 / MOTC 動態路況標準 v2.0 的 VDLive XML。

每個 VD 每分鐘輸出一列：vdid, time, speed, volume, occupancy, status。
- 車道車速 <= 0 或 >= 200（含 -99 等錯誤碼）視為無效。
- VD 車速 = 有效車道的流量加權平均；有效車道流量皆為 0 時取簡單平均。
- Status != 0 或沒有有效車道時 speed 為 None。
只用標準函式庫，下載腳本可直接 import。
"""
from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from typing import Callable, Optional

MIN_SPEED, MAX_SPEED = 0.0, 200.0
FIELDS = ["vdid", "time", "speed", "volume", "occupancy", "status"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(elem, name):
    for c in elem:
        if _local(c.tag) == name:
            return c
    return None


def _children(elem, name):
    return [c for c in elem if _local(c.tag) == name]


def _text(elem, name, default=None):
    c = _child(elem, name)
    return c.text.strip() if c is not None and c.text is not None else default


def _num(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _iter_lanes(vdlive):
    for lf_parent in _children(vdlive, "LinkFlows"):
        for lf in _children(lf_parent, "LinkFlow"):
            for lanes in _children(lf, "Lanes"):
                yield from _children(lanes, "Lane")


def _lane_volume(lane) -> float:
    total = 0.0
    for vs in _children(lane, "Vehicles"):
        for v in _children(vs, "Vehicle"):
            vol = _num(_text(v, "Volume"), 0.0)
            if vol > 0:
                total += vol
    return total


def _parse_vd(vdlive) -> dict:
    status = int(_num(_text(vdlive, "Status"), -1))
    speeds, vols, occs = [], [], []
    total_volume = 0.0
    for lane in _iter_lanes(vdlive):
        vol = _lane_volume(lane)
        sp = _num(_text(lane, "Speed"))
        if sp is None or not (MIN_SPEED < sp < MAX_SPEED):
            continue
        speeds.append(sp)
        vols.append(vol)
        total_volume += vol
        occ = _num(_text(lane, "Occupancy"))
        if occ is not None and 0 <= occ <= 100:
            occs.append(occ)

    speed: Optional[float]
    if status != 0 or not speeds:
        speed = None
    elif total_volume > 0:
        speed = sum(s * v for s, v in zip(speeds, vols)) / total_volume
    else:
        speed = sum(speeds) / len(speeds)

    return {
        "vdid": _text(vdlive, "VDID", ""),
        "time": _text(vdlive, "DataCollectTime", ""),
        "speed": speed,
        "volume": int(total_volume),
        "occupancy": sum(occs) / len(occs) if occs else None,
        "status": status,
    }


def parse_vdlive_xml(
    data: bytes, keep: Optional[Callable[[str], bool]] = None
) -> list[dict]:
    """解析一個 VDLive XML（可為 gzip 壓縮）成列清單。"""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    root = ET.fromstring(data)
    if _local(root.tag) != "VDLiveList":
        raise ValueError(f"不是 VDLiveList 文件（根節點為 {_local(root.tag)}）")
    rows = []
    for container in _children(root, "VDLives"):
        for vdlive in _children(container, "VDLive"):
            vdid = _text(vdlive, "VDID", "")
            if keep is not None and not keep(vdid):
                continue
            rows.append(_parse_vd(vdlive))
    return rows
