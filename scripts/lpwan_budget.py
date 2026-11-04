#!/usr/bin/env python3
"""每輪上傳在 LoRaWAN 上要多久（對應講義 LPWAN：頻寬小、有工作週期限制）。

用 Semtech LoRa 空中時間公式（AN1200.13），BW 125 kHz、CR 4/5、前導 8 個符號、顯式標頭、有 CRC；
每包 LoRaWAN 標頭 13 bytes。工作週期 1%（EU868 規定；台灣 AS923 改用每包 400 ms 停留時間限制）。

  python scripts/lpwan_budget.py
"""
from __future__ import annotations

import math

LORAWAN_OVERHEAD = 13
# 各資料速率每包最大應用層酬載（LoRaWAN Regional Parameters，未啟用停留時間限制）
MAX_PAYLOAD = {7: 242, 9: 115, 12: 51}


def time_on_air_ms(phy_bytes: int, sf: int, bw_hz: int = 125_000, cr: int = 1, preamble: int = 8) -> float:
    t_sym = (2 ** sf) / bw_hz * 1000
    de = 1 if sf >= 11 and bw_hz == 125_000 else 0  # 低速最佳化
    n = 8 + max(math.ceil((8 * phy_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n * t_sym


def upload_cost(nbytes: int, sf: int) -> tuple[int, float]:
    """回傳（封包數, 總空中時間 ms）。"""
    cap = MAX_PAYLOAD[sf]
    packets = math.ceil(nbytes / cap)
    full, last = divmod(nbytes, cap)
    t = full * time_on_air_ms(cap + LORAWAN_OVERHEAD, sf)
    if last:
        t += time_on_air_ms(last + LORAWAN_OVERHEAD, sf)
    return packets, t


def downlink_round_s(nbytes: int, sf: int, n_vd: int = 32, duty: float = 0.10) -> float:
    """閘道器把全域模型逐台送給 n_vd 個 VD（Class A 沒有群播）至少要多少秒。
    EU868 下行用 RX2（869.525 MHz）頻段，工作週期 10%，而且是整個閘道器共用，不是每台 VD 各自 1%。"""
    _, t = upload_cost(nbytes, sf)
    return n_vd * t / 1000 / duty


def main() -> int:
    # 32 VD 實驗的每 VD 每輪上傳量（README 壓縮上傳一節），+32 B HMAC 標籤
    sizes = {"32 位元（不壓縮）": 6276, "8 位元": 1597, "4 位元": 809}
    print("| 上傳 | SF | 封包數 | 空中時間 | 1% 工作週期下每輪至少 |")
    print("|---|---|---|---|---|")
    for name, b in sizes.items():
        for sf in (7, 12):
            n, t = upload_cost(b + 32, sf)
            print(f"| {name} {b + 32} B | SF{sf} | {n} | {t / 1000:.1f} 秒 | {t * 100 / 1000 / 60:.1f} 分鐘 |")
    # 下行：每台 VD 收到的全域模型（量化差值），+64 B Ed25519 簽章；閘道器逐台送 32 台
    down = {"32 位元（不壓縮）": 6276, "8 位元": 1593, "4 位元": 809, "2 位元": 417}
    print("\n| 下行 | SF | 每台封包數 | 每台空中時間 | 閘道器 10% 工作週期下送完 32 台至少 |")
    print("|---|---|---|---|---|")
    for name, b in down.items():
        for sf in (7, 12):
            n, t = upload_cost(b + 64, sf)
            print(f"| {name} {b + 64} B | SF{sf} | {n} | {t / 1000:.1f} 秒 | {downlink_round_s(b + 64, sf) / 60:.1f} 分鐘 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
