"""用第三方 LoRaWAN 模擬器 LoRaEnergySim（GillesC/LoRaEnergySim，GPL-3.0，SimPy）驗證下行時間估算（第 49 頁限制第 3 項）。

模擬器不放進這個 repo：先下載解壓到一個資料夾，用 --les 指過去；它需要 simpy（用 --deps 指到解壓好的 simpy）。
它模擬 Class A：VD 送上行（各自受 1% 工作週期限制、會互相碰撞），閘道器只能在該次上行後的 RX1（同頻道、1%）或
RX2（869.525 MHz、10%）回一包下行，閘道器的工作週期整台共用。原本下行固定 12 bytes 的 ACK，這裡只把大小換成
「一包模型」（應用層 cap + 13 bytes LoRaWAN 標頭），其餘照模擬器原本的規則。數到每台收到 k 包下行要多久。

  python scripts/lpwan_les_sim.py --les <LoRaEnergySim 根目錄> --deps <simpy 所在資料夾> --bytes 6404 --sf 12
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpwan_budget import LORAWAN_OVERHEAD, MAX_PAYLOAD, downlink_round_s  # noqa: E402


def completion_s(times_ms: dict, k: int, n_nodes: int) -> float | None:
    """每台 VD 收到的下行時間（ms）→ 全部 n_nodes 台都收滿 k 包的時間（秒）；有一台沒收滿就是 None。"""
    if len(times_ms) < n_nodes or any(len(t) < k for t in times_ms.values()):
        return None
    return max(sorted(t)[k - 1] for t in times_ms.values()) / 1000


def dl_payload(node_id, got: dict, k: int, cap: int) -> int:
    """這次下行要送多大：還沒收滿 k 包就送一包模型（cap + 13 bytes 標頭），收滿了只回模擬器原本的 12 bytes 確認。"""
    return cap + LORAWAN_OVERHEAD if len(got.get(node_id, [])) < k else 12


def simulate(les: str, deps: str, nbytes: int, sf: int, n_nodes: int = 32, hours: float = 72, seed: int = 0,
             radius_m: int = 1000, rx2_sf: int = 12) -> dict:
    sys.path[:0] = [deps, les]
    import numpy as np
    import random
    import Simulations.GlobalConfig as G
    G.PRINT_ENABLED = G.LOG_ENABLED = False
    G.MAX_DELAY_START_PER_NODE_MS = 10_000  # 原設定是模擬時間的 1/10，這裡要所有 VD 同時開始收新模型
    import simpy
    from Framework import PropagationModel
    from Framework.AirInterface import AirInterface
    from Framework.EnergyProfile import EnergyProfile
    from Framework.Gateway import Gateway
    from Framework.LoRaParameters import LoRaParameters
    from Framework.Location import Location
    from Framework.Node import Node
    from Framework.SNRModel import SNRModel

    np.random.seed(seed)
    random.seed(seed)
    LoRaParameters.RX_2_DEFAULT_SF = rx2_sf
    cap = MAX_PAYLOAD[sf]
    k = math.ceil(nbytes / cap)
    got: dict = {}
    cur = [None]  # 閘道器正在回覆哪一台（check_duty_cycle 本身不知道）
    orig = Gateway.check_duty_cycle
    Gateway.check_duty_cycle = lambda self, _size, s, f, now: orig(self, dl_payload(cur[0], got, k, cap), s, f, now)
    orig_rx = Gateway.packet_received

    def packet_received(self, node, packet, now):
        cur[0] = node.id
        done = len(got.get(node.id, [])) >= k
        dl = orig_rx(self, node, packet, now)
        if done:
            return dl  # 已經收滿的只是確認，不算模型下行
        m = dl.meta
        if not m.weak_packet and not m.dc_limit_reached and m.scheduled_receive_slot is not None:
            got.setdefault(node.id, []).append(now)
        return dl

    Gateway.packet_received = packet_received
    # 模擬器每送一包就呼叫一次 gc.collect、並把每次功率變化存進清單：模擬 30 分鐘要跑 6 分鐘、長時間會吃掉好幾 GB。
    # 這兩個只影響速度和畫圖用的紀錄，不影響誰收到哪一包
    import gc
    orig_gc, orig_tp = gc.collect, Node.track_power
    gc.collect, Node.track_power = (lambda *a: 0), (lambda self, p: None)
    try:
        env = simpy.Environment()
        mid = 0
        gw = Gateway(env, Location(mid, mid))
        air = AirInterface(gw, PropagationModel.LogShadow(std=7.9), SNRModel(), env)
        tx = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}
        rx = {"pre_mW": 8.2, "pre_ms": 3.4, "rx_lna_on_mW": 39, "rx_lna_off_mW": 34, "post_mW": 8.3, "post_ms": 10.7}
        for i in range(n_nodes):
            p = LoRaParameters(freq=LoRaParameters.DEFAULT_CHANNELS[i % 3], sf=sf, bw=125, cr=5, crc_enabled=1,
                               de_enabled=0, header_implicit_mode=0, tp=14)
            a = 2 * math.pi * i / n_nodes
            r = radius_m * (0.3 + 0.7 * (i % 4) / 3)
            node = Node(i, EnergyProfile(5.7e-3, 15, tx, rx_power=rx), p, sleep_time=1, process_time=5, adr=False,
                        location=Location(r * math.cos(a), r * math.sin(a)), base_station=gw, env=env,
                        payload_size=4, air_interface=air, confirmed_messages=True)
            env.process(node.run())
        env.run(until=hours * 3600 * 1000)
    finally:
        Gateway.check_duty_cycle, Gateway.packet_received = orig, orig_rx
        gc.collect, Node.track_power = orig_gc, orig_tp
    return {"packets_per_vd": k, "done_s": completion_s(got, k, n_nodes),
            "min_dl": min((len(v) for v in got.values()), default=0), "nodes_heard": len(got),
            "dl_not_schedulable": gw.dl_not_schedulable, "estimate_s": downlink_round_s(nbytes, sf, n_nodes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", required=True)
    ap.add_argument("--deps", required=True)
    ap.add_argument("--bytes", type=int, required=True, help="每台要收的下行大小（含 64 B 簽章）")
    ap.add_argument("--sf", type=int, default=12)
    ap.add_argument("--rx2-sf", type=int, default=12)
    ap.add_argument("--hours", type=float, default=72)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(simulate(a.les, a.deps, a.bytes, a.sf, hours=a.hours, seed=a.seed, rx2_sf=a.rx2_sf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
