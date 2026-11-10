import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lpwan_budget import battery_days, downlink_round_s, expected_sends, radio_joules, time_on_air_ms, upload_cost  # noqa: E402


class LoraAirtimeTest(unittest.TestCase):
    def test_matches_semtech_calculator(self):
        # Semtech LoRa Calculator：13 bytes、BW125、CR4/5、前導 8
        self.assertAlmostEqual(time_on_air_ms(13, 7), 46.336, places=2)
        self.assertAlmostEqual(time_on_air_ms(13, 12), 1155.072, places=2)

    def test_splits_into_max_payload_packets(self):
        n, t = upload_cost(243, 7)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(t, time_on_air_ms(255, 7) + time_on_air_ms(14, 7))

    def test_gateway_sends_to_every_vd_under_its_duty_cycle(self):
        # 沒有群播：閘道器要逐台送，32 台的總空中時間受 RX2 頻段 10% 工作週期限制
        _, t = upload_cost(500, 12)
        self.assertAlmostEqual(downlink_round_s(500, 12, n_vd=32, duty=0.10), 32 * t / 1000 / 0.10)


class PacketLossTest(unittest.TestCase):
    def test_per_packet_retry_costs_one_over_success_rate(self):
        self.assertAlmostEqual(expected_sends(18, 0.1, "packet"), 18 / 0.9)

    def test_resending_whole_model_blows_up_with_many_packets(self):
        # 任何一包掉了就整份重送：期望送出 k / (1-p)^k 包
        self.assertAlmostEqual(expected_sends(18, 0.1, "whole"), 18 / 0.9 ** 18)
        self.assertGreater(expected_sends(125, 0.1, "whole"), 1000 * 125)

    def test_no_loss_sends_each_packet_once(self):
        for mode in ("packet", "whole"):
            self.assertEqual(expected_sends(7, 0.0, mode), 7)


class EnergyTest(unittest.TestCase):
    def test_radio_energy_is_airtime_times_current_times_voltage(self):
        # 講義 Edge／LPWAN：電量 = 空中時間 × 發射電流 × 電壓（SX1276 +20 dBm 約 120 mA、3.3 V）
        self.assertAlmostEqual(radio_joules(1000.0, ma=120, volts=3.3), 1.0 * 0.120 * 3.3)

    def test_battery_days(self):
        # 12 Wh 電池、每輪 1 J、每天 24 輪 → 12*3600/24 天
        self.assertAlmostEqual(battery_days(1.0, 24, wh=12.0), 12 * 3600 / 24)
        self.assertEqual(battery_days(0.0, 24, wh=12.0), float("inf"))


if __name__ == "__main__":
    unittest.main()


class MulticastFecTest(unittest.TestCase):
    # 講義 LPWAN：下行頻段整個閘道器共用。群播一次送給所有 VD，各台掉的包不同，用 FEC 冗餘包讓每台自己補回（LoRaWAN FUOTA 的做法）
    def test_decodes_original_bytes_despite_losses(self):
        import numpy as np
        from lpwan_budget import fec_decode, fec_packet
        rng = np.random.default_rng(0)
        data = rng.integers(0, 256, 1000, dtype=np.uint8).tobytes()
        cap = 50
        got = [(i, fec_packet(data, cap, i)) for i in range(60) if rng.random() > 0.3]
        self.assertEqual(fec_decode(got, len(data), cap), data)

    def test_too_few_packets_cannot_decode(self):
        from lpwan_budget import fec_decode, fec_packet
        data = bytes(range(200))
        got = [(i, fec_packet(data, 50, i)) for i in (0, 1, 7)]  # 4 個來源包只收到 3 個獨立的
        self.assertIsNone(fec_decode(got, len(data), 50))

    def test_first_k_packets_are_the_plain_source(self):
        from lpwan_budget import fec_packet
        data = bytes(range(120))
        self.assertEqual(fec_packet(data, 50, 1), data[50:100])
        self.assertEqual(fec_packet(data, 50, 2), data[100:] + bytes(30))

    def test_multicast_sends_without_loss_is_k(self):
        from lpwan_budget import multicast_sends
        self.assertEqual(multicast_sends(18, 32, 0.0, seed=0), 18)

    def test_multicast_beats_unicast_retry(self):
        # 32 台逐台補送要 32 × k／0.9 包；群播＋FEC 只要比最慢那台多一點
        from lpwan_budget import multicast_sends
        k = 18
        n = multicast_sends(k, 32, 0.1, seed=1)
        self.assertGreater(n, k / 0.9)
        self.assertLess(n, 2 * k)

    def test_multicast_repeat_each_packet_until_everyone_has_it(self):
        # 沒有 FEC 的群播：每包重送到 32 台都收到；期望次數 = Σ_t [1 − (1 − loss^t)^n]
        from lpwan_budget import multicast_repeat
        self.assertEqual(multicast_repeat(18, 32, 0.0), 18)
        self.assertAlmostEqual(multicast_repeat(1, 1, 0.5), 2.0, places=6)
