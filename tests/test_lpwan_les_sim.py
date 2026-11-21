"""LoRaEnergySim 包裝：從每台收到下行的時間算「全部收滿」要多久。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lpwan_les_sim import completion_s, dl_payload  # noqa: E402


class CompletionTest(unittest.TestCase):
    def test_slowest_vd_decides(self):
        t = {0: [1000, 5000, 3000], 1: [2000, 4000]}
        self.assertEqual(completion_s(t, 2, 2), 4.0)  # 0 號第 2 包在 3 秒、1 號在 4 秒

    def test_missing_vd_or_too_few_packets_is_none(self):
        self.assertIsNone(completion_s({0: [1, 2]}, 2, 2))
        self.assertIsNone(completion_s({0: [1, 2], 1: [1]}, 2, 2))


class DownlinkSizeTest(unittest.TestCase):
    def test_vd_that_already_has_the_model_only_gets_an_ack(self):
        # 收滿 k 包的 VD 不該再佔用閘道器送一整包模型的時間，只回原本 12 bytes 的確認
        got = {0: [1, 2], 1: [1]}
        self.assertEqual(dl_payload(0, got, 2, 51), 12)
        self.assertEqual(dl_payload(1, got, 2, 51), 64)
        self.assertEqual(dl_payload(2, got, 2, 51), 64)


if __name__ == "__main__":
    unittest.main()
