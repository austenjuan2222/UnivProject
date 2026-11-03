import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lpwan_budget import time_on_air_ms, upload_cost  # noqa: E402


class LoraAirtimeTest(unittest.TestCase):
    def test_matches_semtech_calculator(self):
        # Semtech LoRa Calculator：13 bytes、BW125、CR4/5、前導 8
        self.assertAlmostEqual(time_on_air_ms(13, 7), 46.336, places=2)
        self.assertAlmostEqual(time_on_air_ms(13, 12), 1155.072, places=2)

    def test_splits_into_max_payload_packets(self):
        n, t = upload_cost(243, 7)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(t, time_on_air_ms(255, 7) + time_on_air_ms(14, 7))


if __name__ == "__main__":
    unittest.main()
