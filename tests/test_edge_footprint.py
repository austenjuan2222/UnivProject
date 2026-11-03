import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_footprint import crypto_costs  # noqa: E402


class CryptoCostsTest(unittest.TestCase):
    def test_reports_times_and_secagg_doubles_upload(self):
        arrays = {"w": np.ones((4, 3), np.float32), "b": np.zeros(3, np.float32)}
        c = crypto_costs(arrays, peers=4, reps=5)
        for k in ("verify_model_ms", "hmac_ms", "ed25519_sign_ms", "secagg_mask_ms"):
            self.assertGreater(c[k], 0)
        self.assertEqual(c["secagg_upload_bytes"], 2 * sum(a.nbytes for a in arrays.values()))


if __name__ == "__main__":
    unittest.main()
