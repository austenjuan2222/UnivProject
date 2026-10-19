import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import client_info, load_client_arrays, make_windows, split_series


class MakeWindowsTest(unittest.TestCase):
    def test_shapes_and_alignment(self):
        x = np.arange(20, dtype=float)
        X, y = make_windows(x, lookback=12)
        self.assertEqual(X.shape, (8, 12))
        np.testing.assert_array_equal(X[0], np.arange(12))
        self.assertEqual(y[0], 12)
        self.assertEqual(y[-1], 19)

    def test_windows_with_nan_are_dropped(self):
        x = np.arange(20, dtype=float)
        x[5] = np.nan
        X, y = make_windows(x, lookback=12)
        # 含索引 5 的視窗（目標 12..17）以及目標本身為 NaN 的都要去掉
        self.assertEqual(list(y), [18.0, 19.0])

    def test_too_short_series_gives_empty(self):
        X, y = make_windows(np.arange(5, dtype=float), lookback=12)
        self.assertEqual(X.shape, (0, 12))
        self.assertEqual(y.shape, (0,))


class SplitSeriesTest(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(50, 100, 200)

    def test_no_window_crosses_split(self):
        d = split_series(self.x, lookback=12, train_frac=0.8)
        split = 160
        raw_train_targets = d["y_train"] * d["std"] + d["mean"]
        raw_test_inputs = d["X_test"] * d["std"] + d["mean"]
        self.assertTrue(np.all(raw_train_targets <= self.x[split - 1] + 1e-3))
        self.assertTrue(np.all(raw_test_inputs >= self.x[split] - 1e-3))
        self.assertEqual(len(d["y_train"]), split - 12)
        self.assertEqual(len(d["y_test"]), 200 - split - 12)

    def test_normalisation_uses_train_part_only(self):
        d = split_series(self.x, lookback=12, train_frac=0.8)
        changed = self.x.copy()
        changed[160:] += 1000.0
        d2 = split_series(changed, lookback=12, train_frac=0.8)
        self.assertAlmostEqual(d["mean"], d2["mean"])
        self.assertAlmostEqual(d["std"], d2["std"])
        np.testing.assert_allclose(d["X_train"], d2["X_train"])
        self.assertAlmostEqual(d["mean"], float(np.mean(self.x[:160])))

    def test_constant_series_does_not_divide_by_zero(self):
        d = split_series(np.full(100, 80.0), lookback=12)
        self.assertTrue(np.isfinite(d["X_train"]).all())

    def test_arrays_are_float32(self):
        d = split_series(self.x, lookback=12)
        self.assertEqual(d["X_train"].dtype, np.float32)
        self.assertEqual(d["y_test"].dtype, np.float32)


class ClientInfoTest(unittest.TestCase):
    def _make_dir(self, d, n=3):
        clients = []
        for i in range(n):
            f = f"VD{i}.csv"
            pd.DataFrame({
                "time": pd.date_range("2024-03-04", periods=100, freq="5min"),
                "speed": np.linspace(60, 90, 100) + i,
                "volume": 10, "occupancy": 5.0,
            }).to_csv(Path(d, f), index=False)
            clients.append({"partition": i, "vdid": f"VD-{i}", "file": f, "valid_ratio": 1.0})
        Path(d, "manifest.json").write_text(json.dumps({"clients": clients}), encoding="utf-8")

    def test_maps_partition_to_vd(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_dir(d)
            info = client_info(d, partition_id=2, num_partitions=3)
            self.assertEqual(info["vdid"], "VD-2")

    def test_mismatched_client_count_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_dir(d)
            with self.assertRaisesRegex(ValueError, "num-supernodes"):
                client_info(d, partition_id=0, num_partitions=5)

    def test_missing_manifest_raises_helpful_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "preprocess_vd.py"):
                client_info(d, partition_id=0, num_partitions=1)

    def test_load_client_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_dir(d)
            arr = load_client_arrays(d, 1, 3, lookback=12, train_frac=0.8)
            self.assertEqual(arr["vdid"], "VD-1")
            self.assertEqual(arr["X_train"].shape, (68, 12))
            self.assertEqual(arr["X_test"].shape, (8, 12))


if __name__ == "__main__":
    unittest.main()
