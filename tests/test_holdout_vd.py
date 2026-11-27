import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("holdout_vd", Path(__file__).parents[1] / "scripts" / "holdout_vd.py")
hv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hv)


class HoldoutTest(unittest.TestCase):
    def test_every_vd_held_out_exactly_once_and_never_in_its_training_fold(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "d"
            src.mkdir()
            clients = [{"partition": p, "vdid": f"V{p}", "file": f"V{p}.csv"} for p in range(32)]
            for c in clients:
                (src / c["file"]).write_text("time,speed\n", encoding="utf-8")
            (src / "manifest.json").write_text(json.dumps({"clients": clients}), encoding="utf-8")
            hv.make(src)
            seen = []
            for f in range(hv.FOLDS):
                man = json.loads((Path(t) / f"d_hold8_f{f}" / "manifest.json").read_text(encoding="utf-8"))
                kept = {c["orig_partition"] for c in man["clients"]}
                self.assertEqual([c["partition"] for c in man["clients"]], list(range(28)))  # Flower 要 0..n-1
                self.assertFalse(kept & set(hv.held_out(f)))
                seen += hv.held_out(f)
            self.assertEqual(sorted(seen), list(range(32)))


if __name__ == "__main__":
    unittest.main()
