import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from setup_flwr_home import main, write_config  # noqa: E402


class SetupFlwrHomeTest(unittest.TestCase):
    def test_config_sets_num_supernodes(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_config(Path(d), num_supernodes=6, num_cpus=2)
            cfg = tomllib.loads(path.read_text())
            self.assertEqual(cfg["superlink"]["default"], "local")
            self.assertEqual(cfg["superlink"]["local"]["options"]["num-supernodes"], 6)
            res = cfg["superlink"]["local"]["options"]["backend"]["client-resources"]
            self.assertEqual(res["num-cpus"], 2)
            self.assertEqual(res["num-gpus"], 0)

    def test_main_reads_client_count_from_manifest_and_prints_it(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "manifest.json").write_text(json.dumps({"clients": [{}, {}, {}]}))
            n = main(["--data-dir", d, "--flwr-home", str(Path(d, "home"))])
            self.assertEqual(n, 3)
            cfg = tomllib.loads(Path(d, "home", "config.toml").read_text())
            self.assertEqual(cfg["superlink"]["local"]["options"]["num-supernodes"], 3)


if __name__ == "__main__":
    unittest.main()
