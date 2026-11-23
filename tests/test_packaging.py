"""pyproject 的 py-modules 要包含根目錄所有模組：漏一個，從 wheel 安裝後 import 就會失敗（原始碼目錄跑不出這個錯）。"""
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class PackagingTest(unittest.TestCase):
    def test_every_root_module_is_packaged(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        packaged = set(cfg["tool"]["setuptools"]["py-modules"])
        on_disk = {p.stem for p in ROOT.glob("*.py")}
        self.assertEqual(on_disk - packaged, set())


if __name__ == "__main__":
    unittest.main()
