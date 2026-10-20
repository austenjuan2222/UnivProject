import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fix_torch_windows import runtime_dlls  # noqa: E402


class RuntimeDllsTest(unittest.TestCase):
    def test_finds_vc_runtime_dlls_outside_torch_lib(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lib = root / "Lib" / "site-packages" / "torch" / "lib"
            lib.mkdir(parents=True)
            for name in ("vcruntime140.dll", "VCRUNTIME140_1.dll", "msvcp140.dll", "python3.dll"):
                (root / name).write_bytes(b"x")
            (lib / "vcruntime140.dll").write_bytes(b"x")
            found = [p.name for p in runtime_dlls(root, exclude=lib)]
            self.assertEqual(sorted(n.lower() for n in found),
                             ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"])


if __name__ == "__main__":
    unittest.main()
