"""測試輔助：沒有安裝 opacus（或沒有 torch）時，直接由原始碼路徑載入其 RDP 分析模組。

opacus/accountants/analysis/rdp.py 只依賴 numpy/scipy；載入後放進 sys.modules，
讓 privacy.py 的 `from opacus.accountants.analysis import rdp` 能用。
設定環境變數 OPACUS_SRC 指向 opacus 原始碼根目錄即可（預設 ../ref/opacus）。
"""
import importlib.util
import os
import sys
import types
from pathlib import Path


def ensure_opacus_rdp() -> bool:
    try:
        from opacus.accountants.analysis import rdp  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - opacus 或 torch 不存在
        pass
    src = Path(os.environ.get("OPACUS_SRC", Path(__file__).resolve().parents[2] / "ref" / "opacus"))
    path = src / "opacus" / "accountants" / "analysis" / "rdp.py"
    if not path.exists():
        return False
    spec = importlib.util.spec_from_file_location("opacus.accountants.analysis.rdp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("opacus", "opacus.accountants", "opacus.accountants.analysis"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["opacus.accountants.analysis"].rdp = mod
    sys.modules["opacus.accountants.analysis.rdp"] = mod
    return True
