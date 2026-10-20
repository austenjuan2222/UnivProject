#!/usr/bin/env python3
"""Windows：檢查 PyTorch 能否載入；不行就診斷並嘗試不需系統管理員權限的修復。

常見錯誤：OSError [WinError 126] ... torch\\lib\\shm.dll（或 c10.dll / fbgemm.dll）
多半是缺 Microsoft Visual C++ 執行階段。修復順序：
  1. pip 安裝 msvc-runtime，把 vcruntime/msvcp 等 DLL 複製到 torch\\lib（不需管理員）
  2. 改裝 torch 2.6.0（opacus 1.6 支援的最低版本）
結束碼 0 = torch 可用；1 = 仍失敗（請安裝 https://aka.ms/vs/17/release/vc_redist.x64.exe）。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_PREFIXES = ("vcruntime140", "msvcp140", "concrt140", "vcomp140", "vccorlib140")


def torch_ok() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""


def runtime_dlls(root: Path, exclude: Path | None = None) -> list[Path]:
    """在 root 底下找 VC++ 執行階段 DLL（排除 exclude 目錄）。"""
    out = []
    for p in Path(root).rglob("*.dll"):
        if exclude is not None and exclude in p.parents:
            continue
        if p.name.lower().startswith(RUNTIME_PREFIXES):
            out.append(p)
    return sorted(out)


def torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "lib"


def diagnose(lib: Path) -> None:
    print(f"python={sys.version.split()[0]} prefix={sys.prefix}")
    sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll"):
        print(f"  System32\\{name}: {'有' if (sysdir / name).exists() else '缺'}")
    if lib is None or not lib.exists():
        print("  找不到 torch\\lib")
        return
    import ctypes
    os.add_dll_directory(str(lib))
    bad = []
    for dll in sorted(lib.glob("*.dll")):
        try:
            ctypes.WinDLL(str(dll))
        except OSError as e:
            bad.append(f"{dll.name}: {e}")
    print(f"  torch\\lib 無法載入的 DLL：{len(bad)} 個")
    for b in bad[:15]:
        print("   ", b)


def pip(*args) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=False)


def main() -> int:
    ok, msg = torch_ok()
    if ok:
        print(f"torch OK {msg}")
        return 0
    print(f"torch 載入失敗：{msg}")
    lib = torch_lib_dir()
    diagnose(lib)

    print("== 修復 1：msvc-runtime → 複製 DLL 到 torch\\lib ==")
    pip("msvc-runtime")
    if lib is not None:
        for d in runtime_dlls(Path(sys.prefix), exclude=lib):
            shutil.copy2(d, lib / d.name)
            print("  複製", d.name)
    ok, msg = torch_ok()
    if ok:
        print(f"torch OK {msg}（修復 1 生效）")
        return 0
    print(f"仍失敗：{msg}")

    print("== 修復 2：改裝 torch 2.6.0 ==")
    pip("torch==2.6.0")
    lib = torch_lib_dir()
    if lib is not None:
        for d in runtime_dlls(Path(sys.prefix), exclude=lib):
            shutil.copy2(d, lib / d.name)
    ok, msg = torch_ok()
    if ok:
        print(f"torch OK {msg}（修復 2 生效）")
        return 0
    print(f"仍失敗：{msg}")
    diagnose(lib)
    print("請安裝 Microsoft Visual C++ Redistributable：https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return 1


if __name__ == "__main__":
    sys.exit(main())
