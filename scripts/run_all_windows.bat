@echo off
REM One-click run on Windows: venv -> install -> pytest -> smoke test -> download VD -> preprocess -> experiments
REM Progress: status.txt  Full output: run_log.txt  Skip download: run_all_windows.bat skipdownload
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=%CD%\run_log.txt
set STATUS=%CD%\status.txt
echo started %DATE% %TIME% > "%STATUS%"
echo ==== %DATE% %TIME% ==== > "%LOG%"

set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY py -3 -c "import sys" >nul 2>&1 && set PY=py -3
if not defined PY python -c "import sys" >nul 2>&1 && set PY=python
if not defined PY (echo FAIL no-python >> "%STATUS%" & exit /b 1)
echo python=%PY% >> "%STATUS%"

if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv >> "%LOG%" 2>&1 || (echo FAIL venv >> "%STATUS%" & exit /b 1)
)
set VPY=%CD%\.venv\Scripts\python.exe
"%VPY%" -m pip install -U pip >> "%LOG%" 2>&1
"%VPY%" -m pip install -e ".[dev]" >> "%LOG%" 2>&1 || (echo FAIL install >> "%STATUS%" & exit /b 1)
"%VPY%" -m pip list >> "%LOG%" 2>&1
echo OK install >> "%STATUS%"
"%VPY%" scripts\fix_torch_windows.py >> "%LOG%" 2>&1 && (echo OK torch >> "%STATUS%") || (echo FAIL torch-need-vc_redist >> "%STATUS%" & exit /b 1)

"%VPY%" -m pytest -v >> "%LOG%" 2>&1 && (echo OK pytest >> "%STATUS%") || (echo FAIL pytest >> "%STATUS%")

"%VPY%" scripts\run_experiments.py --smoke >> "%LOG%" 2>&1 && (echo OK smoke >> "%STATUS%") || (echo FAIL smoke >> "%STATUS%" & exit /b 1)

if /i "%1"=="skipdownload" goto preprocess
"%VPY%" scripts\download_vd.py --start 2024-03-04 --days 14 --vd-pattern "N1" --workers 2 >> "%LOG%" 2>&1 && (echo OK download >> "%STATUS%") || (echo WARN download >> "%STATUS%")

:preprocess
"%VPY%" scripts\preprocess_vd.py --num-clients 8 --vd-pattern "^VD-N1-[NS]-[0-9.]+-M-" >> "%LOG%" 2>&1 || (echo FAIL preprocess >> "%STATUS%" & exit /b 1)
echo OK preprocess >> "%STATUS%"

"%VPY%" scripts\run_experiments.py --rounds 20 --sigmas 0.8 1.1 1.5 3 --seeds 1 2 3 >> "%LOG%" 2>&1 || (echo FAIL experiments >> "%STATUS%" & exit /b 1)
echo OK experiments >> "%STATUS%"
echo DONE %DATE% %TIME% >> "%STATUS%"
endlocal
