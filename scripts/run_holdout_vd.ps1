# Train the 8 leave-4-out folds (28 VD each, record config: DP sigma=1.1, HMAC + model signing) for scripts/holdout_vd.py.
# Usage (from repo root): powershell -ExecutionPolicy Bypass -File scripts/run_holdout_vd.ps1 [-Seeds 1,2,3]
param([int[]]$Seeds = @(1, 2, 3), [string]$OutRoot = 'results\night27\holdout')
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo '..\UnivProject-run\.venv\Scripts\python.exe'
Set-Location $repo
foreach ($f in 0..7) {
    $data = Join-Path $repo "..\UnivProject-run\data\processed_28d_32_hold8_f$f"
    $out = Join-Path $repo (Join-Path $OutRoot "f$f")
    Write-Output "$(Get-Date -Format s) start fold $f"
    & $py scripts\run_experiments.py --data-dir $data --results-dir $out --sigmas 1.1 --skip-nodp --seeds ($Seeds | ForEach-Object { "$_" })
    Write-Output "$(Get-Date -Format s) done fold $f exit=$LASTEXITCODE"
}
