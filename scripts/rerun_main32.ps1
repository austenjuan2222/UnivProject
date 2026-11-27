# Rerun the README "attack and availability" main table (32 VD, 28 days, DP sigma=1.1, seeds 1 2 3)
# with the current code, so every row gets per-seed paired differences against the same-version control.
# Usage (from repo root): powershell -ExecutionPolicy Bypass -File scripts/rerun_main32.ps1 [-Seeds 4,5,6] [-OutRoot results\night27\main32_s4_10]
# Output: <OutRoot>/<label>/<timestamp>/summary.csv ; skips labels that already have a summary.csv.
param([int[]]$Seeds = @(1, 2, 3), [string]$OutRoot = 'results\night26\main32')
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo '..\UnivProject-run\.venv\Scripts\python.exe'
$data = Join-Path $repo '..\UnivProject-run\data\processed_28d_32'
$clip = "client-dp-noise=0.01 client-dp-clip=0.2"
$runs = [ordered]@{
    'nocheck'      = "verify-updates=false sign-models=false"
    'hmac'         = ""
    'tamper_nover' = "tamper-rate=0.25 verify-updates=false"
    'tamper_hmac'  = "tamper-rate=0.25"
    'a0_median'    = "aggregator='median'"
    'a0_trimmed'   = "aggregator='trimmed'"
    'drop30'       = "dropout-rate=0.3"
    'a3_fedavg'    = "attack-clients=3"
    'a3_median'    = "attack-clients=3 aggregator='median'"
    'a3_trimmed'   = "attack-clients=3 aggregator='trimmed'"
    'a10_median'   = "attack-clients=10 aggregator='median'"
    'clip_a0'      = $clip
    'clip_a10'     = "$clip attack-clients=10"
    'clipmed_a10'  = "$clip attack-clients=10 aggregator='median'"
    'replay_nover' = "replay-rate=0.25 verify-updates=false"
    'replay_hmac'  = "replay-rate=0.25"
    'a6_median'    = "attack-clients=6 aggregator='median'"
    'a6_trimmed'   = "attack-clients=6 aggregator='trimmed'"
    'a10_trimmed'  = "attack-clients=10 aggregator='trimmed'"
    'clip_a3'      = "$clip attack-clients=3"
    'clip_a14'     = "$clip attack-clients=14"
    'a14_median'   = "attack-clients=14 aggregator='median'"
    'noise_fedavg' = "attack-clients=3 attack-type='noise' attack-scale=1.0"
    'noise_median' = "attack-clients=3 attack-type='noise' attack-scale=1.0 aggregator='median'"
    'alldef_a0'     = "update-auth='ed25519' upload-bits=4 aggregator='median' $clip"
    'alldef_attack' = "update-auth='ed25519' upload-bits=4 aggregator='median' $clip attack-clients=3 tamper-rate=0.1 dropout-rate=0.1"
}
Set-Location $repo
foreach ($k in $runs.Keys) {
    $out = Join-Path $repo (Join-Path $OutRoot $k)
    if (Get-ChildItem $out -Recurse -Filter summary.csv -ErrorAction SilentlyContinue) { Write-Output "skip $k"; continue }
    if (Test-Path (Join-Path $repo 'docs\review\STOP')) { Write-Output 'STOP file present; stopping.'; break }
    Write-Output "$(Get-Date -Format s) start $k"
    $a = @('scripts\run_experiments.py', '--data-dir', $data, '--results-dir', $out, '--sigmas', '1.1', '--skip-nodp', '--seeds') + ($Seeds | ForEach-Object { "$_" })
    if ($runs[$k]) { $a += @('--extra', $runs[$k]) }  # PS 5.1 drops an empty-string argument, so omit --extra entirely
    & $py @a
    Write-Output "$(Get-Date -Format s) done $k exit=$LASTEXITCODE"
}
