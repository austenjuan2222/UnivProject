# One round of read-only Codex review; feedback goes to docs/review/round-NN.md.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/codex_review.ps1 [-MaxRounds 10]
# Kept ASCII-only: Windows PowerShell 5.1 reads BOM-less .ps1 files as cp950.
param([int]$MaxRounds = 10)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$repo = Split-Path $PSScriptRoot -Parent
$dir = Join-Path $repo 'docs\review'

if (Test-Path (Join-Path $dir 'STOP')) { Write-Output 'STOP file present; not running.'; exit 2 }
$done = @(Get-ChildItem $dir -Filter 'round-*.md' -ErrorAction SilentlyContinue).Count
if ($done -ge $MaxRounds) { Write-Output "Round limit $MaxRounds reached."; exit 3 }
if (-not (Get-Command codex -ErrorAction SilentlyContinue)) { Write-Output 'codex CLI not found.'; exit 4 }

$out = Join-Path $dir ('round-{0:D2}.md' -f ($done + 1))
$log = "$out.log"
# read-only sandbox: Codex can read but not write files or use the network; -o is written by the CLI itself.
# codex exec waits on stdin when it is not a TTY, and PS 5.1 never closes piped stdin, so run via cmd with < nul.
$prompt = 'Read docs/review/codex_prompt.md and follow its instructions exactly. Reply in Traditional Chinese.'
cmd /c "codex exec --sandbox read-only -C `"$repo`" -o `"$out`" `"$prompt`" < nul > `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { Write-Output "codex exec failed; see $log"; exit 1 }
Write-Output $out
