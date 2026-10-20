#!/usr/bin/env bash
# 煙霧測試：不需網路。用「合成」VD 資料跑完整流程（前處理 → DP 與無 DP 各 2 輪 → 畫圖），
# 並檢查輸出。用來確認環境與程式能端到端運作；結果數字沒有意義。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
TMP="$ROOT/.smoke"
rm -rf "$TMP"
mkdir -p "$TMP"

"$PY" "$ROOT/scripts/make_synthetic_vd.py" --out "$TMP/raw" --days 3 --num-vds 3
"$PY" "$ROOT/scripts/preprocess_vd.py" --raw "$TMP/raw" --out "$TMP/processed" --num-clients 3

DATA_DIR="$TMP/processed" RESULTS_DIR="$TMP/results" ROUNDS=2 SIGMAS="1.1" \
  bash "$ROOT/scripts/run_experiments.sh"

BATCH="$(ls -d "$TMP"/results/*/ | tail -1)"
"$PY" - "$BATCH" <<'EOF'
import csv, sys
from pathlib import Path
b = Path(sys.argv[1])
for run in ("nodp", "dp_sigma1.1"):
    rows = list(csv.DictReader((b / run / "metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 2, (run, rows)
    assert all(r["mae_kmh"] for r in rows), (run, rows)
    clients = list(csv.DictReader((b / run / "clients.csv").open(encoding="utf-8")))
    assert len(clients) == 2 * 3, (run, len(clients))
    assert (b / run / "final_model.pt").exists()
eps = [float(r["epsilon_max"]) for r in csv.DictReader((b / "dp_sigma1.1" / "metrics.csv").open())]
assert eps[1] > eps[0] > 0, f"ε 應隨輪次遞增：{eps}"
assert (b / "comparison.png").exists() and (b / "summary.csv").exists()
print(f"SMOKE TEST OK  ε 逐輪：{eps}  輸出：{b}")
EOF
