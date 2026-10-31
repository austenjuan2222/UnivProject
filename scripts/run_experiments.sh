#!/usr/bin/env bash
# 一次跑「無 DP 對照組」+ 一個或多個 DP 設定，最後畫比較圖。
#
# 用法（在專案根目錄）：
#   bash scripts/run_experiments.sh
#   ROUNDS=30 SIGMAS="0.8 1.1 1.5" bash scripts/run_experiments.sh
#   DATA_DIR=/abs/path/processed EXTRA="local-epochs=2" bash scripts/run_experiments.sh
#
# 環境變數：
#   DATA_DIR    前處理輸出資料夾（預設 data/processed）
#   RESULTS_DIR 結果根目錄（預設 results）；本批次寫在 $RESULTS_DIR/<時間>/
#   ROUNDS      聯邦輪數（預設 20）
#   SIGMAS      DP 的 noise multiplier 清單（預設 "1.1"）
#   SEEDS       每個設定重複跑的 seed 清單，例如 "1 2 3"（預設不重複，用 pyproject 的 seed）
#   SKIP_NODP   設為 1 則不跑無 DP 對照組
#   NUM_CPUS    每個模擬 client 使用的 CPU 數（預設 1）
#   EXTRA       其他要覆寫的 run config，例如 "learning-rate=0.02 batch-size=128"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data/processed}" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
ROUNDS="${ROUNDS:-20}"
SIGMAS="${SIGMAS:-1.1}"
EXTRA="${EXTRA:-}"
BATCH_DIR="$RESULTS_DIR/$(date +%Y%m%d-%H%M%S)"

case "$DATA_DIR$BATCH_DIR" in
  *" "*) echo "路徑不能含空白：$DATA_DIR / $BATCH_DIR" >&2; exit 1 ;;
esac

# 專案專用的 Flower 設定檔：num-supernodes = manifest 中的 VD 數
export FLWR_HOME="$ROOT/.flwr_home"
N="$("$PY" "$ROOT/scripts/setup_flwr_home.py" --data-dir "$DATA_DIR" \
      --flwr-home "$FLWR_HOME" --num-cpus "${NUM_CPUS:-1}")"
echo "client 數 = $N（來自 $DATA_DIR/manifest.json）；結果寫到 $BATCH_DIR"
mkdir -p "$BATCH_DIR"

common="num-clients=$N num-server-rounds=$ROUNDS data-dir='$DATA_DIR' results-dir='$BATCH_DIR' $EXTRA"

read -r -a seeds <<< "${SEEDS:-}"
[[ ${#seeds[@]} -eq 0 ]] && seeds=("")
for seed in "${seeds[@]}"; do
  sfx=""; sopt=""
  [[ -n "$seed" ]] && sopt="seed=$seed"
  [[ ${#seeds[@]} -gt 1 ]] && sfx="_seed$seed"
  if [[ "${SKIP_NODP:-0}" != "1" ]]; then
    echo "=== 無 DP 對照組 $sopt ==="
    flwr run "$ROOT" local --run-config "$common use-dp=false $sopt run-name='nodp$sfx'"
  fi
  for s in $SIGMAS; do
    echo "=== DP σ=$s $sopt ==="
    flwr run "$ROOT" local --run-config "$common use-dp=true noise-multiplier=$s $sopt run-name='dp_sigma$s$sfx'"
  done
done

"$PY" "$ROOT/scripts/plot_results.py" "$BATCH_DIR"
