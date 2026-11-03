# FL LIVE 即時儀表板設計（2026-11-03）

## 目的
自己除錯、觀察聯邦學習訓練；視覺上用「一條條線跑進度」的線束風格。同時監看在本專案使用 Claude Code 的 token 與工具呼叫。

## 架構
Flower 模擬時 client 在獨立 Ray 行程 → 統一附加寫入 `results/live/events.jsonl`（`dashboard_events.Emitter`，每事件一次 os.write）
→ `dashboard/server.py` tail（標準函式庫）→ Server-Sent Events → `dashboard/index.html`（Canvas 2D）。
Claude Code：tail `~/.claude/projects/<專案>/*.jsonl`，以 message.id 去重 usage。監控零 token。

## 事件
run_start / round_start / train_start / progress（節流約 20 次/epoch）/ train_end / dropout / eval /
security / round_train / round_eval / run_end；cc / cc_error。

## 不變式
- `dashboard-events = ""`（預設）時不寫任何東西，原實驗結果不變。
- 寫事件失敗不得中斷訓練。
- 每個 run 結束把該 run 的事件存成 `<run_dir>/events.jsonl`，可 `--replay`。

## 驗收
1. `server.py` + `run_experiments.py --dashboard` 即時看到各 VD batch 進度、每輪 MAE/ε。
2. `--replay <run_dir>` 可重播。
3. Claude Code 活動即時顯示在下半部。
4. 全部 pytest 通過（新增 tests/test_dashboard_events.py）。
