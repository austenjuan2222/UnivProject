# 共同工作指引（Claude 與 Codex 共用）

本檔是 Claude Code 與 OpenAI Codex 共用的專案規則；`CLAUDE.md` 只匯入本檔，兩邊讀到的是同一份。

## 專案
智慧交通聯邦學習（Flower）＋差分隱私（Opacus）＋ IoT 安全機制。結果與結論寫在 `README.md`，
課程對照在「依課程講義補強」一節。

## 環境（Windows 桌機）
- Git 工作副本：`C:\Users\TESTER\Documents\UnivProject-git`
- venv 與資料：`C:\Users\TESTER\Documents\UnivProject-run`（`.venv\Scripts\python.exe`，資料 `data/processed_28d_32`、`_256`）
- 一律設 `PYTHONUTF8=1`（避免 cp950 讀錯 UTF-8 檔）
- `results/`、`data/`、`keys/` 不進 git；數字要能追到本機 `results/**/summary.csv`

## 實驗與結論的規矩
- 每個實驗跑 3 個 seed，並用同一版程式補無攻擊／無防禦對照組。
- 報「同 seed 配對差值」；差值小於 seed 間差距就寫「不顯著」。
- 改程式先寫測試：`python -m pytest`。

## 分工與審查循環（見 `docs/review/README.md`）
- Claude：改程式、跑實驗、更新 `docs/review/progress.md`、依審查回饋修正。
- Codex：唯讀審查，回饋寫在 `docs/review/round-NN.md`，不改任何檔案。
- 只 commit 到工作分支，不推 `main`。
- `docs/review/STOP` 存在時循環立即停止。
