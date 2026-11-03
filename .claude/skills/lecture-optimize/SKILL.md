---
name: lecture-optimize
description: 依 Austen 給的課程簡報（IoT 架構、IoT 安全、RFID、TLS／MAC）持續優化這個聯邦學習＋差分隱私專案，直到 Austen 說「停止」。當 Austen 說「持續優化」「依簡報優化」「繼續做到我說停止」時使用。
---

# 依課程簡報持續優化

Austen 的原話：

> 你現在可以用我的任何skill了，在針對我給你的簡報持續優化然後過程中有要調用甚麼skill或工具自己安排，或者github上找，確認安全再套用，我回來說停止在停止

## 怎麼做

1. **找缺口**：把簡報的主題（五個問題、CIA、TLS、MAC／簽章、RFID 偽造標籤、Edge 資源、LPWAN、攻擊類型）對照 README 的「依課程講義補強」一節，挑一個還沒做或證據不足的項目。
2. **先查地圖**：專案根目錄若有 `graphify-out/graph.json`，用 `graphify query "<問題>"` 找相關程式，不要整個 repo 掃。程式改了之後用 `GRAPHIFY_NO_AUTO_REFRESH=1 graphify update .` 更新地圖（這個環境變數避免它自動改寫使用者層級的 graphify skill）。
3. **做一個可驗證的小單位**：改程式先寫測試；實驗一律跑 3 個 seed，並用同一版程式補無攻擊／無防禦對照組，報「同 seed 配對差值」，差值小於 seed 間差距就說不顯著。
4. **寫回結果**：表格與結論寫進 README 對應章節；進度報告在 `UnivProject-run/reports/`。數字要能追到 `results/` 底下的 summary.csv。
5. **commit 到工作分支並推上去**，不直接推 main。
6. **回到第 1 步**：推完不算完成，直接挑下一個缺口，不要結束回合、不要停下來問「要不要繼續」。只用一兩行回報這輪做了什麼就接著做。等背景實驗時，先做不依賴該結果的下一個項目；真的沒有可做的，才結束回合等實驗通知，收到後繼續。

## 工具與 skill

- 可以自己挑任何已安裝的 skill 或工具，開始用時一句話告訴 Austen 用了什麼、為什麼。
- 從 GitHub 找新的 skill 或套件時，先下載到暫存資料夾檢查：來源 repo、維護者、安裝時會不會執行程式（setup.py、.pth）、有沒有 eval／exec／對外連線。確認安全才裝，裝在使用者層級，不動專案 venv 以外的既有套件版本；若 pip 連帶升級了別的套件，要還原並說明。
- 更新使用者層級的 skill 或插件仍要 Austen 對具體報告同意。

## 環境

- Git 工作副本：`C:\Users\TESTER\Documents\UnivProject-git`；venv 與資料：`C:\Users\TESTER\Documents\UnivProject-run`（`.venv\Scripts\python.exe`，資料 `data/processed_28d_32`、`_256` 等）。
- 一律設 `PYTHONUTF8=1`（不然 Windows cp950 會讀錯 UTF-8 檔）。
- 長實驗放背景跑，不要中斷正在跑的實驗。

## 停止

只有 Austen 本人說「停止」才停；做完一輪、推完分支、清單看似做完，都不算停止的理由（清單做完就找證據不足的項目補強）。停之前：把進行中的實驗結果收完、commit、推分支，並回報做了什麼、還剩什麼。
