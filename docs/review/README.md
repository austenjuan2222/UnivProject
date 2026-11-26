# Codex 審查 ↔ Claude 修正循環

| 檔案 | 誰寫 | 內容 |
|---|---|---|
| `progress.md` | Claude | 目前進度、上一回合改了什麼、待審重點 |
| `codex_prompt.md` | 人 | 給 Codex 的審查指示（固定） |
| `round-NN.md` | Codex（`codex exec -o` 寫出） | 該回合的審查回饋 |
| `STOP` | 人或 Claude | 存在就停止，不再跑下一回合（不進 git） |

## 一回合
1. Claude 更新 `progress.md` 並 commit。
2. `powershell -ExecutionPolicy Bypass -File scripts/codex_review.ps1` → Codex 以 `--sandbox read-only` 讀 repo，回饋寫成 `round-NN.md`。
3. Claude 讀回饋，逐條判斷（接受／不接受並說明理由），改程式、補測試或實驗，結果寫回 `progress.md` 與 README。
4. commit 到工作分支並推上去，回到 1。

## 停止條件
- `STOP` 檔存在，或 Austen 說「停止」。
- 已達回合上限（`codex_review.ps1 -MaxRounds`，預設 10）。
- Codex 連續兩回合沒有 high／medium 等級的問題。
