本回合 high: 1, medium: 1, low: 0

### [high] 合法簽章的指標欄位不一致，仍可中斷聚合

**證據**：`guard.py:108–122` 只檢查 `partition-id`、`num-examples`、`unit-weight` 的值，沒有驗證完整指標欄位集合；`guard.py:751`、`guard.py:931` 將通過的訊息直接交給內層策略。本機 Flower 的 `strategy_utils.py:228–235` 遇到各回覆的 MetricRecord 欄位不同時，會拋出 `InconsistentMessageReplies`。

**為什麼是問題**：一台持合法金鑰的被入侵 VD，只需在指標中新增 `extra=0` 並重新簽章，即可通過現有格式與驗簽檢查，卻讓整批誠實回覆無法聚合。訓練及評估路徑都有此缺口。現有測試涵蓋多出整份紀錄，未涵蓋同一份 MetricRecord 多出欄位。

**建議怎麼驗證或修**：新增 `test_signed_extra_metric_does_not_abort_train`、`test_signed_missing_metric_does_not_abort_evaluate`，以兩份正常回覆及一份合法簽章但指標欄位異常的回覆，呼叫實際 `GuardedStrategy`＋FedAvg，要求拒絕異常回覆並完成其餘聚合。依伺服器設定定義預期指標結構，在交給 Flower 前檢查；不要以第一份回覆決定結構。

### [medium] int8 推論仍沿用舊規則判定不顯著，與配對 t 檢定結果相反

**證據**：`docs/EXPERIMENTS.md:1483–1485` 的配對差為 `+0.0099、+0.0083、+0.0142`；`:1488` 因差值小於 seed 間差距而寫成「不顯著」。`:1941–1944` 已改採配對 t 檢定，但仍保留舊項目的判定。

依表列數字重算，平均差為 **+0.0108 km/h**，95% CI 約 **[+0.0032, +0.0184]**，雙尾 **p≈0.0256**。

**為什麼是問題**：依目前專案規則，這是小但顯著的準確度代價。不能用兩組各自的 seed 波動取代同 seed 差值的檢定；第 6 回合的文字修正也因此需要重新檢視。

**建議怎麼驗證或修**：從原始結果取得未四捨五入的三組數字，以 `scripts/seed_stats.py` 的 `paired_summary()` 重算並記錄結果，更新第 60 項附近的 int8 結論。其他仍用「小於 seed 間差距」判定的段落，也應逐項重算。

### 上回合追蹤

最新 `round-08.md` 為空檔，因此追蹤最近有內容的 `round-07.md`：

- **Multi-Krum 斷線後可能全選：靜態確認已修正。** `robust_agg.py:25–30` 已依本輪有效回覆數檢查門檻並限制選取數；`tests/test_robust_agg.py:47、52` 已補斷線與門檻測試。
- **動態驗證限制**：專案 venv 的 Python 無法啟動，本回合未執行測試；上述程式問題以原始碼及本機 Flower 實作交叉確認。

本回合未修改檔案，也未執行會寫檔的指令。