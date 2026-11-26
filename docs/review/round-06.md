本回合 high: 0, medium: 1, low: 0

### [medium] int8 推論的結論仍把不顯著差異寫成確定代價

**證據**：`README.md:1574`：「配對差 +0.008～+0.014 km/h（約 0.5%），小於 seed 間差距（0.02），依規則算不顯著，但三個 seed 方向一致，應該當成真的小代價。」

**為什麼是問題**：依專案規則，差值小於 seed 間差距應寫「不顯著」。三個 seed 方向一致可以描述為趨勢，但不足以支持「真的小代價」；這句把尚未確認的差異寫成確定結論。

**建議怎麼驗證或修**：改成「三個 seed 都略為變差，但依本專案規則不顯著，目前僅觀察到小幅增加的趨勢」。若要確認代價，固定相同模型與測試樣本，增加配對 seed，逐一報告 float32／int8 的 MAE 與差值，並追溯至本機 `summary.csv`。

### 上回合追蹤

最新 `docs/review/round-06.md` 為空檔，因此追蹤最近有內容的 `round-05.md`：

- **安全聚合查詢回覆的非有限編號造成未捕捉例外：已解決（靜態確認）。** `guard.py:451、674、724、746、757` 的五處解析均已捕捉 `OverflowError`。`tests/test_ephemeral.py:203、222` 已新增公告與一致性測試；`tests/test_guard_recover.py:40–42` 已涵蓋復原回覆的非有限 `pid`／`dropped`。但尚未看到針對 `share-owners`／`eshare-owners` 非有限值的對應測試，建議補入 `test_nonfinite_recovery_ids_fail_closed`。

本回合未修改檔案，也未執行會寫檔的指令。venv 的 Python 無法啟動，因此未動態驗證測試結果。