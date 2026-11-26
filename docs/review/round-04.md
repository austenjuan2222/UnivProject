本回合 high: 1, medium: 0, low: 0

### [high] 指標未驗證 Record 型別，可在驗簽前造成伺服器例外

**證據**：`guard.py:105–112` 將 `partition-id` 轉成 `float` 驗證；`guard.py:128–138` 沒有確認 `metrics` 是 `MetricRecord`。因此，送入 `ConfigRecord({"partition-id": "1.0", "num-examples": 1})` 時，指標檢查會通過，但 `guard.py:155` 的 `int("1.0")` 會拋出 `ValueError`。此處早於 `guard.py:168–170` 的 HMAC 驗證。現有 `tests/test_guard*.py` 未見此種錯誤 Record 型別測試。

**為什麼是問題**：能送達應用層的節點不需要有效的更新簽章，就能利用一份格式異常的回覆中斷聚合，破壞可用性。`_has_records()` 只驗證紀錄名稱，無法阻擋這條路徑。

**建議怎麼驗證或修**：

- 新增 `test_configrecord_metrics_are_rejected_before_auth`：搭配合法浮點陣列、上述 `ConfigRecord`、缺少簽章及 `verify=True`，確認回覆被拒絕且不拋例外。
- 新增 `test_malformed_metrics_do_not_abort_other_replies`：混入正常回覆，確認其他回覆仍可處理。
- 在轉換任何指標前確認 `metrics` 為 `MetricRecord`，並直接限制編號與樣本數的數值型別；訓練與評估入口都應使用同一套檢查。

本項為靜態審查。嘗試以指定 venv 的 Python 加上 `-B` 執行無寫檔重現，但其指向的 Python311 無法啟動，因此尚未動態驗證。

### 上回合追蹤

最新 `round-04.md` 是空檔，以下追蹤最近有內容的 `round-03.md`：

- **安全聚合指標加總溢位：已解決（靜態確認）。** `guard.py:119–125` 已限制指標為 `0～2^53` 的有限整數，`:619` 改用 Python 整數加總；新增 `test_two_huge_clipped_counts_do_not_overflow_the_sum`。
- **攻擊主表配對證據：主要問題已解決。** 本機 `results/night26/main32/` 的 24 組摘要皆有 seed 1、2、3；README 已補配對差值與波動範圍，並撤回裁剪較穩及疊加最佳的結論。兩列「全部防禦」仍沿用舊批次，尚未納入此次同版重跑與配對表，不能視為整張主表已全面補齊。