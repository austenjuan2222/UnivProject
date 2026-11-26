本回合 high: 1, medium: 0, low: 0

### [high] 安全聚合查詢回覆的非有限編號可造成未捕捉例外

**證據**：

- `guard.py:449`：公告回合直接執行 `int(c["pid"])`，但 `guard.py:451` 的例外處理未包含 `OverflowError`。
- `guard.py:669`、`guard.py:674`：一致性回合有相同問題。
- `guard.py:723`、`guard.py:724`：復原回合也直接轉換 `pid` 與 `dropped`，未捕捉 `OverflowError`。
- `tests/test_ephemeral.py:188–241` 涵蓋非二進位公告、長度異常及錯誤 MAC 長度，但未涵蓋非有限編號。

**為什麼是問題**：

`ConfigRecord` 可容納浮點數。節點把回覆的 `pid` 設成 `float("inf")`，即可使 `int()` 拋出 `OverflowError`，中斷安全聚合流程。公告回合只要其餘欄位通過長度檢查，就會在金鑰資格與節點綁定檢查之前觸發；不需要偽造有效 MAC。這使格式異常回覆能破壞整輪可用性。

本項為靜態確認，未執行動態重現。

**建議怎麼驗證或修**：

先新增以下測試，混入正常節點回覆，確認異常回覆不會讓流程拋例外：

- `test_nonfinite_advertise_pid_is_ignored`
- `test_nonfinite_consistency_pid_is_ignored`
- `test_nonfinite_recovery_ids_fail_closed`

涵蓋 `inf`、`-inf`、`NaN`，以及復原回覆的 `dropped`、`share-owners`、`eshare-owners`。轉型前統一檢查編號為有限整數、排除 bool 並限制於合法 VD 範圍；例外處理補上 `OverflowError`。

### 上回合追蹤

最新 `round-05.md` 為空檔，因此追蹤最近有內容的 `round-04.md`：

- **指標 Record 型別造成驗簽前例外：已解決（靜態確認）。** `guard.py:105–118` 已檢查 `MetricRecord` 與數值型別；訓練與評估共用此檢查。`tests/test_guard.py:31` 已新增混合正常／異常回覆、驗證開／關的測試。
- **主表「全部防禦」兩列的重跑與配對證據：已補齊。** 本機兩組巢狀目錄的 `summary.csv` 均包含 seed 1、2、3；攻擊組減無攻擊組約為 `+0.034／+0.009／+0.019`，符合 README:610–611，且已標示不顯著。

本回合未修改任何檔案，也未執行會寫檔的測試或實驗。