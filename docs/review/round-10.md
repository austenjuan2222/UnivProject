本回合 high: 1, medium: 1, low: 0

### [high] 斷線後僅剩的合法更新可用零測試筆數中斷聚合

**證據**：`guard.py:125–132` 允許 `num-test=0`；`results_log.py:31–33` 計算本地 MAE 時直接除以測試筆數總和，沒有處理零分母。`server_app.py:152–155` 將此紀錄函式接入訓練聚合回呼。

**為什麼是問題**：若其他 VD 斷線，只剩一台持合法金鑰的 VD 回覆，它可以保留正常欄位與模型，將 `num-test` 改成 0 後重新簽章。該回覆能通過整數檢查及欄位集合篩選，最後因除以零觸發 `ZeroDivisionError`，中斷聚合。這個情境需要其他 VD 未留下有效回覆，並非單台惡意 VD 在全員正常參與時就能觸發。

**建議怎麼驗證或修**：新增 `test_signed_zero_num_test_with_other_clients_dropped_does_not_abort_train`，串接真實 FedAvg 與 RunLogger 回呼，只保留一份合法簽章、`num-test=0` 的更新。以 `RunLogger.__new__` 建立記憶體物件並替換 `_flush`，避免測試寫檔。加權函式應處理總權重為零；若允許空測試集，回報缺值並繼續聚合。

### [medium] 主表結論仍將不顯著寫成沒有準確度代價

**證據**：

- `docs/EXPERIMENTS.md:469`：HMAC＋模型簽章對照的 p = 0.54。
- `docs/EXPERIMENTS.md:475–476`：無攻擊中位數與截尾平均的平均差分別為 +0.005、+0.002，95% CI 分別為 [−0.018, +0.028]、[−0.014, +0.018]。
- `docs/EXPERIMENTS.md:516、518` 仍寫「加上去不影響準確度」「沒有攻擊時幾乎沒有代價」。

**為什麼是問題**：配對 t 檢定未顯著只能支持目前資料無法辨識差異，不能證明等價。這些結論也與同文件第 465 行「不顯著不表示一樣」的規則不一致。

**建議怎麼驗證或修**：改報平均差、95% CI 與「本次未達顯著」。若要保留「幾乎沒有代價」，須先訂定可接受的 MAE 差異，再做等價檢定；不能只使用現有差異檢定的 p 值。

### 上回合追蹤

最新 `round-10.md` 為空檔，因此追蹤最近有內容的 `round-09.md`：

- **合法簽章的 NaN 整數型指標：靜態確認已修正。** `guard.py:125–132` 已驗證有限、非負整數與上限；測試包含 `test_signed_nan_dp_steps_does_not_abort_train`、`test_signed_infinite_diverged_does_not_abort_train`。本回合發現的是零權重的另一條例外路徑。
- **主表產生器的舊顯著性規則：已修正。** `scripts/paired_main32.py` 已使用 `paired_summary()`，並有 `test_main32_significance_uses_paired_t`。文件已更新原始數據的 p 值及斷線結論；等價推論措辭仍有上述殘留。

指定 venv 的 Python 無法啟動，未執行動態測試。本回合未修改檔案，也未執行會寫檔的指令。