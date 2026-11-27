本回合 high: 1, medium: 1, low: 0

### [high] 合法簽章的 NaN 整數型指標仍可中斷伺服器

**證據**：`guard.py:118–125` 允許其他指標為純量 NaN，只額外限制編號、樣本數與 `unit-weight`。`results_log.py:149、152` 卻直接對 `dp-steps`、`diverged` 呼叫 `int()`；`results_log.py:133` 沒有檢查這些值。`server_app.py:152–155` 將紀錄函式直接接入聚合回呼。

**為什麼是問題**：一台持合法金鑰的 VD，只需保留正常欄位集合，把原有 `dp-steps` 改成 NaN 後重新簽章，即可通過驗簽與欄位多數決，最後在 `int(float("nan"))` 觸發 `ValueError`。因此上一回合修正欄位不一致後，仍有單台 VD 中斷聚合的途徑。現有測試未找到此案例。

**建議怎麼驗證或修**：新增 `test_signed_nan_dp_steps_does_not_abort_train`、`test_signed_infinite_diverged_does_not_abort_train`，串接實際 `GuardedStrategy`、FedAvg 與紀錄回呼，要求拒絕異常回覆並完成誠實回覆的聚合。測試可替換 `_flush` 避免寫檔。所有會轉成整數的指標，應先驗證有限、整數及合理範圍；誤差指標可另行保留 NaN。

### [medium] 主表產生器仍使用舊顯著性規則，結論也保留等價推論

**證據**：

- `scripts/paired_main32.py:35–40` 仍以「配對差同號且皆大於 seed 間差距」決定是否顯著。
- `docs/EXPERIMENTS.md:462–484` 指定用此腳本產生主表，表內仍保留舊判斷。
- `docs/EXPERIMENTS.md:499–505` 另補五項改判，其中 p = 0.049 的結果仍僅由三位小數計算，寫成「暫時算顯著」。
- `docs/EXPERIMENTS.md:521` 仍因差值小於 seed 間差距，稱隨機斷線結果與全員參與「幾乎一樣」。

**為什麼是問題**：重新產表會再次輸出已知錯誤的判斷，表格與補充文字也互相衝突。接近 0.05 的結果不能先用四捨五入數字確定顯著；未顯著也不足以支持等價或沒有代價。

**建議怎麼驗證或修**：將產生器改用 `scripts/seed_stats.py` 的 `paired_summary()`，由原始 `summary.csv` 輸出配對平均差、95% CI 與 p 值。新增 `test_main32_significance_uses_paired_t`，以 `[0.013, 0.008, 0.009]` 等資料確認不再使用舊規則。優先重算 `clipmed_a10` 對 `clip_a10`；斷線結論改報差值與 CI，若要聲稱等價，須預先訂定可接受差異並另做等價驗證。

### 上回合追蹤

最新 `round-09.md` 為空檔，故追蹤最近有內容的 `round-08.md`：

- **合法簽章的指標欄位不一致：靜態確認已處理。** `guard.py:128–138、664、948` 已加入欄位集合篩選；`tests/test_malformed_update.py` 的 `InconsistentMetricKeysTest` 涵蓋訓練多欄位、評估少欄位及同票情況。本回合另發現上述指標值驗證缺口。
- **int8 顯著性判斷錯誤：文件已修正。** `docs/EXPERIMENTS.md:1488` 與 README 已標示平均代價 +0.0108、p = 0.026，為小但顯著的代價。

指定 venv 的 Python 無法啟動，未執行動態測試；本回合未修改檔案，也未執行會寫檔的指令。