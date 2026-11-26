本回合 high: 1, medium: 2, low: 0

### [high] 非 ASCII 的 HMAC 標籤可讓驗證拋出例外

**證據**：`security.py:343` 直接呼叫 `hmac.compare_digest(..., str(tag))`；字串包含非 ASCII 字元時會拋出 `TypeError`。`guard.py:77–82` 將收到的標籤轉成字串，未限制字元；`guard.py:157` 與 `guard.py:102` 均未捕捉此例外。現有 `tests/test_security.py:124–125` 只測試 ASCII 垃圾標籤 `"zz-not-hex"`。

**為什麼是問題**：預設 HMAC 模式下，只要送出格式正常的陣列與指標、將標籤改成 `"錯"`，就可能中斷訓練或評估聚合。這發生在驗證成功之前，不需要知道合法金鑰。

**建議怎麼驗證或修**：新增 `test_non_ascii_hmac_tag_is_rejected_not_raised`，以及訓練、評估各一個整合測試，確認惡意回覆被拒絕、其他回覆仍可聚合。驗證前嚴格檢查標籤為 64 個十六進位字元，格式錯誤直接回傳 `False`。

本次未能執行重現：指定 venv 指向的 Python 3.11 執行檔不存在；上述判斷來自程式碼檢查。

### [medium] 新增比較實驗的數字無法追到本機結果

**證據**：`README.md:1774–1775` 將 Krum 實驗指向 `/mnt/project-files/reports/results/krum32/`；`README.md:1810` 將集中式比較指向 `/mnt/project-files/reports/results/central32/`。本機工作副本 `results/` 與指定執行目錄 `UnivProject-run/results/` 的 `summary.csv` 清單中，未找到這兩批結果。

**為什麼是問題**：無法核對三個 seed、配對差值、同版對照與執行設定，因而無法確認「Multi-Krum 是目前最好的單一防禦」及「聯邦式＋本地微調最好」的證據鏈。這也未符合 AGENTS.md 要求數字能追到本機 `results/**/summary.csv`。

**建議怎麼驗證或修**：將原始結果移交至本機忽略追蹤的 `results/krum32/`、`results/central32/`，附上各 run 的設定、程式版本與 seed。集中式比較目前引用 JSONL，應另產生逐 seed 的 `summary.csv`，再核對 README 表格及配對差值。原始資料無法取得時，重新跑三個 seed 與同版對照。

### [medium] 攻擊與可用性主表仍缺配對證據，部分結論過度確定

**證據**：`README.md:561–585` 多數列只列三 seed 平均，沒有逐 seed 配對差值；`README.md:587–588` 的「與沒有安全檢查配對比較的差值都在 ±0.011 以內」未限定適用列，字面上與表內攻擊及裁剪組的大幅差距矛盾。`README.md:605` 宣稱「斷線不影響可用性」，僅以隨機 30% 斷線實驗支持。另 `README.md:602` 將 14/32 攻擊下輸給 naive 解釋為「穩健聚合的理論極限」。

**為什麼是問題**：平均值不足以判斷差異是否小於 seed 間差距；單一隨機斷線設定不能支持一般性的可用性結論。特定資料上 MAE 超過 naive，也不能證明達到理論容錯極限。

**建議怎麼驗證或修**：逐列補上相對同版、同聚合器無攻擊對照的三個配對差值，並標示 seed 間差距；小於該差距的結果明寫「不顯著」。限定 ±0.011 敘述的適用組別。將斷線結論限定為本次隨機斷線設定；將 14/32 結論改成實驗觀察。若要擴大可用性結論，補跑連續多輪斷線及固定一群 VD 離線，各三個 seed。

### 上回合追蹤

最近的 `docs/review/round-01.md` 為空檔，沒有可追蹤問題；`progress.md` 亦記載這是第一回合。本次未修改任何檔案。