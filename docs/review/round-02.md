本回合 high: 1, medium: 2, low: 0

### [high] 安全聚合指標缺少型別與有限值檢查，可使聚合拋出例外

**證據**：`guard.py:105–112` 僅驗證 `partition-id` 與 `num-examples`；`security.py:312–317` 的簽章序列化接受數字列表及 NaN。通過驗簽後，`guard.py:602` 直接對 `secagg-clipped` 執行 `float()`、`int()`，`guard.py:608–612` 亦直接轉換其他安全聚合設定；秘密分享模式另於 `guard.py:585–586` 轉換 `secagg-self-tag`。

**為什麼是問題**：持有合法金鑰的被入侵 VD，可以在正常遮罩更新中附上 `secagg-clipped=float("nan")` 或列表並重新簽章。這些值可通過現有前置檢查，卻在後續轉型時觸發未捕捉的 `ValueError` 或 `TypeError`，中斷聚合。這屬於專案已納入威脅模型的合法端點攻擊。

**建議怎麼驗證或修**：新增 `test_nonfinite_secagg_metrics_are_rejected_without_crashing` 與 `test_list_secagg_metrics_are_rejected_without_crashing`，涵蓋上述欄位、一般安全聚合及秘密分享模式。使用有效簽章，確認異常回覆被拒絕，整輪依復原規則安全放棄或繼續。所有轉型前應檢查純量型別、有限值、整數性與允許範圍。

本項為靜態審查發現；指定 venv 無法啟動其指向的 Python，未能執行重現。

### [medium] 攻擊主表仍保留無法核對的配對與波動結論

**證據**：`README.md:561–585` 多數列只有平均值；`README.md:588` 明載原始 `results/night5/` 遺失，無法重算逐 seed 配對差值，卻仍宣稱相關差距「遠大於 seed 間差距」。`README.md:583、602` 仍稱裁剪加中位數「最好」，但新增比較表的 Multi-Krum 在 10 台被入侵時為 2.223（`README.md:1782`），低於其 2.371。

**為什麼是問題**：揭露資料遺失尚未補足證據，缺少逐 seed 結果仍無法核對波動判斷。「最好」也未限定比較範圍，與後文結果不一致。

**建議怎麼驗證或修**：以同一版程式重跑缺失情境及各自無攻擊對照，各用 seed 1、2、3；逐列列出配對差值與 seed 間差距。補齊之前，移除無法驗證的波動判斷，將「最好」限定為該表所比較的防禦組合。

### [medium] 集中式比較尚缺本機調參證據與規定的 summary.csv

**證據**：`README.md:1808–1810` 宣稱各方法以驗證集挑選學習率，但 `sweep.jsonl` 仍只在專案資料庫。本機 `results/central32/` 僅有 `central.jsonl`，沒有 AGENTS.md 要求的 `summary.csv`。

**為什麼是問題**：測試結果及三 seed 數字已可追溯，但尚無法在本機核對「分別使用驗證集選參」的過程，也未完成上回合要求的結果格式補強。

**建議怎麼驗證或修**：移交 `sweep.jsonl`，附上候選設定、驗證分數與選擇規則；將 `central.jsonl` 匯出為逐 seed 的 `summary.csv`，保留模式、學習率、微調設定、MAE、ε 及程式版本，重新核對 README 配對差值。

### 上回合追蹤

最新 `round-02.md` 是空檔，以下追蹤最近有內容的 `round-01.md`：

- **非 ASCII HMAC 標籤：程式碼層面已解決。** `security.py:343–344` 已改比對 bytes；新增單元測試及訓練／評估整合測試。本次未能重跑。
- **Krum／集中式結果追不到：部分解決。** 本機已有 Krum 的 23 組結果；重算 3、6、10 台入侵時的 Multi-Krum 相對中位數配對差值，與 README 一致。集中式測試結果已移交，調參紀錄與 `summary.csv` 仍未補齊。
- **主表配對證據與措辭：部分解決。** ±0.011 的適用範圍、隨機斷線結論及理論極限措辭已修正；缺失的原始結果、配對證據及部分比較結論仍待處理。

本次未修改任何檔案。