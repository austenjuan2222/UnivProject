# 進度摘要（Claude 維護，給 Codex 審查用）

更新：2026-11-26（第 7 回合後，合併進 `claude/lecture-optimize-loop`）。之後請審查這個分支：它包含 main 上的第 61～72 項與 README 改版。

## 已完成（摘要在 README 第 5 節，細節在 docs/EXPERIMENTS.md）
- 真實資料：國道 1 號 VD，8／32／256 個 VD，14～28 天；5 分鐘聚合。
- DP：record-level DP-SGD（Opacus），ε 自算並與 Opacus accountant 互驗；選用 client-level DP。
- 完整性與鑑別：每 VD 一把 HMAC-SHA256 金鑰、伺服器 Ed25519 簽全域模型、綁執行編號與輪次防重送；選用 VD 端 Ed25519（不可否認）。
- 金鑰生命週期：撤銷、輪替、有效期限、伺服器金鑰輪替（第 55～58 項）。
- TLS＋SuperNode 認證（對應 RFID 偽造標籤）：`scripts/run_tls_deployment.py`。
- 安全聚合：X25519 成對遮罩、斷線復原、ChaCha20-Poly1305 保護種子。
- 攻擊與防禦：放大、隱藏壅塞、串通；中位數、截尾平均、norm-cap、驗證檢查、串通偵測、Krum／Multi-Krum／Bulyan。
- 輸入驗證：npy 標頭、NaN／inf、筆數上限（第 53、54 項）。
- Edge 與 LPWAN：4 位元量化上傳、int8 推論、LoRaWAN 空中時間、下行群播＋FEC。
- 可靠度：10 seed、時間區塊交叉驗證、按天 bootstrap（`scripts/reliability/`，2026-11-21）。

## round-07（Multi-Krum 斷線後全選）
- `robust_agg._AllMetrics.aggregate_train` 每輪用有效回覆數重新檢查前提：少於 2f+3（Bulyan 4f+3）回 `(None, None)` 跳過該輪；Multi-Krum 選取份數改成 `min(m 或 本輪回覆數−f, 本輪回覆數−f)`。
- 測試：`test_multikrum_dropout_does_not_select_every_reply`（n=10、f=2，收到 6 誠實＋2 惡意）、`test_krum_skips_round_below_byzantine_threshold`（krum／multikrum／bulyan）；修前 3 個失敗，修後全套 392 passed。
- README Krum 表的設定每輪全員參與，修正不影響數字，已加註。

## 上一回合改了什麼（回應 round-09，對 claude/lecture-optimize-loop）
- high 接受：`_well_formed_metrics` 要求 round／dp-steps／diverged／ft-diverged／num-test／num-jam 是有限的非負整數值（≤ 2^31），合法簽章的 NaN、inf、0.5、−1 都被拒絕，不再讓紀錄程式的 `int()` 中斷整輪；`results_log` 的完整性檢查也防同類值。測試 `test_signed_nan_dp_steps_does_not_abort_train`、`test_signed_infinite_diverged_does_not_abort_train`（接真的 RunLogger 與 FedAvg），修前 ValueError／OverflowError。
- 安全檢查的補充：欄位集合同票時不再整輪跳過，改選上一輪被接受的欄位集合；沒有上一輪可依才跳過（`test_tie_breaks_toward_key_set_of_last_accepted_round`）。全套 434 passed、2 skipped。
- medium 接受：`scripts/paired_main32.py` 改用 `seed_stats.paired_summary`（配對 t，p < 0.05），輸出平均差、95% CI 與 p（`tests/test_paired_main32.py`）。docs/EXPERIMENTS.md 主表兩張表用原始 summary.csv 重產；10 台裁剪＋中位數對只裁剪 p = 0.046（舊寫的 0.049 由 3 位小數估）；「疊加最好」撤回（對只中位數 p = 0.20）；斷線改寫成 +0.011 [−0.016, +0.037]、p = 0.23，不再說「幾乎一樣」。

## round-08
- high 接受：`guard._majority_metric_keys` 在交給 Flower 前只留指標欄位集合和多數相同的回覆，最多的集合同票時整輪不聚合；訓練與評估都套用。沒有照建議依伺服器設定列出預期欄位：欄位隨 DP、安全聚合、微調、攻擊模擬而變，列表容易漏，多數決在被入侵 VD 不過半時足夠（過半時聚合本來就守不住），程式裡已註明。
- 同時修舊分支提過的列表型指標：`_well_formed_metrics` 要求 MetricRecord 每個值都是純量數字（NaN 照留）。
- 測試：`InconsistentMetricKeysTest`（多欄位的訓練回覆、少欄位的評估回覆、同票跳過，接 Flower 真的 FedAvg）、列表型 train-loss／mae-kmh／mse-kmh；修前 5 個失敗（InconsistentMessageReplies、列表被聚合），修後全套 431 passed。`test_two_huge_clipped_counts_do_not_overflow_the_sum` 的誠實回覆補上 secagg-clipped 欄位，仍測到溢位那一關。
- medium 接受：int8 推論改判為顯著（+0.0108，p = 0.026），README 摘要也補上。

## 合併時的判斷（2026-11-26）
- round-01～07 的程式修正全部接受，合進包含 main 新進度的分支；`guard._well_formed_metrics` 合併時同時保留 MetricRecord／數字型別檢查與 client-level DP 的 `unit-weight` 必須等於 1。全套 426 passed、2 skipped。
- 主表重跑的「顯著？」判斷改用配對 t 檢定（第 72 項的規則）：5 處改判為顯著，含 HMAC 擋竄改後的 +0.010 小代價（p = 0.023），寫在 docs/EXPERIMENTS.md 主表下方。
- README 已拆成主線＋附錄，審查時引用行號請用 docs/EXPERIMENTS.md。

## 先前回合
- round-06：int8 推論結論改成不顯著趨勢；補復原 owners 非有限值測試。
- round-05：安全聚合公告／一致性／復原五處解析補 `OverflowError`（inf 編號）。
- round-04：`_well_formed_metrics` 要求 MetricRecord 與數字型別；全部防禦兩列重跑並補配對差值。
- round-03 後：主表 24 組×3 seed 重跑，配對差值表，撤回兩項不顯著結論（`scripts/paired_main32.py`）。
- round-03：安全聚合指標加總溢位（0～2^53、整數加總）。
- round-02：安全聚合指標型別檢查（NaN／列表）；移除無據顯著性判斷；集中式調參紀錄與 summary.csv。
- round-01：非 ASCII HMAC 標籤（bytes 比對＋2 個測試）；Krum／集中式結果複製到本機並重算；主表措辭限定。

## 已知還沒做
- 連續多輪斷線、固定一群 VD 離線的可用性實驗。

## 想請審查的重點
- README「攻擊與可用性實驗」各表是否都有 3 seed 與同 seed 配對差值，哪些結論其實不顯著。
- 課程主題中證據最薄的一項。
- `guard.py`、`secagg.py`、`security.py` 有沒有測試沒涵蓋的漏洞。
