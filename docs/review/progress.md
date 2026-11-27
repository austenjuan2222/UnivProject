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

## round-13（抓取長度、連線資源、篩選空狀態）
- `predict_all` 抓取分鐘數改成 `lookback*5 + 10`（lookback=12 仍是 70）：`test_predict_all_fetch_window_scales_with_lookback`（修前要求 0 分鐘、lookback=24 拿不到預測）。
- `forecast_web.BoundedServer`：同時最多 64 個連線，滿了直接關掉新連線不開執行緒；Handler 連線逾時 10 秒。`make_server()` 給 main 與測試共用。測試：`test_incomplete_headers_connection_times_out`、`test_concurrent_connections_are_bounded`（只在本機 socket，沒有對公開通道壓測）。
- 網頁「可能變慢」篩選結果為空時，過期顯示「預測已過期，目前無法判斷…」、全部資料不足顯示「…無法判斷…」，只有真的有預測時才說「沒有預計會變慢」。瀏覽器驗證三種情況。

## round-12（即時預測：horizon、標準化切點、網頁過期）
- `forecast_live.load_model` 遇到 `horizon != 1` 直接拒絕（遞迴預測只對單步模型成立）：`test_live_rejects_non_unit_horizon`。
- 新增 `data.read_vd_csv`（補完整 5 分鐘時間軸），訓練端 `load_client_arrays` 與即時端 `vd_stats` 共用，切點與 mean／std 一致：`test_vd_stats_matches_training_with_missing_timestamps`（中間缺一天，修前 mean 90.49 vs 89.66）。
- 網頁 `expired()` 用真正的現在時間判斷過期；跨過 20 分鐘時 `tick()` 重新呈現，拿掉所有預測格、欄頭、摘要結論與「可能變慢」篩選。沒有 JS 測試環境，改在瀏覽器驗證：同一份回應、時鐘 +40 分 → 0 個預測格、32 個「資料太舊」、摘要「預測已過期」；時鐘回來後恢復 96 格。
- 全套 449 passed。

## round-07（Multi-Krum 斷線後全選）
- `robust_agg._AllMetrics.aggregate_train` 每輪用有效回覆數重新檢查前提：少於 2f+3（Bulyan 4f+3）回 `(None, None)` 跳過該輪；Multi-Krum 選取份數改成 `min(m 或 本輪回覆數−f, 本輪回覆數−f)`。
- 測試：`test_multikrum_dropout_does_not_select_every_reply`（n=10、f=2，收到 6 誠實＋2 惡意）、`test_krum_skips_round_below_byzantine_threshold`（krum／multikrum／bulyan）；修前 3 個失敗，修後全套 392 passed。
- README Krum 表的設定每輪全員參與，修正不影響數字，已加註。

## 上一回合改了什麼（回應 round-11）
- medium 全部接受（即時預測）：
  - 回測不再用未來資料補輸入：輸入只取預測時間點前的列再補值，實際值另外算（`test_backtest_future_rows_do_not_change_forecast_inputs`，修前 07:05 缺桶被 07:10 補齊而給出預測）。
  - 尾端過期：所有 VD 對齊到同一條時間軸，結束點是 `min(預測時間點, 來源最後一分鐘)` 的整 5 分鐘；資料比預測時間點舊超過 20 分鐘就不預測並回報 `stale`、`data_until`、`data_age_min`（`test_stale_tail_is_not_reported_as_fresh_forecast`）。來源晚上傳時沒收齊的最後一桶不用（`test_source_lag_is_not_stale_and_partial_bucket_is_dropped`）。
  - 標示：每個預測回傳完整目標時刻 `at` 與相對預測時間點的 `ahead_min`（`test_forecast_labels_match_target_timestamps`，含跨午夜）；網頁欄位改標實際時刻與相對「真正現在」的「現在這段／幾分後／幾分前」，手機也顯示。
  - 摘要分開算可預測、資料不足與可能變慢：「可預測的 X 處中……；另 Y 處資料不足」；完全沒有預測時不給趨勢結論；資料過期時說資料停在幾點。這是前端邏輯，用瀏覽器灌假資料檢查三種情況，沒有寫 Python 測試。
- low 接受：信賴區間上界改寫成「平均配對差的 95% 信賴區間上界」，不再說成確定的代價上限。
- 另外：網頁每個來源每分鐘最多 120 個請求（經 Cloudflare 通道時看 `CF-Connecting-IP`，直接連線不信任這個標頭；`test_rate_limit_per_client_and_window`），加 `frame-ancestors 'none'`、`Referrer-Policy`。

## 2026-11-27（Austen 要求：檢查嚴謹度與樣本數，再往實際使用走）
- 主表 26 組補到 10 個 seed（`rerun_main32.ps1 -Seeds 4,...,10`；`-Seeds` 改成字串再切，避免 PowerShell 把 "4,5,6" 轉成整數 456）；`paired_main32.py` 可合併多個結果資料夾。
- 整張主表 34 個檢定做 Holm 校正，「顯著？」改看校正後 p，未校正才 < 0.05 的另外標「否（未校正 p < 0.05）」。判斷改變的列與結論改寫在 EXPERIMENTS「2026-11-27 補到 10 個 seed」。
  重點：竄改有 HMAC 的代價不再顯著（+0.003 [−0.002, +0.009]）；竄改沒驗證、雜訊 FedAvg 變顯著；裁剪＋中位數的優勢校正後不顯著。
- 新 VD 測試（8 折留 4 台不訓練）：+0.0095 [+0.0035, +0.0155]，p = 0.021；26／32 台仍贏 naive。
- 即時預測：`scripts/forecast_live.py`（抓高公局即時檔、遞迴預測 5／10／15 分鐘、可回測）、`scripts/forecast_web.py`（標準函式庫 HTTP 伺服器、預設只聽本機、CSP、只用 textContent）。

## 上一回合改了什麼（回應 round-10，對 claude/lecture-optimize-loop）
- high 接受：`results_log.aggregate_train` 測試筆數總和為 0 時本地 MAE 記 NaN，不再除以零（`test_zero_num_test_does_not_divide_by_zero`，修前 ZeroDivisionError）。直接測根本原因的函式，沒有另外接整條 FedAvg。
- 安全檢查的補充：同票時參考的上一輪集合只在某集合過半時才更新，攻擊者靠「最多但沒過半」不能改掉它（`test_plurality_without_majority_does_not_set_last`）。沒有改成「聚合完成後才更新」：後面的過濾都在同一個函式裡，全被擋時這輪本來就不聚合。
- medium 接受：HMAC、無攻擊中位數／截尾平均的「不影響準確度」「幾乎沒有代價」改報平均差、95% CI 與「本次未達顯著」。全套 436 passed、2 skipped。

## round-09
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
