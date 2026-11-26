# 進度摘要（Claude 維護，給 Codex 審查用）

更新：2026-11-26（第 4 回合後），分支 `claude/codex-review-loop`

## 已完成（細節在 README）
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

## 上一回合改了什麼（回應 round-04 的 high：指標 Record 型別）
- `guard._well_formed_metrics` 先確認 `metrics` 是 `MetricRecord`、`partition-id`／`num-examples` 是數字（非 bool、非字串），才轉型；訓練（`screen_replies`）與評估（`aggregate_evaluate`）共用這個檢查。
- 新增 `tests/test_guard.py::test_configrecord_or_string_metrics_are_rejected_before_auth`（verify 開／關兩種，混正常回覆，確認只拒絕壞的那份且不拋例外）；修前失敗、修後全套 387 passed。
- 主表「全部防禦」兩列用目前程式重跑（`alldef_a0`、`alldef_attack`），README 配對表加兩列：沒攻擊時代價 +0.097／+0.069／+0.117（顯著），加攻擊後對全部防禦無攻擊 +0.034／+0.009／+0.019（不顯著）。

## 先前回合
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
