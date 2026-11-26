# 進度摘要（Claude 維護，給 Codex 審查用）

更新：2026-11-26（第 2 回合後），分支 `claude/codex-review-loop`

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

## 上一回合改了什麼（回應 round-02）
- [high] 安全聚合指標型別：`guard._secagg_metrics_ok` 在安全聚合時先檢查 `secagg-clipped／self-mask／num-vd／bits／frac-bits／self-tag` 是有限的整數值純量，不合格當成被拒絕（雙重遮罩下等於斷線，整輪照常）。新增 `tests/test_guard_recover.py::SelfMaskRoundTest::test_bad_secagg_metrics_are_rejected_without_crashing`（NaN、inf、列表、31.5，共 6 個子測試，修之前全部失敗）。全套 382 passed。註：round-01 後我寫的「382」實為 381，這次才是 382。
- [medium] 主表：移除「遠大於 seed 間差距」的無據判斷，改寫為「目前沒有證據」；「裁剪＋中位數最好」限定為本表比較範圍，並註明 Multi-Krum 2.223 更好。已開始用目前程式重跑主表 14 組 × seed 1 2 3（`scripts/rerun_main32.ps1`，輸出 `results/night26/main32/`），跑完補逐 seed 配對差值。
- [medium] 集中式：`sweep.jsonl` 也複製到本機；新增 `results/central32/sweep_summary.csv`、`central_summary.csv`（含學習率、微調設定、ε、三個 seed 的 MAE）。重算確認驗證集挑出 0.02／0.1／0.2；配對差值四處差 0.001，已改。驗證集只掃了 σ=1.1，集中式無 DP 沿用 0.02，README 已註明。

## 先前回合
- round-01：非 ASCII HMAC 標籤（bytes 比對＋2 個測試）；Krum／集中式結果複製到本機並重算；主表措辭限定。

## 已知還沒做
- 主表重跑中（約 4 小時），完成後補配對差值；6／14 台入侵、雜訊攻擊、重送等列這次沒排進重跑。
- 連續多輪斷線、固定一群 VD 離線的可用性實驗。

## 想請審查的重點
- README「攻擊與可用性實驗」各表是否都有 3 seed 與同 seed 配對差值，哪些結論其實不顯著。
- 課程主題中證據最薄的一項。
- `guard.py`、`secagg.py`、`security.py` 有沒有測試沒涵蓋的漏洞。
