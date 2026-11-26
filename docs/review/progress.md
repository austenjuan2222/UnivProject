# 進度摘要（Claude 維護，給 Codex 審查用）

更新：2026-11-26（第 1 回合後），分支 `claude/codex-review-loop`

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

## 上一回合改了什麼（回應 round-01）
- [high] 非 ASCII HMAC 標籤：`security.verify_update` 改成比 bytes。新增 `tests/test_security.py::test_non_ascii_tag_is_rejected_not_raised` 與 `tests/test_metrics_auth.py::test_non_ascii_tag_is_dropped_and_others_still_aggregate`（訓練、評估各一，確認其他回覆仍聚合）。兩個測試在修之前都失敗，修之後全套 382 個通過。沒有另外限制成 64 個十六進位字元：比 bytes 已經不會拋例外，且 Ed25519 模式的標籤長度不同。
- [medium] Krum／集中式結果追不到：從專案資料庫複製到本機 `results/krum32/`（23 組 summary.csv）與 `results/central32/central.jsonl`，重算後平均與 README 一致，配對差值有三處差 0.001（四捨五入），已改成重算值。README 路徑改為本機路徑。
- [medium] 主表措辭：「±0.011 以內」限定適用的列；14/32 改為實驗觀察；斷線結論限定為每輪隨機斷線。主表原始結果 `results/night5/` 不在桌機也不在專案資料庫，逐 seed 配對差值無法重算，README 已註明，尚未重跑。

## 已知還沒做
- 主表有入侵的列缺逐 seed 配對差值（需重跑，32 VD × 約 12 組 × 3 seed）。
- 連續多輪斷線、固定一群 VD 離線的可用性實驗。

## 想請審查的重點
- README「攻擊與可用性實驗」各表是否都有 3 seed 與同 seed 配對差值，哪些結論其實不顯著。
- 課程主題中證據最薄的一項。
- `guard.py`、`secagg.py`、`security.py` 有沒有測試沒涵蓋的漏洞。
