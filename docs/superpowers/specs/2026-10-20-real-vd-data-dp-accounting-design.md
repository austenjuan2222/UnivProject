# 設計：真實 VD 資料、client 切分、ε 累計與對照實驗

日期：2026-10-20　狀態：已核可（使用者於對話中確認需求，並授權 01:42–08:00 自主實作）

## 目標
把「框架能跑通」的原型升級成可重現、可寫進報告的實驗：

1. 使用高公局歷史 VD 動態資訊（真實資料），每個 VD 偵測器 = 一個 client。
2. 每個 client 依時間切訓練／測試集（不打亂、無洩漏）。
3. 正確累計並輸出各 client 的隱私預算 ε（跨輪次）。
4. 輪數、client 數由設定決定且彼此一致。
5. 每次執行輸出紀錄檔；`use-dp=false` 為無 DP 對照組，一鍵跑完兩組並畫圖。
6. 補 `.gitignore`、README。

## 使用者決策
- 資料：台灣高公局 VD（非 METR-LA）。
- 任務：過去 12 筆（5 分鐘間隔，1 小時）預測下一筆車速。
- 切分：每個 VD = 一個 client。
- 取得：下載腳本在使用者本機執行（開發環境無法連到高公局）。

## 架構（方案 A：本機下載 → 前處理 → 模擬讀本地 CSV）

```
download_vd.py ─► data/raw/vd_minute_YYYYMMDD.csv  (每分鐘、VD 層級，已跨車道合併)
preprocess_vd.py ─► data/processed/<VD>.csv + manifest.json  (5 分鐘、補短缺值、選出 N 個 VD)
flwr run . ─► client_app.py ◄─ data.py / task.py / privacy.py
             server_app.py  ─► results/<run>/metrics.csv, clients.csv, config.json
run_experiments.sh ─► DP 與無 DP 兩組 ─► plot_results.py ─► results/comparison.png
```

### 模組
| 檔案 | 職責 | 依賴 |
|---|---|---|
| `vd_parse.py` | 解析 MOTC v2.0 `VDLiveList` XML → 每 VD 每分鐘一列 | 標準函式庫 |
| `scripts/download_vd.py` | 依日期抓 `history/motc20/VD/YYYYMMDD/VDLive_HHMM.xml.gz`，邊抓邊解析、只存 CSV；可續傳、可重試 | 標準函式庫 |
| `scripts/preprocess_vd.py` | 5 分鐘聚合、短缺值內插、依完整度選 VD、寫 manifest | pandas |
| `scripts/make_synthetic_vd.py` | 產生與 raw CSV 同格式的**合成**資料，供離線測試／煙霧測試 | numpy, pandas |
| `data.py` | 讀 manifest、partition 對應、時間切分、標準化、滑動視窗 | numpy, pandas |
| `privacy.py` | 由（σ, q, 步數）歷史計算累計 ε（Opacus RDP 分析函式） | opacus |
| `task.py` | MLP 模型、訓練、評估 | torch |
| `client_app.py` | Flower Message API ClientApp；DP 開關；步數存 `context.state` | flwr, opacus |
| `server_app.py` | FedAvg、彙整指標、寫實驗紀錄 | flwr |
| `results_log.py` | 指標彙整與 CSV/JSON 寫出（純 Python，可測） | 標準函式庫 |

## 關鍵設計

**資料清理**：車道車速 ≤0 或 ≥200、Status≠0 視為無效；VD 車速 = 有效車道的流量加權平均（流量全 0 時取簡單平均）。5 分鐘聚合同樣流量加權。≤3 格（15 分鐘）的缺口線性內插，更長缺口保留 NaN，含 NaN 的視窗直接捨棄。依有效率挑前 N 個符合 `--vd-pattern` 的 VD（預設 `N1`、N=8）。

**時間切分**：在時間軸 80% 處切；訓練視窗全在切點前，測試視窗全在切點後。標準化平均／標準差只用該 client 訓練段。

**ε 累計**：原本每輪重建 `PrivacyEngine`，accountant 歸零 → ε 只反映單輪。改為每輪把（σ, q=1/len(loader), 步數）附加到 `context.state`，每次以全部歷史計算 RDP 總和再轉 (ε, δ)。這是 Opacus `RDPAccountant` 的同一套公式，只是狀態由我們保存。

**設定一致性**：Flower 1.26 起 `num-supernodes` 只能寫在 Flower 設定檔（`--federation-config` 會被忽略）。執行腳本把 `FLWR_HOME` 指到專案內 `.flwr_home/`，依 manifest 中 VD 數量自動寫入 `num-supernodes`，並以同一數值傳 `num-clients` 給 ServerApp（`min_available_nodes`）。client 端檢查 `num-partitions` 與 manifest 數量一致，不一致即報錯。

**路徑**：`flwr run` 會把 app 打包安裝到別處執行，相對路徑不可靠 → 腳本以絕對路徑傳 `data-dir`、`results-dir`。

**對照組**：同一程式碼，`use-dp=false` 時不經 Opacus、不回報 ε；其餘超參數相同、相同 seed。

## 輸出
- `results/<run>/metrics.csv`：round, train_loss, eval_loss, mae_kmh, rmse_kmh, epsilon_max, epsilon_mean
- `results/<run>/clients.csv`：round, partition, vdid, epsilon, train_loss, mae_kmh
- `results/<run>/config.json`：run config、時間、套件版本
- `results/comparison.png`：DP vs 無 DP 的 MAE 曲線與 ε 曲線

## 測試
- 可在無 torch 環境跑：XML 解析、前處理、視窗/切分無洩漏、標準化只用訓練段、ε 計算（與獨立的整數階二項式展開法互相驗證、q=1 封閉解、隨步數單調遞增）、紀錄檔寫出。
- 需要 torch/flwr：模型形狀、DP 訓練一步、端到端模擬（README 提供指令，由使用者本機執行）。

## 不做（YAGNI）
- 不改成 GRU/圖神經網路；不做 client-level DP；不做 TDX API 版本。
