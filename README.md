# 智慧交通可信任聯邦式學習（Flower + Opacus）

以**高公局國道 VD（車輛偵測器）真實資料**做 5 分鐘車速預測：每個 VD 是一個聯邦 client，
資料不出端；client 端以 **DP-SGD（Opacus）** 訓練，並正確累計、輸出各 client 的隱私預算 ε。
同一套程式碼可切換成**無 DP 對照組**，一鍵跑完並畫比較圖。

| 項目 | 設定 |
|---|---|
| 任務 | 過去 12 筆 5 分鐘平均車速（1 小時）→ 預測下一個 5 分鐘車速 |
| 模型 | 殘差 MLP 12→32→32→1：輸出 = 最後一筆車速 + MLP 修正量（Opacus 相容） |
| Client | 1 個 VD = 1 個 client（預設國道 1 號、資料最完整的 8 個 VD） |
| 切分 | 各 client 依時間前 80% 訓練、後 20% 測試；視窗不跨切點；標準化只用訓練段 |
| 聚合 | FedAvg，每輪全部 client 參與；SGD 學習率 0.2、MSE 損失 |
| 隱私 | record-level DP-SGD：逐樣本裁剪（C=1.0）+ 高斯雜訊（σ=1.1）+ Poisson 取樣；RDP accountant，δ=1e-5 |
| 框架 | Flower 1.26（Message API）、PyTorch、Opacus |

## 1. 安裝

需要 Python 3.11+。Windows 可直接用跨平台的 `python scripts/run_experiments.py`（不需 bash），
或點兩下 `scripts/run_all_windows.bat` 一次跑完安裝、測試、煙霧測試、下載、前處理與實驗
（進度寫在 `status.txt`、完整輸出在 `run_log.txt`）。

**Windows 注意**：PyTorch 需要 Microsoft Visual C++ Redistributable（x64）。若出現
`WinError 126 ... torch\lib\shm.dll`，請安裝 https://aka.ms/vs/17/release/vc_redist.x64.exe 後重開終端機。

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 2. 取得真實資料

資料來源：交通部高速公路局「交通資料庫」歷史 VD 動態資訊（MOTC v2.0，每分鐘一檔），
`https://tisvcloud.freeway.gov.tw/history/motc20/VD/YYYYMMDD/VDLive_HHMM.xml.gz`，
授權為政府資料開放授權條款－第 1 版。

```bash
# 下載 14 天、只保留 VDID 含 "N1"（國道 1 號）的偵測器；邊下載邊解析，只存 CSV
python scripts/download_vd.py --start 2024-03-04 --days 14 --vd-pattern "N1"

# 5 分鐘聚合、補短缺口、挑出資料最完整的 8 個 VD
python scripts/preprocess_vd.py --num-clients 8 --vd-pattern "^VD-N1-[NS]-[0-9.]+-M-"   # 只取主線（-M-），排除匝道
```

- 一天約 1,440 個檔案，下載可中斷，重跑會跳過已完成的日期（`.done` 標記）。
- 單日缺檔超過 `--max-failed`（預設 30 分鐘）不會標記完成，重跑即可補抓。
- 想指定特定 VD：`python scripts/preprocess_vd.py --vd-ids <VDID> <VDID> ...`
- 建議避開春節、連假與颱風日，或在報告中說明期間。

**資料清理規則**：車道車速 ≤0 或 ≥200、或 Status≠0 視為無效；VD 車速 = 有效車道的流量加權平均
（流量皆為 0 時取平均）；每 5 分鐘至少 3 分鐘有效才算數；連續 ≤3 格（15 分鐘）的缺口線性內插，
更長的缺口保留空值，含空值的訓練視窗直接捨棄；只挑有效率 ≥80% 的 VD。
（內插在時間切分前進行，切點兩側 ≤15 分鐘的缺口會互相參考鄰近值，影響至多 3 個點，可忽略。）

輸出：`data/processed/<VDID>.csv`（time, speed, volume, occupancy）與 `manifest.json`
（client 編號 ↔ VDID、期間、有效率、篩選條件）。

## 3. 跑實驗（DP + 無 DP 對照組）

```bash
bash scripts/run_experiments.sh                          # 20 輪；無 DP + DP σ=1.1
ROUNDS=30 SIGMAS="0.8 1.1 1.5" bash scripts/run_experiments.sh   # 多個隱私等級
EXTRA="learning-rate=0.02 local-epochs=2" bash scripts/run_experiments.sh
SIGMAS="0.8 1.1 1.5" SEEDS="1 2 3" bash scripts/run_experiments.sh   # 每個設定重複 3 個 seed
# Windows / 不用 bash：
python scripts/run_experiments.py --sigmas 0.8 1.1 1.5 --seeds 1 2 3
```

腳本會：
1. 依 `manifest.json` 的 VD 數，在專案內 `.flwr_home/config.toml` 寫入 `num-supernodes`
   （Flower 1.26 起 client 數只能寫在 Flower 設定檔，`--federation-config` 會被忽略；
   用專案專屬的 `FLWR_HOME` 不會動到你家目錄的設定）；
2. 以相同數字傳 `num-clients` 給 ServerApp，三者不一致時會直接報錯，而不是卡住等待；
3. 以絕對路徑傳 `data-dir`、`results-dir`；
4. 依序跑無 DP 與各 σ 的 DP，最後執行 `scripts/plot_results.py`。

單獨跑一次（需自行確保 Flower 設定檔的 `num-supernodes` = VD 數）：

```bash
flwr run . local --run-config "num-clients=8 use-dp=false data-dir='/abs/path/data/processed' results-dir='/abs/path/results'"
```

所有可調參數在 `pyproject.toml` 的 `[tool.flwr.app.config]`。

### 輸出

```
results/<批次時間>/
├── nodp/            metrics.csv  clients.csv  config.json  final_model.pt
├── dp_sigma1.1/     （同上）
├── comparison.png         左：各設定每輪測試 MAE（多 seed 時畫平均）；右：DP 累計 ε
├── summary.csv            各 run 最終／最佳 MAE、RMSE、最終 ε、naive 基準與改善幅度
└── summary_by_config.csv  同設定跨 seed 的最終 MAE 平均 ± 標準差
```

- `metrics.csv`：每輪實際回覆的 client 數（不足時會發出警告）、train_loss、eval_loss、mae_kmh、rmse_kmh、epsilon_max、epsilon_mean、
  naive_mae_kmh（「下一筆 = 最後一筆」的 MAE，模型至少要贏過它才算有學到東西）、
  local_mae_kmh（各 client 本輪本地訓練完、聚合前的模型在自己測試集的 MAE，即「個人化」模型的參考值；
  它就是 DP 訓練的輸出，不多花隱私預算）
- `summary.csv` 的 `best_mae_kmh` 是在測試集上挑最好的一輪，只供參考；報告請用 `final_mae_kmh`
- `clients.csv`：每輪每個 client（含 VDID）的樣本數、loss、MAE、RMSE、累計 ε、DP 步數與 q
- `config.json`：本次完整 run config、套件版本、資料期間與 client 清單

## 4. 隱私預算 ε 怎麼算

Flower 每輪都會重新建立 ClientApp，若每輪新建 `PrivacyEngine` 並呼叫 `get_epsilon()`，
accountant 會歸零，得到的只是「單輪」ε（舊版程式與 Flower 官方 Opacus 範例都是這樣）。

本專案每輪把 `(σ, q, 步數)` 存進該 client 的 `context.state`，每次以全部歷史計算 RDP 總和再轉成
(ε, δ)，與 Opacus `RDPAccountant` 使用同一組 orders 與同一個函式（`privacy.py`）。
q 取自 Opacus `DPDataLoader.sample_rate`（Poisson 取樣的真實機率 = 1/每 epoch 批次數）；
每輪的步數與 q 也寫進 `clients.csv`（`dp_steps`、`sample_rate`），可自行驗算。
ε 是**每個 client 對其本地資料**的 record-level 保證；`epsilon_max` 為所有 client 中最大者。

**ε 涵蓋的範圍（報告中請註明）**
- 受保護的只有 client 上傳的**模型參數**。DP 模式下 client 仍回報未加雜訊的 `train-loss`、
  訓練／測試樣本數與測試 MAE，這些是研究用診斷資訊，**不在 (ε, δ) 保證內**；
  實際部署若要完整保證，應拿掉或另行加噪並計入預算。
- 為了可重現，取樣與雜訊使用固定 seed 的亂數產生器（未開 Opacus `secure_mode`）；
  形式上的 DP 假設雜訊不可預測，這是研究用設定。

以預設 14 天資料估算（每 client 約 3,200 個訓練視窗、批次 64 → q≈1/50、每輪 1 個 local epoch、δ=1e-5），約為：

| σ | 1 輪 | 5 輪 | 10 輪 | 20 輪 |
|---|---|---|---|---|
| 0.8 | 2.85 | 4.20 | 5.37 | 7.20 |
| 1.1 | 1.27 | 1.93 | 2.58 | 3.59 |
| 1.5 | 0.62 | 1.08 | 1.51 | 2.16 |

（實際值會隨資料量而變，以 `metrics.csv` 為準。）

### 真實資料調參結果（2026-10-30，國道 1 號北向 8 個主線 VD，2024-03-04～03-17，20 輪 × 3 seed）

| 設定 | 無 DP MAE | DP σ=1.1 MAE | naive MAE |
|---|---|---|---|
| 舊版：直接預測車速、lr=0.05 | 2.755 ± 0.018 | 2.918 ± 0.039 | 2.916 |
| 殘差模型、lr=0.05 | 2.740 ± 0.023 | 2.786 ± 0.011 | 2.916 |
| **殘差模型、lr=0.2（新預設）** | 2.743 ± 0.033 | **2.712 ± 0.015** | 2.916 |

- 舊版 DP 模型的 MAE 和「下一筆 = 最後一筆」差不多，等於沒學到東西；改成殘差模型後，模型一開始就等於 naive，
  只需學修正量，裁剪造成的慢收斂不再拖累結果。
- 新預設下 DP 甚至略優於無 DP：只裁剪、不加雜訊（σ=0）也有 2.709 ± 0.011，表示**逐樣本裁剪**對交通資料的
  極端值（事故、壅塞突變）有穩定作用。若無 DP 改用 L1 損失（`loss="l1"`、lr=0.05）可到 2.721 ± 0.026，
  與 DP 相當。結論：在這組設定下，DP 的準確度代價在隨機波動範圍內。
- 學習率 0.3～0.4、裁剪上限 C=0.5 或 2、隱藏層 64、Huber 損失、30 輪、批次 32／128、伺服器動量、
  加時間特徵（時段、假日）或流量／佔有率特徵、回看 24 筆，都沒有比新預設更好。
- 回看 6 筆（30 分鐘）在 5 個 seed 下：DP 2.703 ± 0.013（與 12 筆的 2.708 相當），無 DP 2.707 ± 0.014
  （12 筆為 2.752 ± 0.038）。若想讓無 DP 對照組更強，可用 `--extra "lookback=6"`。
- 個人化參考（各 client 最後一輪本地模型）：DP 2.696、無 DP 2.711，比全域模型再好約 0.02～0.03。
- 不同 VD 差異大：國道 1 號北向 10 公里附近的 3 個 VD，模型與 naive 幾乎一樣（約 3.5～3.8），其他 VD 都明顯較好。
- 新預設已用 Flower 端到端重跑（3 seed）：無 DP 2.744 ± 0.036；DP σ=0.8/1.1/1.5 為 2.711/2.712/2.719（± 0.01），
  ε=7.20/3.59/2.16；無 DP 改用 L1 損失 2.720 ± 0.026。RMSE 方面無 DP（4.45）略優於 DP（4.47），
  表示 DP 的裁剪讓大誤差稍多、平均誤差稍少。

### 解讀：為什麼不同 σ 的 MAE 幾乎一樣？

雜訊確實有加上（ε 隨 σ 明顯不同就是證據），只是在這個設定下影響很小：DP-SGD 每步加在平均梯度上的
雜訊標準差是 σ·C / 批次大小 ≈ 1.1 × 1.0 / 64 ≈ 0.017，再經 8 個 client 平均。
以下是合成資料（8 個 client、20 輪、2026-10-30）的觀察，**真實資料需另外確認**：
- σ=0.8/1.1/1.5 之間的最終 MAE 差約 0.0005～0.002 km/h，小於換 seed 造成的差異（約 0.03 km/h）。
- 只裁剪、不加雜訊（σ=0）的結果與加雜訊幾乎相同 → DP 與無 DP 的差距主要來自**逐樣本梯度裁剪**，而非雜訊。
- 裁剪上限 C=10 明顯變差；C=3～5 與 C=1 沒有一致的優劣。
- 學習率 0.05→0.2（各 3 個 seed）：DP σ=1.1 的 MAE 從 1.684±0.015 降到 1.652±0.010，
  反而略優於無 DP（lr=0.05 為 1.668±0.014）；無 DP 用 lr=0.2 則變差且不穩定（1.725±0.069）。
  裁剪把步長限住，所以 DP 能承受較大的學習率；兩組最好各自調學習率再比較。

因此報告比較 σ 時請用 `--seeds` 跑多個 seed、看 `summary_by_config.csv` 的平均 ± 標準差，
並在真實資料上用相同學習率比較 DP 與無 DP。

## 5. 測試

```bash
python -m pytest                 # 或 python -m unittest discover -s tests -t .
bash scripts/smoke_test.sh       # 不需網路：用合成資料跑完整流程（DP 與無 DP 各 2 輪）並檢查輸出
```

- 資料解析、前處理、時間切分無洩漏、標準化、ε 計算（與獨立實作的整數階 RDP 公式互相驗證、
  隨輪次遞增）、紀錄檔與畫圖都有單元測試。
- `tests/test_task_torch.py` 需要 torch/opacus，包含「本專案 ε 與 Opacus accountant 單輪結果一致」的檢查。
- `scripts/make_synthetic_vd.py` 產生的是**合成資料**，只供測試，請勿用於報告。

## 專案結構

```
vd_parse.py        MOTC v2.0 VDLive XML 解析
vd_preprocess.py   5 分鐘聚合、補缺口、選 VD、manifest
data.py            client 對應、時間切分、標準化、滑動視窗
privacy.py         跨輪次累計 ε
task.py            模型、訓練、評估
client_app.py      Flower ClientApp（DP 開關、ε 回報）
server_app.py      Flower ServerApp（FedAvg、紀錄、client 數檢查）
results_log.py     指標彙整與 CSV/JSON 輸出
scripts/           下載、前處理、合成資料、Flower 設定、實驗、畫圖、煙霧測試
tests/             單元測試
docs/superpowers/  設計文件與實作計畫
```

## 限制與後續

- 目前是 record-level DP（保護單筆 5 分鐘紀錄），不是 client-level DP；相鄰時間視窗高度相關，
  報告中解讀 ε 時應註明。
- 模型為小型 MLP；可再比較 GRU 或加入上下游 VD 特徵（需注意 Opacus 對 RNN 的支援方式）。
