# 智慧交通可信任聯邦式學習（Flower + Opacus）

以**高公局國道 VD（車輛偵測器）真實資料**做 5 分鐘車速預測：每個 VD 是一個聯邦 client，
資料不出端；client 端以 **DP-SGD（Opacus）** 訓練，並正確累計、輸出各 client 的隱私預算 ε。
同一套程式碼可切換成**無 DP 對照組**，一鍵跑完並畫比較圖。

| 項目 | 設定 |
|---|---|
| 任務 | 過去 12 筆 5 分鐘平均車速（1 小時）→ 預測下一個 5 分鐘車速 |
| 模型 | 殘差 MLP 14→32→32→1（12 筆車速＋2 個時段特徵）：輸出 = 最後一筆車速 + MLP 修正量（Opacus 相容） |
| Client | 1 個 VD = 1 個 client（預設國道 1 號、資料最完整的 8 個 VD） |
| 切分 | 各 client 依時間前 80% 訓練、後 20% 測試；視窗不跨切點；標準化只用訓練段 |
| 聚合 | FedAvg，每輪全部 client 參與；SGD 學習率 0.2、MSE 損失 |
| 隱私 | record-level DP-SGD：逐樣本裁剪（C=1.0）+ 高斯雜訊（σ=1.1）+ Poisson 取樣；RDP accountant，δ=1e-5 |
| 通訊安全 | 上傳更新附 HMAC-SHA256、全域模型附 Ed25519 簽章（可關閉）；可選穩健聚合與攻擊模擬 |
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
（內插在時間切分前進行。第 72 項實測 8 週 32 台：51.6 萬個點中只有 527 個（0.10%）是內插的，切點前後 3 格內沒有；完全不內插重跑，在相同的測試視窗上 MAE 只差 +0.002，影響可忽略。）

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
  local_mae_kmh（各 client 本輪本地訓練完、聚合前的模型在自己測試集的 MAE，相當於全域模型再多 1 輪本地訓練，可作為個人化的粗略參考；
  它就是 DP 訓練的輸出，不多花隱私預算）、
  ft_mae_kmh／ft_epsilon_max（只在最後一輪有值：各 VD 拿最終全域模型在本地再訓練 `finetune-epochs` 個 epoch 後的 MAE，
  以及把這幾步也算進去的 ε，見附錄 docs/EXPERIMENTS.md「個人化微調」）
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
ε 是**每個 client 對其本地訓練樣本**的 DP-SGD 帳本結果；`epsilon_max` 為所有 client 中最大者。
精確地說，這是「在固定的前處理與樣本定義下」的保證，**不是**整條資料處理流程對「一筆原始 5 分鐘紀錄」的端到端保證（2026-11-24 外部審查指出，見下方三點）。

**ε 涵蓋的範圍（報告中請註明）**
- 保護單位是一個**訓練樣本**（12 筆輸入＋1 筆目標的視窗），不是一筆原始紀錄：一筆原始 5 分鐘車速最多出現在 13 個樣本裡，
  所以對原始紀錄的保證要用群組隱私換算，會比帳本上的 ε 弱很多（不能只把 ε 乘 13，見下一點）。
- 標準化用的平均與標準差是從該 VD 的私有訓練資料算的，沒有加雜訊、也沒有計入帳本；改一筆原始紀錄會改到全部樣本。
  要得到嚴格的保證，應改用事先固定的公開尺度（例如全國道路的速限或公開統計），或把這兩個數字也用 DP 估計並計帳。這裡沒做。
- 原始資料本來就是高公局公開的 VD 聚合值，不含車牌或個別車輛；本專案做 DP 是為了驗證「資料不出端＋DP」這套機制在交通資料上的代價，
  不是在保護已知的敏感個資。
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

## 5. 主要結果（摘要）

完整的 72 項實驗（設定、每個 seed 的數字、重現指令）在 **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**，這裡只列結論。
除非另外註明，都是 3 個 seed 的平均；「顯著」指配對 t 檢定 p < 0.05（`scripts/seed_stats.py`）。攻擊與防禦主表補到 10 個 seed，並對整張表做 Holm 多重比較校正。

**怎麼測試**

所有成績都是用模型訓練時沒看過、而且時間在後面的資料算的。

- **依時間切開**：每台 VD 的資料照時間排好，前 80% 訓練、後 20% 測試（28 天資料約前 22 天訓練、最後 5～6 天測試）。
  測試段完全不參與訓練；標準化的平均與標準差只用訓練段算；1 小時的輸入視窗不跨過切點。
  不隨機打散：相鄰 5 分鐘的資料幾乎一樣，打散會讓成績虛高，也不符合「用過去預測未來」的實際用法。
- **換資料再驗證**：8／32／128／256 台 VD；14 天、28 天、8 週；南向 32 台（DP 1.777，naive 2.085）；
  5 折時間區塊交叉驗證（輪流拿不同時段當測試，避免剛好測到週末）、按天 bootstrap、10 個 seed；拿掉春節期間重跑。
- **新 VD**：32 台分 8 折，每次留 4 台完全不參加訓練，只拿來測（標準化用它自己前 80% 的資料）。
  當新 VD 時全域 MAE 2.193，有參加訓練 2.183，差 +0.0095 [+0.0035, +0.0155]（p = 0.021，顯著但很小）；26／32 台仍贏 naive，
  輸 naive 的 6 台有參加訓練時也輸。只測了同一條國道北向，其他路段沒有測（附錄最後一節）。

**準確度與隱私**
- 8 個 VD、14 天：殘差 MLP 加 DP-SGD（σ=1.1，ε≈3.6）MAE 2.712，無 DP 2.744，naive 2.916；DP 的準確度代價在 seed 波動內。
  殘差設計是關鍵：舊版直接預測車速時，DP 模型和 naive 差不多。
- 32 個 VD、28 天：DP σ=1.1 全域模型 2.183、本地微調後 2.116，naive 2.429。
- 8 週、南向 32 台：DP 1.777、無 DP 1.768、naive 2.085；DP 多 0.009，p = 0.067，沒有顯著差異，但不能說成沒有代價。
- client-level DP（保護整台 VD）8 週 32 台：σ=1 ε≈30、σ=2 ε≈12，MAE 約 2.33～2.42。台數少時代價大，256 台加抽樣才實用。
- ε 的意義與限制見上一節：是固定前處理下、以訓練視窗為單位的帳本，不是對一筆原始紀錄的端到端保證。

**攻擊與防禦**（32 個 VD、28 天、DP σ=1.1；這一段是 10 個 seed，整張主表 34 個檢定做 Holm 校正，校正後 p < 0.05 才算顯著）
- 25% 上傳被竄改或重送：沒驗證時竄改讓模型平均輸給 naive（2.787，+0.606 [+0.305, +0.908]，校正後 p = 0.025）；HMAC 把輪次簽進去後全部被拒絕。
- 3 台 VD 被入侵（有合法金鑰）：FedAvg 完全壞掉（MAE 約 296）；中位數、截尾平均 2.23，仍贏 naive 8%。
  10 台時中位數略輸 naive（2.467），Multi-Krum 2.22 最好（3 seed）；14 台（44%）全域模型守不住。
  「裁剪＋中位數疊加最好」撤回：10 台時疊加對只中位數 −0.085，未校正 p = 0.006，校正後 0.085。
- HMAC 擋下竄改後每輪少用約 1/4 的更新：3 個 seed 時代價 +0.010（p = 0.023），補到 10 個 seed 是 +0.003 [−0.002, +0.009]（p = 0.16），不再顯著；平均配對差的 95% 信賴區間上界是 0.009 km/h（不是確定的代價上限）。
- 沒有攻擊時中位數多 +0.006 [+0.002, +0.011]（未校正 p = 0.012，校正後 0.14）；每輪 30% 斷線多 +0.006 [−0.001, +0.014]：都不算顯著，但不是沒有代價。
- 全部防禦一起開，同時 3 台被入侵、10% 竄改、10% 斷線：2.291，比全部防禦沒攻擊多 +0.020（顯著），仍贏 naive 5.7%，每台每輪只上傳 0.79 KB。
- 驗證集過濾（ValidationFilter）擋固定攻擊大多有效；攻擊者會躲（只偶爾出手）時擋下的比例掉到約一半，
  模型多 0.016（p = 0.028，顯著）。細節見附錄第 66～67 項。
- 安全聚合（X25519 成對遮罩加 Shamir 復原）、上傳量化、int8 邊緣推論都有實作與測試（int8 推論誤差多 0.011，p = 0.026，小但顯著；模型從 6.3 KB 降到 2.1 KB）；int8 也能輸出純整數 C 檔，但這台電腦沒有 C 編譯器，「編譯後逐位元相同」的測試還沒實際跑過。

**資料**
- 春節期間的資料有用：拿掉後 DP 變差 0.026（p = 0.022）。
- 內插只占 0.10% 的點，完全不內插在相同測試視窗上只差 0.002。
- 資料本身晚 4～7 分鐘，`horizon=1` 在即時情境下接近補報現在；要真的預測未來需 `horizon` ≥ 3。

## 6. 測試

```bash
python -m pytest                 # 或 python -m unittest discover -s tests -t .
bash scripts/smoke_test.sh       # 不需網路：用合成資料跑完整流程（DP 與無 DP 各 2 輪）並檢查輸出
```

- 資料解析、前處理、時間切分無洩漏、標準化、ε 計算（與獨立實作的整數階 RDP 公式互相驗證、
  隨輪次遞增）、紀錄檔與畫圖都有單元測試。
- `tests/test_task_torch.py` 需要 torch/opacus，包含「本專案 ε 與 Opacus accountant 單輪結果一致」的檢查。
- `scripts/make_synthetic_vd.py` 產生的是**合成資料**，只供測試，請勿用於報告。

## 7. 實際使用：即時車速預測

把訓練好的全域模型拿來預測國道 1 號北向這 32 台 VD 未來 5／10／15 分鐘的車速。資料即時從高公局開放資料（`tisvcloud.freeway.gov.tw`，和下載腳本同來源）抓最近約 70 分鐘的每分鐘檔，彙整成 5 分鐘車速後輸入模型。

```bash
# 指令列：印出每台 VD 的預測表
python scripts/forecast_live.py --run-dir results/<批次>/dp_sigma1.1_seed1 --data-dir data/processed_28d_32
# 回測：指定過去的時間點，會一起列出實際值
python scripts/forecast_live.py --run-dir ... --data-dir ... --vd VD-N1-N-10.060-M-RS --at "2024-03-29 08:00"
# 網頁：每 5 分鐘自動更新，瀏覽器打開 http://127.0.0.1:8000
python scripts/forecast_web.py --run-dir ... --data-dir data/processed_28d_32
```

- 每格是「那個時刻開始的 5 分鐘」平均車速；顏色 80 以上順暢、40～80 車多、40 以下壅塞。最後 1 小時有缺值的 VD 顯示「資料不足」，不硬猜。
- +10、+15 分鐘是把前一步的預測接回輸入再算（遞迴），誤差會累積；塞車時預測的回升通常比實際快。
- 開放資料本身晚 4～7 分鐘上傳（見第 5 節）。預測從最後一段完整的實測往後推三個 5 分鐘，所以網頁每欄直接標實際時刻和「現在這段／幾分後」，第一欄通常就是現在這 5 分鐘。
- 資料比預測時間點舊超過 20 分鐘（開放資料停更）就不預測，網頁會說資料停在幾點；回測時多抓的未來資料只拿來對答案，不參與補值。
- 公開網址：可用 Cloudflare 快速通道（`cloudflared tunnel --url http://127.0.0.1:8000`）。伺服器對每個來源限制每分鐘 120 個請求。
- 標準化參數取自訓練資料，所以只支援 `data-dir` 裡有的 VD；沒參加訓練的新 VD 平均只差 0.01（見第 5 節「新 VD」），但要先收一段它自己的資料，算出標準化參數並加進 `data-dir`。
- 網頁預設只聽本機；`--host 0.0.0.0` 可給同網路的人看。沒有登入機制，只用標準函式庫的 http.server，適合展示，不適合直接放上網際網路。
- 模型檔用 `torch.load(..., weights_only=True)` 載入，不會執行檔案裡的程式；需要「昨天同時刻」特徵（lag-day）的模型目前不支援即時預測。

## 專案結構

```
vd_parse.py        MOTC v2.0 VDLive XML 解析
vd_preprocess.py   5 分鐘聚合、補缺口、選 VD、manifest
data.py            client 對應、時間切分、標準化、滑動視窗
privacy.py         跨輪次累計 ε
task.py            模型、訓練、評估
client_app.py      Flower ClientApp（DP 開關、ε 回報）
server_app.py      Flower ServerApp（聚合方式、紀錄、client 數檢查）
robust_agg.py      聚合方式登錄表：FedAvg／中位數／截尾平均／Krum／Multi-Krum／Bulyan
security.py        HMAC 更新驗證、Ed25519 模型簽章、攻擊與斷線模擬
guard.py           伺服器端安全閘道（驗簽、剔除斷線 VD、模擬中間人、安全聚合加總）
secagg.py          安全聚合：X25519 成對遮罩
compress.py        上傳壓縮：更新量隨機捨入量化
results_log.py     指標彙整與 CSV/JSON 輸出
scripts/forecast_live.py   即時預測（指令列）；scripts/forecast_web.py 預測網頁
scripts/quick_ablation.py  不經 Flower 的快速消融實驗（預測距離、個人化、成員推論攻擊、相鄰 VD、集中式／各自訓練對照）
scripts/           下載、前處理、合成資料、Flower 設定、實驗、畫圖、煙霧測試
tests/             單元測試
docs/superpowers/  設計文件與實作計畫
```

## 限制與後續

- 預設是 record-level DP，保護單位是一個訓練視窗；一筆原始紀錄最多出現在 13 個視窗，報告中解讀 ε 時應註明（見第 4 節）。
  client-level DP 已可選用，但台數少時代價很大（見第 5 節）。
- 模型為小型 MLP；DP-GRU 與上下游 VD 特徵都試過（2026-10-31），在這組資料上沒有比較好。
- 個人化微調後的模型只在各 VD 本地使用；若要對外發布，請用全域模型的 ε 與 MAE。
- 只用 VD 數值資料，沒有影像。未來可在路口攝影機的端點用影像辨識（例如 YOLO 偵測＋區域計數）算出每 5 分鐘車流量與車速，**只上傳數字、不上傳影像**，當作新特徵或新的 client；這和本專案「資料不出端」的設計一致。車牌辨識、個別車輛追蹤會產生可識別個人的資料，與本專案的隱私目標衝突，不列入規劃（見「參考外部專案補強」）。
