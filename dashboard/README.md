# 即時儀表板（FL LIVE）

即時顯示聯邦學習的訓練過程，以及在本專案中使用 Claude Code 的 token 和工具呼叫。
**監控本身不呼叫任何 LLM，所以不花 token**；只用 Python 標準函式庫加一個 HTML 檔，不需要額外安裝套件。

## 用法（Windows 開兩個終端機，都先 `.venv\Scripts\activate`）

```bash
# 終端機 1：儀表板（會自動開瀏覽器 http://127.0.0.1:8765）
python dashboard/server.py

# 終端機 2：照常跑實驗，多加 --dashboard
python scripts/run_experiments.py --rounds 20 --sigmas 1.1 --dashboard
python scripts/run_experiments.py --dashboard --extra "dropout-rate=0.1 attack-clients=1 tamper-rate=0.05"  # 看攻擊／斷線效果
```

重播跑完的實驗（每個 run 結束時都會留一份 `events.jsonl`）：

```bash
python dashboard/server.py --replay results/<批次>/dp_sigma1.1 --speed 2 --loop
```

## 畫面怎麼讀

| 元素 | 意義 |
|---|---|
| 一束線 = 一個 VD | 從外圈的 VD 連到中央的聚合伺服器 |
| 綠線往中心推進 | 本地訓練進度（batch 步數 / 總步數），線頭白點是目前位置 |
| 洋紅色雜訊點 | DP-SGD 正在加高斯雜訊 |
| 青色脈衝衝進中心 | 上傳模型更新 |
| 白色從中心往外 | 廣播（簽章後的）全域模型，新一輪開始 |
| 中心綠色波紋 | FedAvg 聚合完成 |
| 紅色虛線閃爍／外圈紅閃 | VD 斷線／安全閘道拒絕或偵測到竄改 |
| 琥珀色 | 本地訓練發散或模擬攻擊者 |
| 節點旁的洋紅弧 | 該 VD 的累計 ε（以目前最大的 ε 為滿圈） |
| 外圈點陣環 | 輪次進度 |
| 下方 tool calls | Claude Code 每次工具呼叫是一條線，粗細代表 token 數，洋紅色是子代理 |

## 設定

- 事件檔預設是 `results/live/events.jsonl`（`results/` 已在 .gitignore 裡）。不加 `--dashboard` 時完全不寫事件，實驗結果不受影響。
- Claude Code 紀錄會自動到 `%USERPROFILE%\.claude\projects\<本專案路徑>` 找；找不到時用 `--claude-dir` 指定。
- 花費是依照 `dashboard/prices.json` 的單價估算出來的，請自行對照官方價格修改。
- 別台電腦要看：`--host 0.0.0.0`（只建議在可信任的區網使用）。
