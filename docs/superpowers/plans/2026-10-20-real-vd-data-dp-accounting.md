# 實作計畫：真實 VD 資料、ε 累計與對照實驗

對應 spec：`docs/superpowers/specs/2026-10-20-real-vd-data-dp-accounting-design.md`
執行方式：inline（executing-plans），每步先寫測試（TDD）。

## 環境限制
開發環境無法連 PyPI 與高公局網站：torch / flwr / opacus 無法安裝。
- 純 Python/numpy/pandas 部分：在此實跑測試。
- `privacy.py`：測試時以檔案路徑載入 Opacus 的 `accountants/analysis/rdp.py`（純 numpy/scipy）。
- torch/flwr 部分：語法檢查 + 對照 Flower 1.26.1 / Opacus 原始碼人工審查；端到端由使用者本機執行。

## 步驟
1. `vd_parse.py` + `tests/test_vd_parse.py`：命名空間無關解析、無效車速處理、流量加權。
2. `scripts/make_synthetic_vd.py`：產生 raw CSV（日週期、雜訊、缺口、一個爛 VD）。
3. `scripts/preprocess_vd.py` + 測試：5 分鐘聚合、補短缺口、選 VD、manifest。
4. `scripts/download_vd.py` + 測試：URL 產生、續傳判斷、解析整合（用本機假檔，不連網）。
5. `data.py` + 測試：manifest/partition、時間切分無洩漏、標準化、視窗。
6. `privacy.py` + 測試：歷史累計、與獨立二項式法一致、單調、q=1 封閉解。
7. `results_log.py` + 測試：彙整函式、CSV/JSON 寫出。
8. `task.py`、`client_app.py`、`server_app.py`（Message API）；`pyproject.toml` 設定。
9. `scripts/run_experiments.sh`、`scripts/plot_results.py`、`scripts/smoke_test.sh`。
10. `.gitignore`、`README.md`。
11. 驗證：全部測試、`py_compile`、獨立 agent 審查、修正、推送 feature 分支。
