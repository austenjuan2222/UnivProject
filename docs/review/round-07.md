本回合 high: 1, medium: 0, low: 0

### [high] Multi-Krum 在斷線後可能把所有惡意更新一起平均

**證據**：`robust_agg.py:60–65` 只在建立策略時檢查參與數，並將選取份數固定成 `n−f`；`robust_agg.py:21–27` 聚合時直接呼叫 Flower，沒有依本輪有效回覆數重新檢查。已核對本機安裝的 Flower `multikrum.py`：`select_multikrum()` 使用 `np.argsort(scores)[:num_nodes_to_select]`，選取份數大於或等於回覆數時，會保留全部回覆。

`tests/test_robust_agg.py::test_multikrum_averages_honest_updates` 只測完整參與，未涵蓋斷線後的選取行為。

**為什麼是問題**：例如設定 `n=10、f=2`，預設選取 8 份；兩台誠實 VD 斷線後，只剩六份誠實更新與兩份惡意更新，策略仍選取 8 份，距離篩選完全失效。若等權重的誠實參數都是 1、惡意參數都是 100，聚合結果會是 25.75。這時有效回覆數 8 仍符合 `2f+3=7`，單純補參與數下限檢查也不足以解決。

**建議怎麼驗證或修**：

- 先新增 `tests/test_robust_agg.py::test_multikrum_dropout_does_not_select_every_reply`：建立 `_strategy("multikrum", 2, 10)`，傳入 `_replies(6, 2)`，要求結果維持在誠實更新附近；目前流程會選入兩份惡意更新。
- 每輪依通過安全閘道後的有效回覆數重新決定預設選取份數，並檢查 Krum／Multi-Krum 的參與數下限；不足時跳過該輪。
- 新增 `test_krum_skips_round_below_byzantine_threshold`，確認誠實 VD 斷線或更新被拒絕後，不會繼續在不符合條件的回覆集合上聚合。

### 上回合追蹤

最新 `docs/review/round-07.md` 是空檔，因此追蹤最近有內容的 `round-06.md`：

- **int8 推論將不顯著差異寫成確定代價：已解決。** `README.md:1574` 已改成「不顯著」，並限定為小幅增加的趨勢。
- **復原回覆非有限編號的測試缺口：已補上，靜態確認。** `tests/test_ephemeral.py:91–114` 已分別測試 `share-owners`／`eshare-owners` 含 `inf` 時，`_recover()` 回傳 `None`。

本回合未修改檔案，也未執行會寫檔的指令。專案 venv 的 Python 無法啟動，因此未動態驗證測試結果。