"""聚合方式登錄表：平均型（FedAvg／中位數／截尾平均）與距離式穩健聚合（Krum／Multi-Krum／Bulyan）。

距離式方法的想法取自 FLYOLO（聯邦 YOLO，比較 FedAvg、FedMedian、Krum、Bulyan 等 8 種策略）；
程式直接用 Flower 內建的實作，不複製該 repo 的程式碼。

- Krum（Blanchard et al. 2017）：每份更新算它和最近 n−f−2 份的距離平方和，挑總和最小的「一份」當新模型。
- Multi-Krum：挑分數最小的 m 份再平均（預設 m = n − f），比 Krum 少浪費誠實 VD 的資訊。
- Bulyan（El Mhamdi et al. 2018）：先用 Krum 反覆挑出 n−2f 份，再逐參數取最接近中位數的 n−4f 份平均。

f 是「假設最多幾台 VD 被入侵」，要在部署前決定（講義：依威脅模型選防禦）。
"""
from __future__ import annotations

from flwr.serverapp.strategy import Bulyan, FedAvg, FedMedian, FedTrimmedAvg, Krum, MultiKrum


class _AllMetrics:
    """Flower 的 Krum 系列只把「被選中」的回覆交給指標函式；這裡改成交出所有有效回覆，
    讓紀錄程式照樣記下每台 VD 的 ε、訓練 loss（模型參數仍只用被選中的那些）。"""

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        # 先記下所有有效回覆（Bulyan 聚合時會把參數取走，事後就讀不到了）
        contents = [m.content for m in replies if not m.has_error()]
        # 斷線或被拒後回覆變少：每輪重新檢查 Krum 系列的前提（建策略時只看設定的 n）
        if len(contents) < _need(self.__class__, self.num_malicious_nodes):
            return None, None
        if isinstance(self, MultiKrumAll):  # 預設挑 n−f 份要用本輪的 n，不然斷線後會連惡意的一起挑進來
            k = len(contents) - self.num_malicious_nodes
            self.num_nodes_to_select = min(self._m or k, k)
        fn, self.train_metrics_aggr_fn = self.train_metrics_aggr_fn, lambda _r, _k: None
        try:
            arrays, _ = super().aggregate_train(server_round, replies)
        finally:
            self.train_metrics_aggr_fn = fn
        if arrays is None:
            return None, None
        return arrays, fn(contents, self.weighted_by_key)


class KrumAll(_AllMetrics, Krum):
    pass


class MultiKrumAll(_AllMetrics, MultiKrum):
    _m = 0  # 使用者指定的 m；0 表示每輪用「本輪回覆數 − f」


class BulyanAll(_AllMetrics, Bulyan):
    pass


def _need(cls, f: int) -> int:
    return 4 * f + 3 if issubclass(cls, Bulyan) else 2 * f + 3


AGGREGATORS = {"fedavg": FedAvg, "median": FedMedian, "trimmed": FedTrimmedAvg,
               "krum": KrumAll, "multikrum": MultiKrumAll, "bulyan": BulyanAll}


def make_strategy(name: str, f: int, n: int, kwargs: dict, multikrum_m: int = 0):
    """依名稱建立聚合策略。f＝假設被入侵的 VD 數（只有 Krum 系列用到），n＝每輪參與訓練的 VD 數。"""
    cls = AGGREGATORS.get(name)
    if cls is None:
        raise ValueError(f"aggregator 必須是 {'／'.join(AGGREGATORS)}，收到 {name}")
    if name not in ("krum", "multikrum", "bulyan"):
        return cls(**kwargs)
    if f < 0:
        raise ValueError(f"byzantine-f 不能是負數，收到 {f}")
    need = _need(cls, f)
    if n < need:
        raise ValueError(f"{name} 假設 f={f} 台被入侵時，每輪至少要 {need} 台 VD 參與，現在只有 {n}")
    kw = dict(kwargs, num_malicious_nodes=f)
    if name == "multikrum":
        kw["num_nodes_to_select"] = multikrum_m or n - f
    s = cls(**kw)
    if name == "multikrum":
        s._m = multikrum_m
    return s
