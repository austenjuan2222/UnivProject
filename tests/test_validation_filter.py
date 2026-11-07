"""伺服器用自己的參考 VD 檢查每份更新（防針對塞車時段的攻擊）。"""
import unittest

import numpy as np

from guard import ValidationFilter


class ValidationFilterTest(unittest.TestCase):
    def test_rejects_updates_much_worse_on_reference_data(self):
        # 分數 = 參考資料上塞車時段的誤差；第 7、8 號明顯變差
        scores = {0: 5.0, 1: 5.1, 2: 4.9, 3: 5.05, 4: 4.95, 5: 5.0, 6: 5.1, 7: 9.0, 8: 8.5}
        f = ValidationFilter(lambda arrays: float(arrays["s"][0]), z=3.0)
        bad = f.screen({p: {"s": np.array([s])} for p, s in scores.items()})
        self.assertEqual(bad, {7, 8})

    def test_keeps_everyone_when_scores_are_similar(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        rng = np.random.default_rng(0)
        bad = f.screen({p: {"s": np.array([5.0 + 0.05 * rng.standard_normal()])} for p in range(20)})
        self.assertEqual(bad, set())

    def test_better_than_usual_is_not_rejected(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        scores = {p: 5.0 + 0.01 * p for p in range(8)} | {8: 1.0}
        self.assertEqual(f.screen({p: {"s": np.array([s])} for p, s in scores.items()}), set())

    def test_non_finite_score_is_rejected(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        scores = {p: 5.0 for p in range(5)} | {5: float("nan")}
        self.assertEqual(f.screen({p: {"s": np.array([s])} for p, s in scores.items()}), {5})

    def test_too_few_updates_only_rejects_broken_ones(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        ups = {0: {"s": np.array([1.0])}, 1: {"s": np.array([2.0])}, 2: {"s": np.array([np.inf])}}
        self.assertEqual(f.screen(ups), {2})

    def test_too_few_updates_skips_screening(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        self.assertEqual(f.screen({0: {"s": np.array([1.0])}, 1: {"s": np.array([99.0])}}), set())


    def test_margin_keeps_updates_only_slightly_worse_than_a_tight_cluster(self):
        # 多數更新幾乎一樣時 MAD 很小，只差 2% 也會超過 z×MAD；margin=0.05 要求至少比中位數差 5% 才拒絕
        scores = {p: 5.0 + 0.001 * p for p in range(9)} | {9: 5.1, 10: 6.0}
        ups = {p: {"s": np.array([s])} for p, s in scores.items()}
        self.assertEqual(ValidationFilter(lambda a: float(a["s"][0]), z=3.0).screen(ups), {9, 10})
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, margin=0.05)
        self.assertEqual(f.screen(ups), {10})
        self.assertEqual(f.last_scores[9], 5.1)


    def test_group_mode_scores_only_group_averages_and_rejects_whole_group(self):
        seen = []

        def score(a):
            seen.append(float(a["s"][0]))
            return float(a["s"][0])

        ups = {p: {"s": np.array([5.0 + 0.01 * (p % 3)])} for p in range(12)} | {12: {"s": np.array([40.0])}}
        f = ValidationFilter(score, z=3.0, group=4, seed=0)
        bad = f.screen(ups)
        self.assertEqual(len(seen), 3)  # 13 份分成 4、4、5：只看到 3 個平均，看不到個別更新
        # 被拒絕的正好是 12 號所在的那一整組（其他組平均都約 5）
        self.assertEqual(bad, {m for m in ups if f.last_scores[m] == f.last_scores[12]})
        self.assertIn(len(bad), (4, 5))

    def test_group_mode_remainder_merges_without_losing_members(self):
        sizes = []
        f = ValidationFilter(lambda a: sizes.append(1) or float(a["s"][0]), z=3.0, group=2, seed=0)
        ups = {p: {"s": np.array([5.0])} for p in range(5)}  # 2、2、1 → 2、3
        f.screen(ups)
        self.assertEqual(set(f.last_scores), set(ups))  # 每一份都有被評分（沒有整組被吃掉）
        self.assertEqual(len(sizes), 2)

    def test_memory_rejects_consistently_suspicious_update_below_single_round_threshold(self):
        # 9 號每輪都只比多數差一點（單輪 z 約 2，低於 3），累積幾輪後平均可疑程度超過 memory 就拒絕
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, memory=1.5)
        base = [5.0, 5.1, 4.9, 5.05, 4.95, 5.0, 5.1, 4.9, 5.0]
        rounds = []
        for r in range(3):
            ups = {p: {"s": np.array([v])} for p, v in enumerate(base)} | {9: {"s": np.array([5.25])}}
            rounds.append(f.screen(ups))
        self.assertEqual(rounds[0], set())
        self.assertIn(9, rounds[2])
        self.assertTrue(all(p == 9 for p in rounds[2]))

    def test_memory_off_by_default(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        base = [5.0, 5.1, 4.9, 5.05, 4.95, 5.0, 5.1, 4.9, 5.0]
        for _ in range(5):
            bad = f.screen({p: {"s": np.array([v])} for p, v in enumerate(base)} | {9: {"s": np.array([5.25])}})
        self.assertEqual(bad, set())

    def test_group_mode_falls_back_when_too_few_updates(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4)
        ups = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(6)} | {6: {"s": np.array([50.0])}}
        self.assertEqual(f.screen(ups), {6})


class GuardedStrategyValidationTest(unittest.TestCase):
    def test_rejected_updates_never_reach_inner_aggregator(self):
        from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

        from guard import GuardedStrategy

        seen = []

        class Inner:
            def aggregate_train(self, rnd, replies):
                seen.extend(int(m.content["metrics"]["partition-id"]) for m in replies)
                return None, None

        def reply(pid, v):
            sent = Message(content=RecordDict(), dst_node_id=100 + pid, message_type="train")
            c = RecordDict({"arrays": ArrayRecord({"s": Array(np.array([v], np.float32))}),
                            "metrics": MetricRecord({"partition-id": pid, "num-examples": 10})})
            return Message(content=c, reply_to=sent)

        stats = {}
        g = GuardedStrategy(Inner(), None, None, on_round=lambda r, s: stats.update(s),
                            validation_filter=ValidationFilter(lambda a: float(a["s"][0])))
        g.aggregate_train(1, [reply(p, 5.0 + 0.01 * p) for p in range(6)] + [reply(6, 50.0)])
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4, 5])
        self.assertEqual(stats["val_rejected"], "6")
        self.assertEqual(g.validation_scores[0][0], 1)
        got = sorted(g.validation_scores[0][1])
        self.assertEqual([p for p, _ in got], list(range(7)))
        np.testing.assert_allclose([v for _, v in got], [5.0 + 0.01 * p for p in range(6)] + [50.0], rtol=1e-6)

        # 惡意更新冒用誠實 VD 的編號：仍要被擋下，不能被同編號的誠實更新蓋過
        seen.clear()
        g.aggregate_train(2, [reply(p, 5.0 + 0.01 * p) for p in range(6)] + [reply(3, 50.0)])
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4, 5])
        self.assertEqual(stats["val_rejected"], "3")


class JamScorerTest(unittest.TestCase):
    def test_model_that_hides_congestion_scores_worse(self):
        import torch

        from task import jam_scorer

        mean, std = 80.0, 20.0
        X = np.zeros((4, 3), np.float32)
        X[:, -1] = ((np.array([40.0, 45.0, 100.0, 90.0]) - mean) / std)
        y = X[:, -1].copy()  # 下一筆 = 最後一筆
        refs = [{"X": X, "y": y, "mean": mean, "std": std}]

        class Shift(torch.nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = torch.nn.Parameter(torch.tensor(float(b)))

            def forward(self, x):
                return x[:, -1] + self.b

        score = jam_scorer(refs, lambda: Shift(0.0))
        honest = score({"b": np.array(0.0, np.float32)})
        hiding = score({"b": np.array(1.0, np.float32)})  # 全部預測快 20 km/h
        self.assertAlmostEqual(honest, 0.0, places=4)
        self.assertAlmostEqual(hiding, 20.0, places=3)


if __name__ == "__main__":
    unittest.main()
