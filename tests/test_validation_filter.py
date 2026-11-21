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


    def test_contrast_tells_jam_only_worse_apart_from_worse_everywhere(self):
        # 分數 = (塞車時段誤差, 順暢時段誤差)。8 號兩邊都差（沒塞車經驗的誠實 VD 的樣子），
        # 9 號塞車時段差、順暢時段反而好（把塞車標籤改高的攻擊者的樣子），兩者塞車誤差一樣
        jam = {p: 5.0 + 0.01 * p for p in range(8)} | {8: 5.18, 9: 5.18}
        free = {p: 2.0 + 0.01 * p for p in range(8)} | {8: 2.2, 9: 1.8}
        ups = {p: {"s": np.array([jam[p], free[p]])} for p in jam}
        pair = lambda a: (float(a["s"][0]), float(a["s"][1]))
        self.assertEqual(ValidationFilter(lambda a: float(a["s"][0]), z=4.0).screen(ups), set())
        f = ValidationFilter(pair, z=4.0, contrast=1.0)
        self.assertEqual(f.screen(ups), {9})
        self.assertEqual(f.last_scores[9], 5.18)  # 存下來的仍是塞車誤差

    def test_contrast_forgiveness_is_capped(self):
        # 順暢時段也故意弄很差，不能換到更多寬容：塞車時段差很多的照樣擋
        jam = {p: 5.0 + 0.01 * p for p in range(8)} | {8: 9.0}
        free = {p: 2.0 + 0.01 * p for p in range(8)} | {8: 50.0}
        ups = {p: {"s": np.array([jam[p], free[p]])} for p in jam}
        f = ValidationFilter(lambda a: (float(a["s"][0]), float(a["s"][1])), z=4.0, contrast=1.0)
        self.assertEqual(f.screen(ups), {8})

    def test_contrast_does_not_reject_typical_jam_error_with_better_free_flow(self):
        # 塞車誤差和大家差不多、只是順暢時段特別好：沒有隱藏塞車的跡象，不擋
        jam = {p: 5.0 + 0.01 * p for p in range(8)} | {8: 5.06}
        free = {p: 2.0 + 0.01 * p for p in range(8)} | {8: 1.0}
        ups = {p: {"s": np.array([jam[p], free[p]])} for p in jam}
        f = ValidationFilter(lambda a: (float(a["s"][0]), float(a["s"][1])), z=4.0, contrast=1.0)
        self.assertEqual(f.screen(ups), set())

    def test_contrast_rejects_non_finite_free_flow_score(self):
        ups = {p: {"s": np.array([5.0 + 0.01 * p, 2.0 + 0.01 * p])} for p in range(8)} | {8: {"s": np.array([5.0, np.nan])}}
        f = ValidationFilter(lambda a: (float(a["s"][0]), float(a["s"][1])), z=4.0, contrast=1.0)
        self.assertEqual(f.screen(ups), {8})

    def test_contrast_ignores_tiny_free_flow_differences_when_all_alike(self):
        # 順暢誤差幾乎完全一樣時 MAD≈0，只好一點點也不能被放大成「順暢時段好很多」
        jam = {p: 5.0 + 0.01 * p for p in range(8)} | {8: 5.13}
        free = {p: 2.0 for p in range(8)} | {8: 1.9999}
        ups = {p: {"s": np.array([jam[p], free[p]])} for p in jam}
        f = ValidationFilter(lambda a: (float(a["s"][0]), float(a["s"][1])), z=4.0, contrast=1.0)
        self.assertEqual(f.screen(ups), set())

    def test_contrast_needs_individual_updates(self):
        with self.assertRaises(ValueError):
            ValidationFilter(lambda a: (0.0, 0.0), contrast=1.0, group=2)


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
            # 每輪訊息的鍵（順序）不同，但 ids 對到穩定的 VD 編號：歷史要跟著 VD 走
            ups = {(p + r) % 10: {"s": np.array([v])} for p, v in enumerate(base)} | {(9 + r) % 10: {"s": np.array([5.25])}}
            rounds.append(f.screen(ups, {(p + r) % 10: p for p in range(10)}))
        self.assertEqual(rounds[0], set())
        self.assertEqual(rounds[2], {(9 + 2) % 10})

    def test_memory_good_rounds_do_not_offset_bad_ones(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, memory=1.5)
        base = [5.0, 5.1, 4.9, 5.05, 4.95, 5.0, 5.1, 4.9, 5.0]
        ids = {p: p for p in range(10)}
        f.screen({p: {"s": np.array([v])} for p, v in enumerate(base)} | {9: {"s": np.array([2.0])}}, ids)  # 很好的一輪
        for _ in range(3):
            bad = f.screen({p: {"s": np.array([v])} for p, v in enumerate(base)} | {9: {"s": np.array([5.3])}}, ids)
        self.assertIn(9, bad)  # 負的可疑程度截成 0，不能存起來抵銷之後的可疑輪次

    def test_memory_rejects_all_messages_sharing_one_vd_id(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, memory=1.5)
        ups = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(10)}
        self.assertEqual(f.screen(ups, {p: (3 if p == 7 else p) for p in range(10)}), {3, 7})

    def test_group_memory_singles_out_the_vd_whose_groups_keep_failing(self):
        # 只看 4 台一組的平均（模擬組內安全聚合）。每輪隨機重新分組，19 號那組每輪都壞；
        # 和它同組的誠實 VD 每輪不同，只有 19 號每輪都在壞組裡，累積幾輪後單獨把它排除，之後它的組就不再連坐誠實 VD
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([30.0])}}
        ids = {p: f"vd{p}" for p in ups}
        rounds = [f.screen(ups, ids) for _ in range(10)]
        self.assertTrue(all(19 in r for r in rounds))
        self.assertTrue(all(len(r) >= 4 for r in rounds[:8]))  # 前 8 輪還不下判斷：整組一起擋
        self.assertEqual(rounds[-1], {19})  # 之後單獨排除，同組的誠實 VD 不再被連帶擋掉

    def test_group_memory_needs_ids(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=2, memory=1.0)
        with self.assertRaises(ValueError):
            f.screen({p: {"s": np.array([5.0])} for p in range(6)})

    def test_group_memory_colluders_cannot_frame_honest_vds(self):
        # 8 台串通的 VD 每輪都交很差的更新：誠實 VD 大半輪次會和其中一台同組，組分數平均也偏高。
        # 只看「平均超過 memory」會把大批誠實 VD 永久排除；要和所有 VD 的平均相比也明顯突出才排除
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(24)} | {p: {"s": np.array([30.0])} for p in range(24, 32)}
        ids = {p: f"vd{p}" for p in ups}
        rounds = [f.screen(ups, ids) for _ in range(14)]
        always = set(range(24)).intersection(*rounds[-4:])
        self.assertLessEqual(len(always), 2, always)

    def test_group_memory_rejects_every_copy_of_a_duplicated_vd(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(17)}
        ids = {p: f"vd{p}" for p in range(16)} | {16: "vd3"}
        out = f.screen(ups, ids)
        self.assertTrue({3, 16} <= out)
        self.assertEqual(len(f.history["vd3"]), 1)  # 同一台同一輪只記一筆

    def test_group_memory_keeps_an_excluded_vd_out_when_few_updates_arrive(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([30.0])}}
        ids = {p: f"vd{p}" for p in ups}
        for _ in range(10):
            f.screen(ups, ids)
        few = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(5)} | {19: {"s": np.array([5.02])}}
        self.assertIn(19, f.screen(few, ids))  # 只來 6 份、不夠分組時，已排除的 VD 也不能混回來

    def test_group_memory_lets_an_excluded_vd_back_after_a_while(self):
        # 排除不是永久的：一段時間後清掉紀錄重新觀察，誠實 VD 不會因為某幾輪運氣差就再也回不來
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([30.0])}}
        ids = {p: f"vd{p}" for p in ups}
        for _ in range(10):
            f.screen(ups, ids)
        ups[19] = {"s": np.array([5.0])}
        later = [f.screen(ups, ids) for _ in range(22)]
        self.assertFalse(all(19 in r for r in later[-4:]))

    def test_group_memory_catches_a_returning_attacker_quickly(self):
        # 放回來後還是交壞更新：不用再等 8 輪，3 輪就再排除，而且這次排除更久
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([30.0])}}
        ids = {p: f"vd{p}" for p in ups}
        for _ in range(8):
            f.screen(ups, ids)
        f.screen(ups, ids)
        until = f.banned["vd19"]
        for _ in range(until - f._round + 3):
            f.screen(ups, ids)
        self.assertGreater(f.banned.get("vd19", 0), until + 16)

    def test_group_with_mismatched_update_rejects_only_that_vd(self):
        # 參數名稱或形狀是公開資訊，分組前就能個別擋掉；不能讓一份格式不對的更新拖垮整組誠實 VD
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0])} for p in range(15)} | {15: {"x": np.array([5.0])}}
        ids = {p: f"vd{p}" for p in ups}
        self.assertEqual([f.screen(ups, ids) for _ in range(3)], [{15}] * 3)

    def test_group_memory_bans_the_vd_whose_groups_keep_breaking(self):
        # 格式正確但讓模型輸出變 nan 的更新：組分數算不出來，同組每個人都記一筆最可疑，攻擊者每輪都在，終究被單獨排除
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([np.nan])}}
        ids = {p: f"vd{p}" for p in ups}
        rounds = [f.screen(ups, ids) for _ in range(10)]
        self.assertEqual(rounds[-1], {19})

    def test_group_memory_forgets_strikes_after_a_clean_stretch(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, group=4, memory=1.5, seed=0)
        ups = {p: {"s": np.array([5.0 + 0.2 * (p % 5)])} for p in range(19)} | {19: {"s": np.array([30.0])}}
        ids = {p: f"vd{p}" for p in ups}
        for _ in range(9):
            f.screen(ups, ids)
        ups[19] = {"s": np.array([5.0])}
        for _ in range(40):
            f.screen(ups, ids)
        self.assertNotIn("vd19", f._strikes)
        self.assertLessEqual(max(len(h) for h in f.history.values()), 32)  # 歷史只留最近幾輪，不會無限長

    def test_secret_grouping_cannot_be_predicted_from_the_config_seed(self):
        # 分組亂數若由設定檔的 seed 決定，攻擊者能算出自己和誰同組；secret=True 改用作業系統的亂數
        ups = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(16)}
        ids = {p: p for p in ups}

        def groups(f):
            f.screen(ups, ids)
            return tuple(sorted(f.last_scores.items()))
        same = [groups(ValidationFilter(lambda a: float(a["s"][0]), group=4, memory=1.5, seed=7)) for _ in range(2)]
        self.assertEqual(same[0], same[1])
        diff = {groups(ValidationFilter(lambda a: float(a["s"][0]), group=4, memory=1.5, seed=7, secret=True)) for _ in range(5)}
        self.assertGreater(len(diff), 1)

    def test_strata_compare_each_vd_only_with_its_own_kind_of_road(self):
        # 0～3 號是幾乎不塞車的路段，參考資料的塞車時段本來就預測得比較差（約 6.0）；4～17 號常塞車（約 5.0）
        free = {p: 6.0 + 0.01 * (p % 3) for p in range(4)}
        jam = {p: 5.0 + 0.01 * (p % 3) for p in range(4, 18)}
        ups = {p: {"s": np.array([s])} for p, s in (free | jam).items()}
        ids = {p: 100 + p for p in ups}  # 訊息鍵與 VD 編號不同：分層要看 VD 編號
        strata = {100 + p: ("free" if p < 4 else "jam") for p in ups}
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        self.assertEqual(f.screen(ups, ids), set(range(4)))  # 不分層：整段順暢路段都被當成可疑
        g = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, strata=strata)
        self.assertEqual(g.screen(ups, ids), set())
        # 常塞車路段裡出現藏塞車的更新（分數像順暢路段），分層後反而更明顯
        ups[17] = {"s": np.array([6.0])}
        self.assertEqual(g.screen(ups, ids), {17})

    def test_too_small_stratum_is_checked_against_everyone(self):
        # 這輪某類只來了 2 份（太少無法自己判斷）：不能就此不檢查，要併回全部一起比
        ups = {p: {"s": np.array([5.0 + 0.01 * p])} for p in range(10)} | {10: {"s": np.array([5.0])}, 11: {"s": np.array([50.0])}}
        strata = {p: ("small" if p >= 10 else "big") for p in ups}
        g = ValidationFilter(lambda a: float(a["s"][0]), z=3.0, strata=strata)
        self.assertEqual(g.screen(ups, {p: p for p in ups}), {11})

    def test_strata_need_vd_ids_and_cannot_be_combined_with_groups(self):
        g = ValidationFilter(lambda a: float(a["s"][0]), strata={0: "a"})
        with self.assertRaises(ValueError):
            g.screen({p: {"s": np.array([5.0])} for p in range(5)})
        with self.assertRaises(ValueError):
            ValidationFilter(lambda a: 0.0, strata={0: "a"}, group=2)

    def test_memory_off_by_default(self):
        f = ValidationFilter(lambda a: float(a["s"][0]), z=3.0)
        base = [5.0, 5.1, 4.9, 5.05, 4.95, 5.0, 5.1, 4.9, 5.0]
        for _ in range(5):
            bad = f.screen({p: {"s": np.array([v])} for p, v in enumerate(base)} | {9: {"s": np.array([5.25])}},
                           {p: p for p in range(10)})
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

        # 開跨輪記憶時：誠實更新被原封不動重送一份，不能讓它被當成「同編號送兩份」而拒絕
        seen.clear()
        g.validation_filter = ValidationFilter(lambda a: float(a["s"][0]), memory=1.5)
        g.aggregate_train(3, [reply(p, 5.0 + 0.01 * p) for p in range(6)] + [reply(2, 5.02)])
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4, 5])
        self.assertEqual(stats["val_rejected"], "")


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
        both = jam_scorer(refs, lambda: Shift(0.0), free=True)
        self.assertEqual(len(both({"b": np.array(1.0, np.float32)})), 2)
        j, f = both({"b": np.array(1.0, np.float32)})
        self.assertAlmostEqual(j, 20.0, places=3)
        self.assertAlmostEqual(f, 20.0, places=3)  # 100、90 km/h 這兩筆是順暢時段
        # 參考資料完全沒有塞車樣本：塞車分數退回全部樣本，順暢分數就沒有意義，回傳 nan（不給寬容也不加懷疑）
        flat = [{"X": X[2:], "y": y[2:], "mean": mean, "std": std}]
        j, f = jam_scorer(flat, lambda: Shift(0.0), free=True)({"b": np.array(1.0, np.float32)})
        self.assertAlmostEqual(j, 20.0, places=3)
        self.assertTrue(np.isnan(f))


if __name__ == "__main__":
    unittest.main()
