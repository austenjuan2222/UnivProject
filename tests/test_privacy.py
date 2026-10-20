import math
import unittest

from tests._opacus_rdp import ensure_opacus_rdp

HAVE_RDP = ensure_opacus_rdp()
if HAVE_RDP:
    from privacy import append_history, compute_epsilon


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def independent_int_order_epsilon(history, delta, orders=range(2, 257)):
    """獨立實作：整數階 SGM 的 RDP（Mironov et al. 2019 二項式展開）+ Balle 2020 轉換。"""
    best = math.inf
    for a in orders:
        rdp = 0.0
        for sigma, q, steps in history:
            if q == 1.0:
                rdp += steps * a / (2 * sigma**2)
                continue
            terms = [
                _log_comb(a, k) + (a - k) * math.log1p(-q) + k * math.log(q)
                + (k * k - k) / (2 * sigma**2)
                for k in range(a + 1)
            ]
            m = max(terms)
            log_a = m + math.log(sum(math.exp(t - m) for t in terms))
            rdp += steps * log_a / (a - 1)
        eps = rdp - (math.log(delta) + math.log(a)) / (a - 1) + math.log((a - 1) / a)
        best = min(best, eps)
    return best


@unittest.skipUnless(HAVE_RDP, "需要 opacus 或其原始碼（OPACUS_SRC）")
class ComputeEpsilonTest(unittest.TestCase):
    def test_no_training_means_zero_epsilon(self):
        self.assertEqual(compute_epsilon([], delta=1e-5), 0.0)

    def test_matches_independent_integer_order_computation(self):
        hist = [(1.1, 0.25, 4 * 3)]  # σ=1.1、每輪 4 步、3 輪
        ours = compute_epsilon(hist, delta=1e-5)
        ref = independent_int_order_epsilon(hist, delta=1e-5)
        # Opacus 另含分數階，所以結果 <= 整數階版本，且差距很小
        self.assertLessEqual(ours, ref + 1e-9)
        self.assertLess(ref - ours, 0.05 * ref)

    def test_full_batch_closed_form(self):
        hist = [(2.0, 1.0, 10)]
        ours = compute_epsilon(hist, delta=1e-5)
        ref = independent_int_order_epsilon(hist, delta=1e-5)
        self.assertLessEqual(ours, ref + 1e-9)
        self.assertLess(ref - ours, 0.05 * ref)

    def test_epsilon_accumulates_over_rounds(self):
        hist, eps = None, []
        for _ in range(5):
            hist = append_history(hist, sigma=1.1, q=0.25, steps=4)
            eps.append(compute_epsilon(list(zip(hist["sigma"], hist["q"], hist["steps"])), 1e-5))
        self.assertTrue(all(b > a for a, b in zip(eps, eps[1:])), eps)
        one_shot = compute_epsilon([(1.1, 0.25, 20)], 1e-5)
        self.assertAlmostEqual(eps[-1], one_shot, places=9)

    def test_append_history_merges_identical_settings(self):
        h = append_history(None, sigma=1.1, q=0.25, steps=4)
        h = append_history(h, sigma=1.1, q=0.25, steps=4)
        h = append_history(h, sigma=0.8, q=0.25, steps=4)
        self.assertEqual(h["steps"], [8, 4])
        self.assertEqual(h["sigma"], [1.1, 0.8])

    def test_more_noise_gives_smaller_epsilon(self):
        self.assertLess(compute_epsilon([(2.0, 0.1, 100)], 1e-5),
                        compute_epsilon([(1.0, 0.1, 100)], 1e-5))


if __name__ == "__main__":
    unittest.main()
