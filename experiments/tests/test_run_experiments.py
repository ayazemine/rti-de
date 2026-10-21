"""Tests for the statistics, FE accounting and safeguards of run_experiments.py."""
import argparse
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_experiments as rx  # noqa: E402

warnings.filterwarnings("ignore")


def _sphere(X):
    return np.sum(np.asarray(X, float) ** 2, axis=-1)


class TestVerdict(unittest.TestCase):
    def test_not_significant(self):
        self.assertEqual(rx._verdict(0.2, np.arange(5.0), np.arange(5.0) + 1), "≈")
        self.assertEqual(rx._verdict(np.nan, np.zeros(5), np.zeros(5)), "≈")

    def test_direction(self):
        a, b = np.arange(10.0), np.arange(10.0) + 5
        self.assertEqual(rx._verdict(0.001, a, b), "+")          # a lower = better
        self.assertEqual(rx._verdict(0.001, b, a), "-")

    def test_equal_medians(self):
        """Equal medians (both 0 after the 1e-8 threshold): the direction follows the test statistic."""
        a = np.zeros(30)
        b = np.r_[np.zeros(20), np.linspace(1e-6, 1e-3, 10)]
        self.assertEqual(np.median(a), np.median(b))
        p = rx._mannwhitney(a, b)
        self.assertLess(p, 0.05)
        self.assertEqual(rx._verdict(p, a, b), "+")
        self.assertEqual(rx._verdict(p, b, a), "-")

    def test_infeasible_runs(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = np.array([np.inf, np.inf, np.inf, 4.5, 5.5, 6.5])
        self.assertEqual(rx._verdict(0.01, a, b), "+")
        self.assertEqual(rx._verdict(0.01, b, a), "-")


class TestMannWhitney(unittest.TestCase):
    def test_identical_samples(self):
        self.assertTrue(np.isnan(rx._mannwhitney(np.ones(10), np.ones(10))))

    def test_inf_is_worst_value(self):
        a = np.arange(1.0, 11.0)
        self.assertAlmostEqual(rx._mannwhitney(a, np.full(10, np.inf)), rx._mannwhitney(a, np.full(10, 1e6)))

    def test_matches_scipy(self):
        from scipy import stats
        rng = np.random.default_rng(0)
        a, b = rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
        self.assertEqual(rx._mannwhitney(a, b), float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))

    def test_order_of_the_runs_does_not_matter(self):
        """The runs of two algorithms are independent samples, not pairs."""
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 30), rng.normal(0.4, 1, 30)
        self.assertEqual(rx._mannwhitney(a, b), rx._mannwhitney(a, rng.permutation(b)))


class TestEffectSizes(unittest.TestCase):
    def test_cliffs_delta(self):
        self.assertEqual(rx._cliffs_delta([1, 2, 3], [4, 5, 6]), -1.0)
        self.assertEqual(rx._cliffs_delta([4, 5, 6], [1, 2, 3]), 1.0)
        self.assertEqual(rx._cliffs_delta([1, 2], [1, 2]), 0.0)

    def test_cohens_d(self):
        self.assertAlmostEqual(rx._cohens_d([0, 1, 2], [1, 2, 3]), -1.0)
        self.assertTrue(np.isnan(rx._cohens_d([0, np.inf], [1, 2])))


class TestCounter(unittest.TestCase):
    def test_counts_and_trace(self):
        c = rx.Counter(_sphere, lambda x: float(_sphere(x)))
        c.vectorized(np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
        c.single(np.array([0.5, 0.0]))
        c.single(np.array([4.0, 0.0]))
        self.assertEqual(c.n, 5)
        self.assertEqual(c.invalid, 0)
        self.assertEqual(c.best, 0.25)
        self.assertEqual(c.trace, [(1, 9.0), (2, 1.0), (4, 0.25)])      # (FE index, new best)

    def test_non_finite_candidates_are_infeasible(self):
        c = rx.Counter(_sphere, lambda x: float(_sphere(x)))
        f = c.vectorized(np.array([[np.nan, 0.0], [1.0, 1.0], [np.inf, 0.0]]))
        np.testing.assert_array_equal(f, [np.inf, 2.0, np.inf])
        self.assertTrue(np.isinf(c.single(np.array([np.nan, 1.0]))))
        self.assertEqual((c.n, c.invalid), (4, 3))


class TestBudget(unittest.TestCase):
    def test_budget_of(self):
        self.assertEqual(rx.budget_of(dict(budget="fe", max_fes=20000, agents=50, iterations=None)), 20000)
        self.assertEqual(rx.budget_of(dict(budget="iter", max_fes=None, agents=50, iterations=1000)), 50050)

    def test_epochs_for_budget(self):
        self.assertEqual(rx.epochs_for_budget(1_000_000, 50), 19_999)
        self.assertEqual(rx.epochs_for_budget(1_000_000, 50, "GWO"), 19_999)
        self.assertEqual(rx.epochs_for_budget(10, 50), 1)

    def _run(self, algo, max_fes, f_star=None, target_stop=False, dim=5, seed=0):
        """One run of ``algo`` on a sphere through rx._optimize; returns the counter."""
        obj = dict(lb=np.full(dim, -5.0), ub=np.full(dim, 5.0))
        cnt = rx.BudgetCounter(_sphere, lambda x: float(_sphere(x)), max_fes,
                               f_star + rx.ERROR_FLOOR if target_stop else None)
        try:
            rx._optimize(dict(algo=algo, seed=seed), obj, cnt, 10, max_fes)
        except rx.TargetReached:
            pass
        return cnt

    def test_every_algorithm_uses_the_whole_budget(self):
        max_fes = 3000
        for algo in rx.ALL_ALGORITHMS + list(rx.VARIANTS):
            if algo in rx.FES_PER_EPOCH:
                continue
            with self.subTest(algo=algo):
                c = self._run(algo, max_fes)
                self.assertGreaterEqual(c.n, max_fes)
                self.assertLess(c.n, max_fes + 10)                # mealpy finishes its epoch at most
                self.assertEqual(c.best_f, float(_sphere(c.best_x)))

    def test_points_after_the_budget_do_not_count(self):
        c = rx.BudgetCounter(_sphere, lambda x: float(_sphere(x)), 3)
        c.vectorized(np.array([[3.0, 0.0], [2.0, 0.0]]))
        c.vectorized(np.array([[1.0, 0.0], [0.1, 0.0]]))          # the fourth evaluation is over the budget
        self.assertEqual((c.n, c.best_f), (4, 1.0))
        np.testing.assert_array_equal(c.best_x, [1.0, 0.0])
        c.single(np.array([0.0, 0.0]))
        self.assertEqual(c.best_f, 1.0)

    def test_target_stop(self):
        c = self._run("LSHADE", 200_000, f_star=0.0, target_stop=True)
        self.assertLess(c.n, 200_000)
        self.assertLessEqual(c.best_f, rx.ERROR_FLOOR)
        with self.assertRaises(rx.TargetReached):
            rx.BudgetCounter(_sphere, None, 10, 1.0).vectorized(np.array([[0.5, 0.5]]))


class TestConfigGuard(unittest.TestCase):
    def _args(self, out, **kw):
        a = dict(out=out, agents=50, iterations=1000, budget="iter", fes={}, seed=0, runs=30,
                 algorithms=list(rx.ALL_ALGORITHMS), workers=1, no_target_stop=False)
        a.update(kw)
        return argparse.Namespace(**a)

    def test_refuses_to_mix_settings(self):
        prov = {"mealpy": "x"}
        with tempfile.TemporaryDirectory() as tmp:
            rx.prepare_suite("composite", 16, self._args(tmp), prov)
            rx.prepare_suite("composite", 16, self._args(tmp), prov)             # same settings: fine
            with self.assertRaises(SystemExit):
                rx.prepare_suite("composite", 16, self._args(tmp, agents=30), prov)
            with self.assertRaises(SystemExit):
                rx.prepare_suite("composite", 16, self._args(tmp, seed=1), prov)

    def test_cec_source_is_locked(self):
        """A CEC result directory without cec_source in config.json cannot be continued."""
        import json
        prov = {"mealpy": "x"}
        with tempfile.TemporaryDirectory() as tmp:
            sdir = rx.prepare_suite("cec2022", 20, self._args(tmp), prov)
            path = os.path.join(sdir, "config.json")
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.assertIn("official C code", cfg["settings"]["cec_source"])
            del cfg["settings"]["cec_source"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            with self.assertRaises(SystemExit):
                rx.prepare_suite("cec2022", 20, self._args(tmp), prov)

    def test_unreadable_result_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = os.path.join(tmp, "a.pkl"), os.path.join(tmp, "b.pkl")
            rx._atomic_dump({"status": "ok"}, good)
            open(bad, "wb").close()
            self.assertTrue(rx._readable_result(good))
            self.assertFalse(rx._readable_result(bad))


if __name__ == "__main__":
    unittest.main()
