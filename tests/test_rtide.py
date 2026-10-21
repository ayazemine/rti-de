"""Tests for python/RTIDE/optimizer.py and the reference implementations of DE and L-SHADE."""
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "experiments")                 # run_experiments.py, benchmarks/
for _p in (os.path.join(REPO_ROOT, "python"), ROOT, os.path.join(ROOT, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from RTIDE import rti_de  # noqa: E402
from de_reference import de  # noqa: E402
from lshade_reference import lshade  # noqa: E402


def _rastrigin(X):
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


def _shifted_sphere(X):
    return 500.0 + np.sum((X - 1.5) ** 2, axis=1)


class Counted:
    def __init__(self, f):
        self.f, self.n, self.X = f, 0, []

    def __call__(self, X):
        self.n += X.shape[0]
        self.X.append(np.array(X))
        return self.f(X)


ALGOS = {"rti_de": rti_de, "de": de, "lshade": lshade}


class TestBudgetAndBounds(unittest.TestCase):
    def test_exact_budget_and_points_inside_the_box(self):
        lb, ub = np.full(8, -5.0), np.full(8, 5.0)
        for name, fn in ALGOS.items():
            for max_fes in (1000, 2357):
                with self.subTest(algo=name, max_fes=max_fes):
                    c = Counted(_rastrigin)
                    x, f, fes = fn(c, lb, ub, max_fes, seed=0)
                    self.assertEqual((c.n, fes), (max_fes, max_fes))
                    X = np.vstack(c.X)
                    self.assertTrue(np.all(X >= lb) and np.all(X <= ub))
                    self.assertEqual(f, float(_rastrigin(x[None])[0]))
                    self.assertEqual(f, min(float(_rastrigin(b).min()) for b in c.X))

    def test_same_seed_same_run(self):
        lb, ub = np.full(6, -5.0), np.full(6, 5.0)
        for name, fn in ALGOS.items():
            with self.subTest(algo=name):
                a, b = fn(_rastrigin, lb, ub, 3000, seed=3), fn(_rastrigin, lb, ub, 3000, seed=3)
                self.assertTrue(np.array_equal(a[0], b[0]) and a[1] == b[1])
                self.assertFalse(np.array_equal(a[0], fn(_rastrigin, lb, ub, 3000, seed=4)[0]))


class TestRTIDE(unittest.TestCase):
    lb, ub = np.full(10, -5.0), np.full(10, 5.0)

    def test_converges_on_a_sphere(self):
        _, f, _ = rti_de(_shifted_sphere, self.lb, self.ub, 20000, seed=0)
        self.assertLess(f - 500.0, 1e-6)

    def test_chain_is_inactive_before_chain_from(self):
        """With chain_from = 1 the chain is never used: the run equals the run without chain."""
        a = rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1, chain_from=1.0)
        b = rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1, chain=0)
        self.assertTrue(np.array_equal(a[0], b[0]) and a[1] == b[1])

    def test_chain_changes_the_run(self):
        full = rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1)
        for kw in (dict(chain=0), dict(chain=1), dict(chain_from=0.0)):
            with self.subTest(**kw):
                self.assertFalse(np.array_equal(full[0], rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1, **kw)[0]))

    def test_switches_change_the_run(self):
        full = rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1)
        for kw in (dict(lpsr=False), dict(adapt=False), dict(CR=None), dict(CR=0.5)):
            with self.subTest(**kw):
                self.assertFalse(np.array_equal(full[0], rti_de(_rastrigin, self.lb, self.ub, 6000, seed=1, **kw)[0]))

    def test_constant_population_without_lpsr(self):
        c = Counted(_rastrigin)
        rti_de(c, self.lb, self.ub, 3000, seed=0, n_init=40, lpsr=False)
        self.assertEqual({b.shape[0] for b in c.X[:-1]}, {40})

    def test_reference_run(self):
        """The default algorithm must not change: stored results of two reference runs
        (CEC2017 F12, D = 30, 300,000 evaluations, seed 8), with CR = 0.9 and without crossover.

        The values were stored on Windows (MinGW build of the CEC library) and are bit-exact only
        there: the C math library of another platform rounds differently in the last bit, and
        the run amplifies that. Noise of one ulp on the objective values moved the two results
        within [2203.875, 2203.897] and [2712.22, 2712.35]; elsewhere the tolerance is 1e-3 relative."""
        try:
            sys.path.insert(0, ROOT)
            import run_experiments as rx
            obj = rx.load_objective("cec2017", "F12", 30)
        except Exception as e:                       # CEC library not built
            self.skipTest(f"CEC2017 functions not available: {e!r}")
        for kw, ref in ((dict(), 2203.883115060296), (dict(CR=None), 2712.3308479876405)):
            with self.subTest(**kw):
                f = rti_de(obj["vec"], obj["lb"], obj["ub"], 300000, seed=8, **kw)[1]
                if os.name == "nt":
                    self.assertEqual(f, ref)
                else:
                    self.assertAlmostEqual(f, ref, delta=1e-3 * ref)

    def test_trace_does_not_change_the_run_and_is_consistent(self):
        max_fes, n_init = 6000, 180
        plain = rti_de(_rastrigin, self.lb, self.ub, max_fes, seed=1)
        log = []
        traced = rti_de(_rastrigin, self.lb, self.ub, max_fes, seed=1, trace=log)
        self.assertTrue(np.array_equal(plain[0], traced[0]) and plain[1] == traced[1])
        ev = np.array([g["evaluated"] for g in log])
        acc = np.array([g["accepted"] for g in log])
        imp = np.array([g["improved"] for g in log])
        fes = np.array([g["fes"] for g in log])
        self.assertEqual(int(ev.sum()), max_fes - n_init)            # one candidate per evaluation
        self.assertTrue(np.array_equal(np.diff(np.r_[n_init, fes]), ev.sum(axis=1)))
        self.assertEqual(fes[-1], max_fes)
        self.assertTrue(np.all(imp <= acc) and np.all(acc <= ev))
        self.assertEqual(log[-1]["f_best"], plain[1])
        self.assertTrue(np.all(np.diff([g["f_best"] for g in log]) <= 0))
        # the chain is used only after chain_from of the budget
        before = fes < 0.7 * max_fes
        self.assertEqual(int(ev[before][:, 1:].sum()), 0)
        self.assertGreater(int(ev[:, 1].sum()), 0)
        self.assertGreater(int(ev[:, 2].sum()), 0)
        # third points: one factor per evaluated point, all in [-1, 1/4) and not near 0
        theta = np.concatenate([g["theta"] for g in log])
        self.assertEqual(theta.size, int(ev[:, 2].sum()))
        self.assertEqual(int(np.concatenate([g["theta_accepted"] for g in log]).sum()), int(acc[:, 2].sum()))
        self.assertTrue(np.all(theta >= -1.0) and np.all(theta < 0.25) and np.all(np.abs(theta) >= 0.02))
        # every half step that is rejected leads to a third point or to a skipped one
        rejected_half = int((ev[:, 1] - acc[:, 1]).sum())
        third_or_skipped = int(ev[:, 2].sum()) + sum(g["skipped"] for g in log)
        self.assertLessEqual(third_or_skipped, rejected_half)        # agents removed by the reduction are lost
        self.assertGreaterEqual(third_or_skipped, rejected_half - n_init)

    def test_trace_summary_of_the_harness(self):
        sys.path.insert(0, ROOT)
        import run_experiments as rx
        max_fes, log = 6000, []
        rti_de(_rastrigin, self.lb, self.ub, max_fes, seed=1, trace=log)
        s = rx.summarize_rtide_trace(log, max_fes, bins=20)
        for key in ("evaluated", "accepted", "improved"):
            self.assertTrue(np.array_equal(s[key].sum(axis=0), np.sum([g[key] for g in log], axis=0)), key)
        self.assertEqual(int(s["generations"].sum()), len(log))
        self.assertEqual(int(s["skipped"].sum()), sum(g["skipped"] for g in log))
        self.assertEqual((s["mu_F"][-1], s["n"][-1]), (log[-1]["mu_F"], log[-1]["n"]))
        self.assertEqual(s["theta_outside"], 0)
        self.assertEqual(int(s["theta_accepted"].sum() + s["theta_rejected"].sum()),
                         int(s["evaluated"][:, 2].sum()))
        self.assertEqual(int(s["evaluated"][:14, 1:].sum()), 0)       # parts that end before 0.7 of the budget

    def test_nan_objective_is_treated_as_infeasible(self):
        def f(X):
            v = _rastrigin(X)
            v[X[:, 0] > 4.0] = np.nan
            return v
        x, fx, _ = rti_de(f, self.lb, self.ub, 3000, seed=0)
        self.assertTrue(np.isfinite(fx) and x[0] <= 4.0)


if __name__ == "__main__":
    unittest.main()
