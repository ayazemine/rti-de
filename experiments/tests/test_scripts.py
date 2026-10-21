"""Tests for the helper code under scripts/ (table statistics)."""
import os
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")


class TestHolm(unittest.TestCase):
    def setUp(self):
        try:
            import make_tables
        except Exception as e:                       # matplotlib / pandas missing
            self.skipTest(f"make_tables cannot be imported: {e!r}")
        self.holm = make_tables.holm

    def test_known_values(self):
        np.testing.assert_allclose(self.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(self.holm([0.5, 0.9]), [1.0, 1.0])
        np.testing.assert_allclose(self.holm([0.02]), [0.02])

    def test_monotone_and_nan(self):
        out = self.holm([0.001, np.nan, 0.01, 0.01, 0.2])
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2, 3, 4]], [0.004, 0.03, 0.03, 0.2])


class TestBehaviourFigure(unittest.TestCase):
    """The figure of the RTI-DE trace is built from run dicts as run_experiments.py stores them."""

    def setUp(self):
        try:
            import make_figures
            import figlib
        except Exception as e:                       # matplotlib missing
            self.skipTest(f"make_figures cannot be imported: {e!r}")
        self.mf, self.fl = make_figures, figlib

    def _suite(self, with_trace=True):
        import run_experiments as rx
        from RTIDE import rti_de
        max_fes, lb, ub = 6000, np.full(10, -5.0), np.full(10, 5.0)

        def run(seed):
            log = []
            _, f, _ = rti_de(lambda X: np.sum(X ** 2, axis=1), lb, ub, max_fes, seed=seed, trace=log)
            r = dict(f=f, stopped_at_target=False)
            if with_trace:
                r["rtide_trace"] = rx.summarize_rtide_trace(log, max_fes)
            return r

        class S:
            name, dir, fids, n_runs, budget, n_files, settings = "cec2022", ROOT, ["F1", "F2"], 2, max_fes, 4, {}
            runs = {fid: {rx.PROPOSED: [run(s) for s in (k, k + 1)]} for k, fid in enumerate(("F1", "F2"))}
        return S

    def _ctx(self, S):
        class Ctx:
            out = self.fl.Output(ROOT, dry_run=True)
            load = staticmethod(lambda suite: S)
        return Ctx

    def test_figure_is_built(self):
        self.fl.apply_style()
        self.mf.behaviour(self._ctx(self._suite()), "cec2022")

    def test_runs_without_trace_are_refused(self):
        with self.assertRaises(SystemExit):
            self.mf.behaviour(self._ctx(self._suite(with_trace=False)), "cec2022")


if __name__ == "__main__":
    unittest.main()
