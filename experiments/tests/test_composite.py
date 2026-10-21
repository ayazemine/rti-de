"""Validation tests for applications/composite/laminate_buckling.py.

Run from experiments/:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "applications", "composite"))
import laminate_buckling as lb  # noqa: E402


class TestPublishedValues(unittest.TestCase):
    def test_reproduces_ahmid_2019_table3(self):
        """All four published sequences reproduce their λ_cb to the printed 2 decimals."""
        for genes, published in lb.PUBLISHED:
            with self.subTest(sequence=lb.describe(genes)):
                self.assertAlmostEqual(float(lb.buckling_load_factor(genes)), published, delta=0.005)


class TestModel(unittest.TestCase):
    def test_batch_equals_single(self):
        G = np.random.RandomState(0).randint(0, 3, (50, lb.DIM))
        batch = lb.buckling_load_factor(G)
        single = np.array([lb.buckling_load_factor(g) for g in G])
        np.testing.assert_allclose(batch, single, rtol=1e-12)

    def test_decode_bounds(self):
        np.testing.assert_array_equal(lb.decode([0.0, 0.999, 1.0, 2.5, 3.0]), [0, 0, 1, 2, 2])

    def test_objective_is_negative_lambda(self):
        x = np.array(lb.PUBLISHED[0][0], dtype=float) + 0.5
        self.assertAlmostEqual(lb.objective(x), -float(lb.buckling_load_factor(lb.PUBLISHED[0][0])))

    def test_quasi_isotropic_symmetry(self):
        """Swapping all 0_2 and 90_2 stacks swaps D11 and D22 (checks the Q̄ transform)."""
        g = np.array([0, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2, 1, 0, 2, 1, 0])
        swapped = np.where(g == 0, 2, np.where(g == 2, 0, 1))
        d, ds = lb.bending_stiffness(g), lb.bending_stiffness(swapped)
        self.assertAlmostEqual(d[0], ds[2], places=6)
        self.assertAlmostEqual(d[1], ds[1], places=6)
        self.assertAlmostEqual(d[3], ds[3], places=6)


class TestReferenceOptimum(unittest.TestCase):
    """reference_optimum.json (exhaustive search) must agree with the model."""

    @classmethod
    def setUpClass(cls):
        import json
        path = os.path.join(os.path.dirname(lb.__file__), "reference_optimum.json")
        with open(path, encoding="utf-8") as fh:
            cls.ref = json.load(fh)

    def test_stored_optima_are_reproduced(self):
        for key in ("free", "contiguity"):
            r = self.ref[key]
            self.assertAlmostEqual(float(lb.buckling_load_factor(r["genes"])), r["lambda_cb"], places=8)
            self.assertEqual(lb.describe(r["genes"]), r["sequence"])
        self.assertEqual(lb.contiguity_violation(self.ref["contiguity"]["genes"]), 0)
        self.assertLessEqual(self.ref["contiguity"]["lambda_cb"], self.ref["free"]["lambda_cb"])

    def test_values_quoted_in_the_paper(self):
        self.assertAlmostEqual(self.ref["free"]["lambda_cb"], 3973.0137, places=4)
        self.assertAlmostEqual(self.ref["contiguity"]["lambda_cb"], 3953.2830, places=4)

    def test_no_published_design_beats_the_optimum(self):
        for genes, _ in lb.PUBLISHED:
            self.assertLessEqual(float(lb.buckling_load_factor(genes)), self.ref["free"]["lambda_cb"] + 1e-9)


class TestPenalty(unittest.TestCase):
    def test_penalty_per_violation(self):
        self.assertEqual(lb.PENALTY, 1000.0)
        g = [2, 2, 2] + [1] * (lb.DIM - 3)                       # 2 violations
        x = np.array(g, dtype=float) + 0.5
        lam = float(lb.buckling_load_factor(g))
        self.assertAlmostEqual(float(np.ravel(lb.objective(x))[0]), -lam)
        self.assertAlmostEqual(float(np.ravel(lb.objective(x, contiguity=True))[0]), -lam + 2 * lb.PENALTY)

    def test_feasible_design_has_no_penalty(self):
        x = np.array([1] * lb.DIM, dtype=float) + 0.5
        self.assertEqual(float(np.ravel(lb.objective(x))[0]), float(np.ravel(lb.objective(x, contiguity=True))[0]))


class TestContiguity(unittest.TestCase):
    def test_all_45_is_feasible(self):
        self.assertEqual(lb.contiguity_violation([1] * lb.DIM), 0)

    def test_three_stacks_violate(self):
        """Six contiguous 90° plies at the surface; the symmetric laminate has
        this block on both outer surfaces, so it counts twice."""
        g = [2, 2, 2] + [1] * (lb.DIM - 3)
        self.assertEqual(lb.contiguity_violation(g), 2)

    def test_midplane_counts(self):
        g = [1] * (lb.DIM - 2) + [0, 0]             # 0_2 0_2 | 0_2 0_2 across the mid-plane
        self.assertEqual(lb.contiguity_violation(g), 2)


if __name__ == "__main__":
    unittest.main()
