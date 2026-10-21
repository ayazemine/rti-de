"""Tests for applications/uav/path_planning.py (port of Phung & Ha's MATLAB code).

Run from experiments/:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "applications", "uav"))
import path_planning as pp  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_matlab_round(self):
        np.testing.assert_array_equal(pp.matlab_round([2.5, 3.5, -2.5, 1.49, 0.5]), [3, 4, -3, 1, 1])

    def test_dist_point_segment(self):
        self.assertAlmostEqual(pp.dist_point_segment((0, 1), (-1, 0), (1, 0)), 1.0)
        self.assertAlmostEqual(pp.dist_point_segment((3, 0), (-1, 0), (1, 0)), 2.0)
        self.assertAlmostEqual(pp.dist_point_segment((3, 4), (0, 0), (0, 0)), 5.0)

    def test_terrain_loaded(self):
        self.assertEqual(pp.H.shape, (879, 1045))
        self.assertGreaterEqual(pp.H.min(), 0.0)

    def test_search_space(self):
        self.assertEqual(pp.DIM, 30)
        self.assertAlmostEqual(pp.R_MAX, 2 * np.hypot(600, 700) / 10)
        self.assertTrue(np.all(pp.LB < pp.UB))


class TestCost(unittest.TestCase):
    def test_zero_length_segments(self):
        """r = 0: every node stays at the start; J1 is the start→end 3-D distance."""
        x = np.concatenate([np.zeros(10), np.zeros(10), np.full(10, pp.PHI0)])
        nx, ny, nz = pp.spherical_to_cart(x)
        np.testing.assert_allclose(nx, 200.0)
        np.testing.assert_allclose(ny, 100.0)
        np.testing.assert_allclose(nz, 150.0)
        J1 = pp.cost_components(x)[0]
        dz = pp.terrain_height(800.0, 800.0) - pp.terrain_height(200.0, 100.0)
        self.assertAlmostEqual(J1, float(np.sqrt(600**2 + 700**2 + dz**2)), places=6)

    def test_collision_is_infinite(self):
        """Heading straight at threat 4 (350, 200) from the start → collision → inf."""
        phi = np.arctan2(350 - 200, 200 - 100)           # angle from the y-axis, as in the code
        x = np.concatenate([np.full(10, pp.R_MAX), np.zeros(10), np.full(10, phi)])
        self.assertTrue(np.isinf(pp.cost(x)))

    def test_nodes_clipped(self):
        x = pp.UB.copy()
        nx, ny, nz = pp.spherical_to_cart(x)
        self.assertTrue(np.all((nx >= pp.XMIN) & (nx <= pp.XMAX)))
        self.assertTrue(np.all((ny >= pp.YMIN) & (ny <= pp.YMAX)))
        self.assertTrue(np.all((nz >= pp.ZMIN) & (nz <= pp.ZMAX)))

    def test_vectorized_matches_scalar(self):
        rng = np.random.RandomState(1)
        X = pp.LB + rng.rand(20, pp.DIM) * (pp.UB - pp.LB)
        np.testing.assert_array_equal(pp.objective(X), np.array([pp.cost(r) for r in X]))


if __name__ == "__main__":
    unittest.main()
