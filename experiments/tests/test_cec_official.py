"""Official CEC2017 / CEC2022 functions through ctypes (benchmarks/cec_official_functions.py)."""
import os
import shutil
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import cec_official_functions as co  # noqa: E402

HAVE_LIB = shutil.which("g++") is not None or all(os.path.exists(co._lib_path(s)) for s in co._SUITES)

# Reference values computed with the official C code compiled as a stand-alone program (not through
# this module), D = 10, at x = 0 and at x_i = (-1)^i * 10 * (i + 1), i = 0..9.
POINTS = np.array([np.zeros(10), [(-1) ** i * 10.0 * (i + 1) for i in range(10)]])
REFERENCE = {
    ("cec2017", "F1"): (29975432515.940056, 39147196778.575722),
    ("cec2017", "F5"): (726.71456129591127, 1000.2991911583277),
    ("cec2017", "F9"): (4306.1324978942675, 58335.893990915007),
    ("cec2017", "F15"): (769548252.85083985, 29021036655.373482),
    ("cec2017", "F25"): (4820.812334105729, 9434.8643282761768),
    ("cec2017", "F30"): (506077323.00365406, 2252482695.6065025),
    ("cec2022", "F1"): (15908044999.492702, 6449409470.0264235),
    ("cec2022", "F3"): (741.77549410442805, 815.64739557413907),
    ("cec2022", "F8"): (87756.646127370987, 895036940.27836049),
    ("cec2022", "F12"): (4978.8884425246797, 5401.0420431187595),
}


@unittest.skipUnless(HAVE_LIB, "g++ is needed to compile the official CEC code")
class TestOfficialCEC(unittest.TestCase):

    def test_reference_values(self):
        for (suite, fid), ref in REFERENCE.items():
            f = co.get_official_function(suite, fid, 10)
            np.testing.assert_allclose(f.evaluate_batch(POINTS), ref, rtol=1e-12, err_msg=f"{suite} {fid}")

    def test_optimum(self):
        for suite, dims, fids in (("cec2022", (10, 20), [f"F{k}" for k in range(1, 13)]),
                                  ("cec2017", (10, 100), co.CEC2017_LABELS)):
            for d in dims:
                for fid in fids:
                    f = co.get_official_function(suite, fid, d)
                    self.assertTrue(np.all(np.abs(f.x_global) <= 100.0), f"{suite} {fid} D={d}")
                    self.assertLess(abs(f.evaluate(f.x_global) - f.f_bias), 1e-6, f"{suite} {fid} D={d}")

    def test_switching_functions_and_batch(self):
        X = np.random.default_rng(0).uniform(-100, 100, (5, 10))
        a = co.get_official_function("cec2017", "F5", 10)
        b = co.get_official_function("cec2017", "F21", 10)
        c = co.get_official_function("cec2022", "F7", 10)
        va, vb, vc = a.evaluate_batch(X), b.evaluate_batch(X), c.evaluate_batch(X)
        for f, v in ((c, vc), (a, va), (b, vb), (a, va)):          # interleaved: data must be reloaded
            np.testing.assert_array_equal([f.evaluate(x) for x in X], v)

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            co.get_official_function("cec2017", "F2", 10)
        with self.assertRaises(ValueError):
            co.get_official_function("cec2017", "F1", 20)
        with self.assertRaises(ValueError):
            co.get_official_function("cec2022", "F6", 2)


if __name__ == "__main__":
    unittest.main()
