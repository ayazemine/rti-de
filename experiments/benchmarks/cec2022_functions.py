#!/usr/bin/env python3
"""
CEC2022 Benchmark Functions (F1–F12) — the organisers' C code.

The functions are evaluated by the official ``cec22_test_func.cpp`` through
``cec_official_functions.py`` (ctypes); they are not re-implemented here.
opfunu 1.0.4 is not used: its cec2022 module ships the official data but its
values differ from the official code for 11 of the 12 functions at D = 10 and
20 (only F2 agrees). This module adds:

* one place listing each function's official name, category, F* and bounds;
* strict dimension checking;
* the evaluation settings of the CEC2022 technical report.

Each object returned by ``get_all_cec2022_functions`` has ``.evaluate(x)``,
``.lb``, ``.ub``, ``.f_bias``, ``.f_global``, ``.x_global`` and ``.ndim``
(``cec_official_functions.OfficialFunction``).

Reference
---------
A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, and P. N. Suganthan,
"Problem Definitions and Evaluation Criteria for the CEC 2022 Special Session
and Competition on Single Objective Bound Constrained Numerical Optimization,"
Nanyang Technological University, Singapore, Tech. Rep., Dec. 2021.
https://github.com/P-N-Suganthan/2022-SO-BO
"""

from collections import namedtuple

import numpy as np

__all__ = [
    "CEC2022_INFO",
    "SUPPORTED_DIMS",
    "PLOT_2D_FUNCTIONS",
    "SEARCH_RANGE",
    "MAX_FES",
    "RUNS_PER_PROBLEM",
    "get_cec2022_function",
    "get_all_cec2022_functions",
]

# ---------------------------------------------------------------------------
# Evaluation settings from the technical report (Section 2.1)
# ---------------------------------------------------------------------------

SUPPORTED_DIMS = (10, 20)
SEARCH_RANGE = (-100.0, 100.0)
MAX_FES = {10: 200_000, 20: 1_000_000}
RUNS_PER_PROBLEM = 30

# Functions defined at D = 2 (for landscape plots only).
# F6–F8 are hybrid functions that split the variables into N = 3, 6 and 5
# groups, so they are not defined for D = 2.
PLOT_2D_FUNCTIONS = ("F1", "F2", "F3", "F4", "F5", "F9", "F10", "F11", "F12")

# ---------------------------------------------------------------------------
# Function table (names as in the report's summary table, Section 1.2)
# ---------------------------------------------------------------------------

FuncInfo = namedtuple("FuncInfo", ["fid", "name", "category", "f_star"])

CEC2022_INFO = (
    FuncInfo("F1", "Shifted and full Rotated Zakharov Function", "Unimodal", 300.0),
    FuncInfo("F2", "Shifted and full Rotated Rosenbrock's Function", "Basic", 400.0),
    FuncInfo("F3", "Shifted and full Rotated Expanded Schaffer's f6 Function", "Basic", 600.0),
    FuncInfo("F4", "Shifted and full Rotated Non-Continuous Rastrigin's Function", "Basic", 800.0),
    FuncInfo("F5", "Shifted and full Rotated Levy Function", "Basic", 900.0),
    FuncInfo("F6", "Hybrid Function 1 (N = 3)", "Hybrid", 1800.0),
    FuncInfo("F7", "Hybrid Function 2 (N = 6)", "Hybrid", 2000.0),
    FuncInfo("F8", "Hybrid Function 3 (N = 5)", "Hybrid", 2200.0),
    FuncInfo("F9", "Composition Function 1 (N = 5)", "Composition", 2300.0),
    FuncInfo("F10", "Composition Function 2 (N = 4)", "Composition", 2400.0),
    FuncInfo("F11", "Composition Function 3 (N = 5)", "Composition", 2600.0),
    FuncInfo("F12", "Composition Function 4 (N = 6)", "Composition", 2700.0),
)

_INFO_BY_ID = {info.fid: info for info in CEC2022_INFO}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_cec2022_function(fid, ndim=20, allow_2d=False):
    """Return the official function object for one CEC2022 function.

    Parameters
    ----------
    fid : str
        Function id, "F1" … "F12".
    ndim : int
        Dimension. Must be 10 or 20 (the report's settings). D = 2 is allowed
        only with ``allow_2d=True`` and only for ``PLOT_2D_FUNCTIONS``.
    allow_2d : bool
        Permit D = 2 for landscape plots. Never use D = 2 for benchmarking.

    Raises
    ------
    ValueError
        For an unknown id or an unsupported dimension.
    """
    if fid not in _INFO_BY_ID:
        raise ValueError(f"Unknown CEC2022 function id {fid!r}; expected F1–F12.")
    if ndim == 2 and allow_2d:
        if fid not in PLOT_2D_FUNCTIONS:
            raise ValueError(f"CEC2022 {fid} is a hybrid function and is not defined for D = 2.")
    elif ndim not in SUPPORTED_DIMS:
        raise ValueError(f"CEC2022 is defined for D in {SUPPORTED_DIMS}, got D = {ndim}.")

    from cec_official_functions import get_official_function

    return get_official_function("cec2022", fid, ndim)


def get_all_cec2022_functions(ndim=20):
    """Return a list of (fid, func_object) for F1–F12 at D = 10 or 20."""
    return [(info.fid, get_cec2022_function(info.fid, ndim)) for info in CEC2022_INFO]


# ---------------------------------------------------------------------------
# Self-test: every function evaluates to F* at its documented optimum
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    ndim = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    print(f"CEC2022 (official C code) — D = {ndim}, MaxFES = {MAX_FES.get(ndim, 'n/a')}")
    print("=" * 78)
    ok = True
    for fid, f in get_all_cec2022_functions(ndim):
        info = _INFO_BY_ID[fid]
        err = f.evaluate(f.x_global) - info.f_star
        bounds_ok = np.all(f.lb == SEARCH_RANGE[0]) and np.all(f.ub == SEARCH_RANGE[1])
        passed = abs(err) < 1e-8 and bounds_ok
        ok &= passed
        print(f"{fid:>4} {info.category:<12} F*={info.f_star:>7.0f}  "
              f"f(x*)-F*={err: .1e}  bounds={'ok' if bounds_ok else 'WRONG'}  "
              f"{'PASS' if passed else 'FAIL'}")
    print("=" * 78)
    print("All checks passed." if ok else "SOME CHECKS FAILED.")
    sys.exit(0 if ok else 1)
