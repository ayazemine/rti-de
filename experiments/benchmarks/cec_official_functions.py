#!/usr/bin/env python3
"""
Official CEC2017 and CEC2022 test functions: the organisers' C code, called through ctypes.

The functions are not re-implemented here. ``official_cec/<suite>/`` holds the
unmodified ``cec17_test_func.cpp`` / ``cec22_test_func.cpp`` and their
``input_data`` (see ``official_cec/README.md`` for the sources and checksums).
They are compiled once into ``official_cec/_build/lib<suite>.so`` with ``g++``.

Numbering is the one of the C code: CEC2017 F1, F3-F30 (``func_num`` 2 is the
deleted function) with F* = 100 * func_num; CEC2022 F1-F12 with the F* of
``cec2022_functions.CEC2022_INFO``.

Each object has ``.evaluate(x)``, ``.evaluate_batch(X)``, ``.lb``, ``.ub``,
``.f_bias``, ``.f_global``, ``.x_global``, ``.ndim`` and ``.name``.

The C code keeps the data of one (function, dimension) in global variables and
reads ``input_data/...`` relative to the working directory. This module reloads
the data whenever another function or dimension is evaluated (with a temporary
``chdir``), so objects of different functions can be mixed freely in one
process. It is not thread-safe; use processes.

    python benchmarks/cec_official_functions.py            # build + self-test
    python benchmarks/cec_official_functions.py --build    # build only

References
----------
N. H. Awad, M. Z. Ali, P. N. Suganthan, J. J. Liang, B. Y. Qu, "Problem Definitions and
Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective
Real-Parameter Numerical Optimization," Tech. Rep., NTU Singapore, 2016.
https://github.com/P-N-Suganthan/CEC2017-BoundContrained
A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, P. N. Suganthan, "Problem Definitions and
Evaluation Criteria for the CEC 2022 Special Session and Competition on Single Objective
Bound Constrained Numerical Optimization," Tech. Rep., NTU Singapore, Dec. 2021.
https://github.com/P-N-Suganthan/2022-SO-BO
"""

import ctypes
import hashlib
import os
import subprocess

import numpy as np

__all__ = ["CEC2017_LABELS", "CEC2017_NAMES", "CEC2017_MAX_FES", "get_official_function",
           "build", "source_sha256"]

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_cec")
BUILD = os.path.join(BASE, "_build")

_SUITES = {
    "cec2017": dict(src="cec17_test_func.cpp", entry="cec17_test_func", dims=(10, 30, 50, 100),
                    shuffled=set(range(11, 21)) | {29, 30}),
    "cec2022": dict(src="cec22_test_func.cpp", entry="cec22_test_func", dims=(2, 10, 20),
                    shuffled={6, 7, 8}),
}

CEC2017_LABELS = ["F1"] + [f"F{k}" for k in range(3, 31)]

# Names from Table I of the CEC2017 technical report, keyed by the func_num of the C code
# (the report's table omits the deleted function and numbers the rest 1-29).
_HYBRID_N = (3, 3, 3, 4, 4, 4, 5, 5, 5, 6)
_COMPOSITION_N = (3, 3, 4, 4, 5, 5, 6, 6, 3, 3)
CEC2017_NAMES = {
    1: "Shifted and Rotated Bent Cigar Function",
    3: "Shifted and Rotated Zakharov Function",
    4: "Shifted and Rotated Rosenbrock's Function",
    5: "Shifted and Rotated Rastrigin's Function",
    6: "Shifted and Rotated Expanded Scaffer's F6 Function",
    7: "Shifted and Rotated Lunacek Bi_Rastrigin Function",
    8: "Shifted and Rotated Non-Continuous Rastrigin's Function",
    9: "Shifted and Rotated Levy Function",
    10: "Shifted and Rotated Schwefel's Function",
    **{10 + j: f"Hybrid Function {j} (N={n})" for j, n in enumerate(_HYBRID_N, 1)},
    **{20 + j: f"Composition Function {j} (N={n})" for j, n in enumerate(_COMPOSITION_N, 1)},
}
CEC2017_MAX_FES = {10: 100_000, 30: 300_000, 50: 500_000, 100: 1_000_000}   # 10000 * D (report, Section 2.1)

_LIBS = {}
_LOADED = {}            # suite -> (func_num, ndim) currently held in the C globals


def _suite_dir(suite):
    return os.path.join(BASE, suite)


def source_sha256(suite):
    """SHA-256 of the suite's C source (recorded in the experiment config)."""
    with open(os.path.join(_suite_dir(suite), _SUITES[suite]["src"]), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _lib_path(suite):
    # Windows needs its own build (PE); the .so files in _build/ are the Linux (ELF) ones.
    return os.path.join(BUILD, f"lib{suite}.{'dll' if os.name == 'nt' else 'so'}")


def build(suite, force=False):
    """Compile the suite's shared library if it is missing or older than its sources."""
    cfg = _SUITES[suite]
    src = os.path.join(_suite_dir(suite), cfg["src"])
    wrapper = os.path.join(BASE, "wrapper.cpp")
    out = _lib_path(suite)
    if (not force and os.path.exists(out)
            and os.path.getmtime(out) >= max(os.path.getmtime(src), os.path.getmtime(wrapper))):
        return out
    os.makedirs(BUILD, exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"
    cmd = ["g++", "-O2", "-w", "-shared", "-fPIC", "-Wl,-Bsymbolic", f"-DENTRY={cfg['entry']}",
           "-I", os.path.join(BASE, "stub"), wrapper, src, "-lm", "-o", tmp]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("g++ was not found; it is needed to compile the official CEC code.")
    if res.returncode != 0:
        raise RuntimeError(f"Compiling {src} failed:\n{res.stderr}")
    os.replace(tmp, out)
    return out


def _lib(suite):
    if suite not in _LIBS:
        lib = ctypes.CDLL(build(suite))
        lib.cec_eval.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.cec_eval.restype = None
        _LIBS[suite] = lib
    return _LIBS[suite]


class OfficialFunction:
    """One official CEC function at a fixed dimension."""

    def __init__(self, suite, num, ndim, f_bias, name):
        cfg = _SUITES[suite]
        data = os.path.join(_suite_dir(suite), "input_data")
        needed = [f"M_{num}_D{ndim}.txt", f"shift_data_{num}.txt"]
        if num in cfg["shuffled"]:
            needed.append(f"shuffle_data_{num}_D{ndim}.txt")
        missing = [n for n in needed if not os.path.exists(os.path.join(data, n))]
        if ndim not in cfg["dims"] or missing:
            raise ValueError(f"{suite} F{num} is not defined for D = {ndim}"
                             + (f" (missing {', '.join(missing)})" if missing else "") + ".")
        self.suite, self.num, self.ndim = suite, int(num), int(ndim)
        self.name, self.f_bias, self.f_global = name, float(f_bias), float(f_bias)
        self.lb, self.ub = np.full(ndim, -100.0), np.full(ndim, 100.0)
        # The optimum is the first shift vector: the first D numbers of shift_data_<num>.txt.
        with open(os.path.join(data, f"shift_data_{num}.txt")) as fh:
            self.x_global = np.array(fh.read().split()[:ndim], dtype=float)
        if suite == "cec2017" and num == 9:
            # cec17 levy_func uses w = 1 + (z - 1)/4 with z = M(x - o): its minimum is at z = 1,
            # i.e. x* = o + M^-1 1 (cec22 levy_func uses z - 0, so there x* = o).
            M = np.loadtxt(os.path.join(data, f"M_{num}_D{ndim}.txt")).reshape(-1)[:ndim * ndim]
            self.x_global = self.x_global + np.linalg.solve(M.reshape(ndim, ndim), np.ones(ndim))
        self._lib = _lib(suite)

    def evaluate_batch(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64).reshape(-1, self.ndim)
        f = np.empty(X.shape[0], dtype=np.float64)
        args = (X.ctypes.data, f.ctypes.data, self.ndim, X.shape[0], self.num)
        key = (self.num, self.ndim)
        if _LOADED.get(self.suite) != key:               # the C code (re)reads input_data/ now
            cwd = os.getcwd()
            os.chdir(_suite_dir(self.suite))
            try:
                self._lib.cec_eval(*args)
            finally:
                os.chdir(cwd)
            _LOADED[self.suite] = key
        else:
            self._lib.cec_eval(*args)
        return f

    def evaluate(self, x):
        return float(self.evaluate_batch(x)[0])


def get_official_function(suite, fid, ndim):
    """Official function ``fid`` ("F1", ...) of ``suite`` ("cec2017" or "cec2022") at D = ``ndim``."""
    if suite not in _SUITES:
        raise ValueError(f"Unknown suite {suite!r}.")
    num = int(fid[1:])
    if suite == "cec2017":
        if fid not in CEC2017_LABELS:
            raise ValueError(f"CEC2017 {fid} is not available (F2 is excluded; F1, F3-F30 exist).")
        return OfficialFunction(suite, num, ndim, 100.0 * num, CEC2017_NAMES[num])
    from cec2022_functions import CEC2022_INFO
    info = {i.fid: i for i in CEC2022_INFO}
    if fid not in info:
        raise ValueError(f"Unknown CEC2022 function id {fid!r}; expected F1-F12.")
    return OfficialFunction(suite, num, ndim, info[fid].f_star, info[fid].name)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for s in _SUITES:
        print(f"{s}: {build(s, force='--build' in sys.argv)}  source sha256 {source_sha256(s)[:16]}")
    if "--build" in sys.argv:
        sys.exit(0)
    ok = True
    for s, dims, fids in (("cec2022", (10, 20), [f"F{k}" for k in range(1, 13)]),
                          ("cec2017", (10, 30, 50, 100), CEC2017_LABELS)):
        for d in dims:
            funcs = [get_official_function(s, fid, d) for fid in fids]
            worst = max(abs(f.evaluate(f.x_global) - f.f_bias) for f in funcs)
            passed = worst < 1e-6
            ok &= passed
            print(f"{s} D={d:<3} {len(fids)} functions  max |f(x*) - F*| = {worst:.1e}  {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if ok else 1)
