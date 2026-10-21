#!/usr/bin/env python3
"""Verification of the MATLAB/Octave port matlab/rti_de.m against python/RTIDE/optimizer.py.

Two steps, with the Octave script in between (see matlab/VERIFICATION.md):

    python verify_rtide.py prepare      # Python side: inputs, references, recorded random numbers
    octave-cli --eval "run_octave_checks"
    python verify_rtide.py compare      # compares, writes output/summary.json and prints a report

prepare writes into output/:
  rules_in / rules_py      fixed inputs of every update rule and the Python results (the
                           expressions below are copied from optimizer.py, lines cited)
  draws_<case>             the random numbers drawn by an UNMODIFIED Python run (recorded by a
                           test-only proxy of numpy.random.Generator; optimizer.py is not changed)
  py_<case>                the Python trajectory of that run (one row per generation) and result
  counts_py                evaluation counts for the count cases
  stat_py                  final errors of 30 independent Python runs per function

The proxy draws exactly the numbers NumPy would draw (uniform = lb + (ub - lb) * random(),
normal = loc + scale * standard_normal(), integers and random unchanged), and every recorded run is
checked to be bit-identical to the same run without the proxy.
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")
sys.path.insert(0, os.path.join(REPO, "python"))

import RTIDE  # noqa: E402
from RTIDE import optimizer  # noqa: E402
from RTIDE import rti_de  # noqa: E402

CASES = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8"))


# --------------------------------------------------------------------------- array files
def write_arrays(prefix, arrays):
    with open(prefix + ".idx", "w") as fi, open(prefix + ".bin", "wb") as fb:
        for name, a in arrays.items():
            a = np.asarray(a, dtype="<f8")
            a = a.reshape(1, 1) if a.ndim == 0 else (a.reshape(-1, 1) if a.ndim == 1 else a)
            fi.write(f"{name} {a.shape[0]} {a.shape[1]}\n")
            fb.write(np.ascontiguousarray(a).tobytes())


def read_arrays(prefix):
    out, data = {}, open(prefix + ".bin", "rb").read()
    o = 0
    for line in open(prefix + ".idx"):
        name, r, c = line.split()
        r, c = int(r), int(c)
        out[name] = np.frombuffer(data, "<f8", r * c, o).reshape(r, c)
        o += 8 * r * c
    return out


# --------------------------------------------------------------------------- test functions
# The MATLAB versions (run_octave_checks.m) sum along rows with rtide_rules().npsum_rows, the order
# of np.sum, so that the objective values agree to the last bit except for the library cos().
def sphere(X):
    return np.sum(X ** 2, axis=1)


def shifted_sphere(X):
    return 500.0 + np.sum((X - 1.5) ** 2, axis=1)


def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def rastrigin_nan(X):
    v = rastrigin(X)
    v[X[:, 0] > 4.0] = np.nan
    return v


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


FUNCS = dict(sphere=sphere, shifted_sphere=shifted_sphere, rastrigin=rastrigin,
             rastrigin_nan=rastrigin_nan, rosenbrock=rosenbrock)
F_STAR = dict(sphere=0.0, shifted_sphere=500.0, rastrigin=0.0, rosenbrock=0.0)


def kwargs_of(case):
    return {k: v for k, v in case["options"].items()}


def bounds(case):
    return np.full(case["dim"], float(case["lb"])), np.full(case["dim"], float(case["ub"]))


# --------------------------------------------------------------------------- recording proxy
class RecordingGenerator:
    """Wraps numpy.random.Generator; draws the same numbers and records the primitive draws."""
    UNIFORM, RANDINT, RANDN, RAND = 1, 2, 3, 4

    def __init__(self, gen):
        self.g, self.log = gen, []

    def uniform(self, low, high, size):
        u = self.g.random(size)
        self.log.append((self.UNIFORM, 0, u.copy()))
        low, high = np.asarray(low, float), np.asarray(high, float)
        return low + (high - low) * u

    def integers(self, low, high=None, size=None):
        assert high is None, "only integers(high, size) is used by optimizer.py"
        v = self.g.integers(low, size=size)
        self.log.append((self.RANDINT, int(low), v.copy()))   # copy: optimizer.py changes r1, r2 in place
        return v

    def normal(self, loc, scale, size):
        z = self.g.standard_normal(size)
        self.log.append((self.RANDN, 0, z.copy()))
        return loc + scale * z

    def random(self, size):
        v = self.g.random(size)
        self.log.append((self.RAND, 0, v.copy()))
        return v


def run_recorded(func, lb, ub, max_fes, seed, **kw):
    real = np.random.default_rng
    holder = {}

    def fake(s=None):
        holder["rec"] = RecordingGenerator(real(s))
        return holder["rec"]

    np.random.default_rng = fake                  # test-only: optimizer.py calls np.random.default_rng
    try:
        trace = []
        res = rti_de(func, lb, ub, max_fes, seed=seed, trace=trace, **kw)
    finally:
        np.random.default_rng = real
    return res, trace, holder["rec"].log


def write_draws(prefix, log):
    with open(prefix + ".idx", "w") as fi, open(prefix + ".bin", "wb") as fb:
        for kind, high, v in log:
            a = np.asarray(v, dtype="<f8")
            a = a.reshape(-1, 1) if a.ndim == 1 else a
            fi.write(f"{kind} {high} {a.shape[0]} {a.shape[1]}\n")
            fb.write(np.ascontiguousarray(a).tobytes())


def trace_arrays(res, trace):
    x, f, fes = res
    th = [g["theta"] for g in trace]
    return dict(x=x[None, :], f=f, fes=fes,
                gen_fes=[g["fes"] for g in trace], gen_n=[g["n"] for g in trace],
                gen_mu_F=[g["mu_F"] for g in trace], gen_f_best=[g["f_best"] for g in trace],
                gen_evaluated=np.array([g["evaluated"] for g in trace]).reshape(-1, 3),
                gen_accepted=np.array([g["accepted"] for g in trace]).reshape(-1, 3),
                gen_improved=np.array([g["improved"] for g in trace]).reshape(-1, 3),
                gen_skipped=[g["skipped"] for g in trace],
                theta=np.concatenate(th) if th else np.zeros(0),
                theta_accepted=np.concatenate([g["theta_accepted"] for g in trace]) if th else np.zeros(0))


# --------------------------------------------------------------------------- rule references
# Each function repeats the lines of python/RTIDE/optimizer.py named in its comment, verbatim
# except for the variable names of the inputs.
def ref_parabola(fit, f_one, f_half):
    # optimizer.py lines 97-104
    d_one, d_half = f_one - fit, f_half - fit
    with np.errstate(all="ignore"):
        beta = (d_half - 0.5 * d_one) / -0.25
        t = -(d_one - beta) / (2.0 * beta)
    t = np.where(np.isfinite(t) & (beta > 0), t, -0.5)
    ok = np.isfinite(d_one) & np.isfinite(d_half)
    t = np.clip(np.where(ok, t, 0.0), -1.0, 1.0)
    ok &= (np.abs(t) >= 0.02) & (np.abs(t - 0.5) >= 0.02) & (np.abs(t - 1.0) >= 0.02)
    return t, ok


def ref_repair(V, X, lb, ub):
    # optimizer.py lines 128-129
    V = np.where(V < lb, (lb + X) / 2.0, V)
    V = np.where(V > ub, (ub + X) / 2.0, V)
    return V


def ref_update_muF(mu_F, fit, f_new, F, T):
    # optimizer.py lines 149-155 (inputs already restricted to the counted agents)
    gain = fit - f_new
    gain = np.where(np.isfinite(gain), gain, 0.0)
    w = gain / gain.sum() if gain.sum() > 0 else np.full(gain.size, 1.0 / gain.size)
    s_f = F * np.abs(T)
    den = np.sum(w * s_f)
    lehmer = np.sum(w * s_f ** 2) / den if den > 0 else 0.0
    return float(np.clip(0.9 * mu_F + 0.1 * lehmer, 0.05, 1.0))


def ref_lpsr(n_min, n_init, max_fes, fes):
    # optimizer.py line 181
    return max(n_min, int(round((n_min - n_init) / max_fes * fes + n_init)))


def ref_elite(p, n):
    # optimizer.py line 117
    return max(2, int(round(p * n)))


def make_rule_inputs(rng):
    # parabola: hand-made cases (convex contraction, convex extrapolation clipped to -1, not convex,
    # a = 0, exact minimiser at the agent, within delta of 0, infinite values) + random cases
    hand = [  # (fit, f_one, f_half)
        (0.0, 10.0, 1.0),        # D1 > 4 Dh: contraction, t* in (0, 1/4)
        (0.0, 3.0, 1.0),         # 2 Dh < D1 < 4 Dh: extrapolation behind the agent
        (0.0, 2.1, 1.0),         # extrapolation beyond -1: clipped to -1
        (0.0, 1.0, 1.0),         # not convex: -1/2
        (0.0, 2.0, 1.0),         # a = 0: -1/2
        (0.0, 4.0, 1.0),         # D1 = 4 Dh: t* = 0, skip
        (0.0, 4.1, 1.0),         # t* = 0.0061, within 0.02: skip
        (0.0, 3.85, 1.0),        # t* = -0.0102, skip
        (0.0, 5.0, 1.0),         # t* = 1/12, kept
        (1.0, np.inf, 2.0),      # infinite f(x + d): skip
        (1.0, 3.0, np.inf),      # infinite half step: skip
        (np.inf, np.inf, np.inf),  # agent infinite: skip
        (5.0, 5.0 + 1e-300, 5.0 + 1e-300),
    ]
    fit = np.array([h[0] for h in hand] + list(rng.normal(0, 10, 2000)))
    f_one = np.array([h[1] for h in hand] + list(fit[len(hand):] + rng.exponential(5.0, 2000)))
    f_half = np.array([h[2] for h in hand] + list(fit[len(hand):] + rng.exponential(2.0, 2000)))
    # repair
    X = rng.uniform(-1.0, 1.0, (500, 7))
    lb, ub = np.full(7, -1.0), np.full(7, 1.0)
    V = X + rng.normal(0, 1.0, (500, 7))
    # mu_F update: groups of different sizes (pairwise summation above 8 and above 128 values),
    # with infinite and zero improvements
    groups = []
    for size in (1, 2, 7, 8, 9, 16, 50, 128, 129, 200, 300):
        fitg = rng.normal(100.0, 30.0, size)
        f_new = fitg - rng.exponential(5.0, size)
        groups.append((rng.uniform(0.05, 1.0), fitg, f_new, rng.uniform(0.05, 1.0, size), np.ones(size)))
    fitg = np.array([np.inf, 3.0, 4.0]); groups.append((0.5, fitg, np.array([7.0, 2.0, 1.0]), np.array([0.3, 0.6, 0.9]), np.ones(3)))
    fitg = np.array([np.inf, np.inf]); groups.append((0.7, fitg, np.array([1.0, 2.0]), np.array([0.2, 0.8]), np.ones(2)))
    groups.append((0.04, np.array([2.0]), np.array([1.0]), np.array([0.05]), np.ones(1)))   # mu_F below 0.05: clipped
    # LPSR and m: grids that contain exact ties (x.5)
    lpsr = [(4, 180, 352, fes) for fes in range(0, 353)] + [(4, 180, 100000, fes) for fes in range(180, 100001, 997)] \
        + [(6, 40, 6000, fes) for fes in range(40, 6001, 37)]
    elite = [(p, n) for p in (0.11, 0.05, 0.2, 0.125) for n in range(4, 401)]
    return dict(fit=fit, f_one=f_one, f_half=f_half, X=X, V=V, lb=lb, ub=ub, groups=groups, lpsr=lpsr,
                elite=elite)


def prepare():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(12345)
    I = make_rule_inputs(rng)
    rin = dict(par_fit=I["fit"], par_f_one=I["f_one"], par_f_half=I["f_half"],
               rep_X=I["X"], rep_V=I["V"], rep_lb=I["lb"][None], rep_ub=I["ub"][None],
               lpsr=np.array(I["lpsr"], float), elite=np.array(I["elite"], float),
               ng=len(I["groups"]))
    t, ok = ref_parabola(I["fit"], I["f_one"], I["f_half"])
    rpy = dict(par_t=t, par_ok=ok, rep_V=ref_repair(I["V"], I["X"], I["lb"], I["ub"]),
               lpsr=[ref_lpsr(*a) for a in I["lpsr"]], elite=[ref_elite(*a) for a in I["elite"]])
    mus = []
    for g, (mu, fitg, f_new, F, T) in enumerate(I["groups"]):
        rin[f"g{g}_mu"], rin[f"g{g}_fit"], rin[f"g{g}_fnew"], rin[f"g{g}_F"], rin[f"g{g}_T"] = mu, fitg, f_new, F, T
        mus.append(ref_update_muF(mu, fitg, f_new, F, T))
    rpy["muF"] = mus
    # np.sum of vectors of every length 1..400 (the order that npsum must reproduce)
    sums_in = rng.standard_normal((400, 400)) * 10.0 ** rng.integers(-6, 6, (400, 400))
    rin["sum_in"] = sums_in
    rpy["sum_out"] = [np.sum(sums_in[i, :i + 1]) for i in range(400)]
    write_arrays(os.path.join(OUT, "rules_in"), rin)
    write_arrays(os.path.join(OUT, "rules_py"), rpy)

    # trace-level cases: recorded run, checked against the same run without the proxy
    for case in CASES["trace_cases"]:
        lb, ub = bounds(case)
        func = FUNCS[case["func"]]
        plain_trace = []
        plain = rti_de(func, lb, ub, case["max_fes"], seed=case["seed"], trace=plain_trace, **kwargs_of(case))
        res, trace, log = run_recorded(func, lb, ub, case["max_fes"], case["seed"], **kwargs_of(case))
        same = (np.array_equal(plain[0], res[0]) and plain[1] == res[1] and plain[2] == res[2]
                and [g["f_best"] for g in plain_trace] == [g["f_best"] for g in trace]
                and [g["mu_F"] for g in plain_trace] == [g["mu_F"] for g in trace])
        if not same:
            sys.exit(f"recording changed the run of {case['name']}")
        write_draws(os.path.join(OUT, "draws_" + case["name"]), log)
        write_arrays(os.path.join(OUT, "py_" + case["name"]), trace_arrays(res, trace))
        print(f"  {case['name']}: {len(trace)} generations, {len(log)} draws, recorded run = plain run")

    # evaluation counts
    rows = []
    for case in CASES["count_cases"]:
        lb, ub = bounds(case)
        calls = []
        f = FUNCS[case["func"]]
        x, fx, fes = rti_de(lambda X: (calls.append(X.shape[0]), f(X))[1], lb, ub, case["max_fes"], seed=0)
        rows.append([case["max_fes"], fes, sum(calls), len(calls)])
    write_arrays(os.path.join(OUT, "counts_py"), dict(rows=np.array(rows, float)))

    # statistical comparison: independent runs
    st = CASES["stat"]
    lb, ub = np.full(st["dim"], st["lb"]), np.full(st["dim"], st["ub"])
    t0 = time.time()
    err = np.zeros((st["runs"], len(st["funcs"])))
    for j, name in enumerate(st["funcs"]):
        for r in range(st["runs"]):
            err[r, j] = rti_de(FUNCS[name], lb, ub, st["max_fes"], seed=st["seed_base"] + r)[1] - F_STAR[name]
    write_arrays(os.path.join(OUT, "stat_py"), dict(err=err))
    env = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
               processor=platform.processor(),
               optimizer_sha256=hashlib.sha256(open(optimizer.__file__, "rb").read()).hexdigest(),
               stat_runtime_s=round(time.time() - t0, 1))
    json.dump(env, open(os.path.join(OUT, "env_py.json"), "w"), indent=2)
    print("prepare done:", env)


# --------------------------------------------------------------------------- compare
def diff(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape:
        return dict(shape_equal=False, n_a=a.size, n_b=b.size)
    fin = np.isfinite(a) & np.isfinite(b)
    same_nonfinite = bool(np.array_equal(a[~fin], b[~fin]) and np.array_equal(np.isfinite(a), np.isfinite(b)))
    ad = np.abs(a[fin] - b[fin])
    rd = ad / np.maximum(np.abs(a[fin]), 1e-300)
    return dict(shape_equal=True, n=int(a.size), bit_identical=bool(np.array_equal(a, b)),
                n_unequal=int((a != b).sum()), nonfinite_equal=same_nonfinite,
                max_abs=float(ad.max()) if ad.size else 0.0, max_rel=float(rd.max()) if rd.size else 0.0)


def compare():
    from scipy import stats
    S = {}
    rin, rpy, roc = read_arrays(os.path.join(OUT, "rules_in")), read_arrays(os.path.join(OUT, "rules_py")), \
        read_arrays(os.path.join(OUT, "rules_oct"))
    S["rules"] = {k: diff(rpy[k], roc[k]) for k in ("par_t", "par_ok", "rep_V", "muF", "lpsr", "elite", "sum_out")}
    S["octave_known_answer_checks"] = dict(passed=int(roc["kat_passed"][0, 0]), total=int(roc["kat_total"][0, 0]))
    S["trace"] = {}
    for case in CASES["trace_cases"]:
        p, o = read_arrays(os.path.join(OUT, "py_" + case["name"])), read_arrays(os.path.join(OUT, "oct_" + case["name"]))
        d = {k: diff(p[k], o[k]) for k in p}
        d["generations_py"], d["generations_oct"] = int(p["gen_fes"].size), int(o["gen_fes"].size)
        d["draws_left_unused"] = int(o["draws_left"][0, 0])
        S["trace"][case["name"]] = d
    cp, co = read_arrays(os.path.join(OUT, "counts_py"))["rows"], read_arrays(os.path.join(OUT, "counts_oct"))["rows"]
    S["counts"] = dict(columns=["max_fes", "fes returned", "evaluations counted", "calls"],
                       python=cp.tolist(), octave=co.tolist())
    st = CASES["stat"]
    ep, eo = read_arrays(os.path.join(OUT, "stat_py"))["err"], read_arrays(os.path.join(OUT, "stat_oct"))["err"]
    S["stat"] = {}
    for j, name in enumerate(st["funcs"]):
        a, b = ep[:, j], eo[:, j]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        S["stat"][name] = dict(n_py=int(a.size), n_oct=int(b.size),
                               median_py=float(np.median(a)), median_oct=float(np.median(b)),
                               q25_py=float(np.percentile(a, 25)), q75_py=float(np.percentile(a, 75)),
                               q25_oct=float(np.percentile(b, 25)), q75_oct=float(np.percentile(b, 75)),
                               mean_py=float(a.mean()), mean_oct=float(b.mean()),
                               mannwhitney_U=float(u.statistic), p_value=float(u.pvalue))
    S["env_py"] = json.load(open(os.path.join(OUT, "env_py.json")))
    S["env_oct"] = json.load(open(os.path.join(OUT, "env_oct.json")))
    S["scipy"] = __import__("scipy").__version__
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(S, indent=2))
    write_report(S)



def write_report(S):
    """Writes matlab/VERIFICATION.md; every number in it is read from output/summary.json."""
    ep, eo, st = S["env_py"], S["env_oct"], CASES["stat"]

    def yes(b):
        return "yes" if b else "**NO**"

    def same(d):
        return "bit-identical" if d.get("bit_identical") else "max abs. diff. %.3g" % d.get("max_abs", float("nan"))

    L = [
        "# Verification of the MATLAB/Octave port `rti_de.m`",
        "",
        "Generated by `matlab/verification/verify_rtide.py compare` from `matlab/verification/output/summary.json`;",
        "do not edit by hand. Reference implementation: `python/RTIDE/optimizer.py`, SHA-256",
        f"`{ep['optimizer_sha256']}`.",
        "",
        "## Environment",
        "",
        f"- Python {ep['python']}, NumPy {ep['numpy']}, SciPy {S['scipy']} ({ep['platform']}; {ep['processor']})",
        f"- GNU Octave {eo['octave']} ({eo['computer']})",
        f"- Runtime of the independent runs (c): Python {ep['stat_runtime_s']} s, Octave {eo['stat_runtime_s']} s",
        "- The port uses only functions available in both MATLAB (R2016b or later, for implicit",
        "  expansion) and GNU Octave; the checks below were run with GNU Octave.",
        "",
        "## Commands (from `matlab/verification/`)",
        "",
        "```",
        "python verify_rtide.py prepare",
        "octave-cli --no-gui --quiet --eval \"run_octave_checks\"",
        "python verify_rtide.py compare",
        "```",
        "",
        "Settings of all cases: `cases.json`. Seeds: trace cases as listed there; rule inputs from",
        "`numpy.random.default_rng(12345)`; independent runs with the seeds "
        f"{st['seed_base']}..{st['seed_base'] + st['runs'] - 1} in each language.",
        "",
        "## (a) Update rules with fixed inputs",
        "",
        "Python reference: the lines of `optimizer.py` cited in `verify_rtide.py` (ref_* functions).",
        "Octave: `rtide_rules.m`, the functions called by `rti_de.m`.",
        "",
        "| Rule | values compared | bit-identical | max. abs. difference |",
        "|---|---|---|---|",
    ]
    names = dict(par_t="parabola factor t* (convex, clipped to -1, non-convex, a = 0, infinite values, random)",
                 par_ok="skip rule (delta = 0.02, non-finite increases)", rep_V="midpoint bound repair",
                 muF="mu_F update (weighted Lehmer mean; 1-300 values, infinite gains, clipping)",
                 lpsr="population size of the LPSR, incl. ties x.5", elite="m = max(2, round(p N)), incl. ties",
                 sum_out="summation order (np.sum vs npsum, lengths 1-400)")
    for k, d in S["rules"].items():
        L.append(f"| {names[k]} | {d['n']} | {yes(d['bit_identical'])} | {d['max_abs']:.3g} |")
    k = S["octave_known_answer_checks"]
    L += ["",
          f"Known-answer and run-level checks in Octave (`run_octave_checks.m` prints the list): {k['passed']} of "
          f"{k['total']} passed. They cover hand-computed parabola factors, repair and Lehmer mean, ties of the",
          "rounding, the exact budget, points inside the box, returned value = best evaluated value, a constant",
          "population without LPSR, no chain before 0.7 of the budget, t* in [-1, 1/4), the LPSR schedule of",
          "the trace, same seed = same run, and convergence on a shifted sphere.",
          "",
          "Evaluation counts (objective wrapped by a counter; Rastrigin, D = 8, so N_init = 144):",
          "",
          "| max_fes | Python: fes / evaluations counted / calls | Octave: fes / evaluations counted / calls |",
          "|---|---|---|"]
    for a, b in zip(S["counts"]["python"], S["counts"]["octave"]):
        L.append(f"| {int(a[0])} | {int(a[1])} / {int(a[2])} / {int(a[3])} | {int(b[1])} / {int(b[2])} / {int(b[3])} |")
    L += ["",
          "With max_fes below N_init both implementations evaluate the whole initial population and stop",
          "(fes = N_init > max_fes), which is the behaviour of `optimizer.py`.",
          "",
          "## (b) Trace-level equivalence with injected random numbers",
          "",
          "The random numbers drawn by unmodified Python runs are recorded by a test-only proxy of",
          "`numpy.random.Generator` (installed by replacing `numpy.random.default_rng` during the call;",
          "`optimizer.py` is not changed). Each recorded run is checked to be bit-identical to the same",
          "run without the proxy. Octave replays the draws through the `rng_source` option of `rti_de.m`;",
          "`replay_source.m` stops with an error if the kind, bound or shape of a requested draw differs",
          "from the recorded one. The Octave test functions sum in the order of `np.sum`.",
          "",
          "| case | generations (Py / Oct) | draws left over | final x | final f | fes | per generation: f_best, mu_F, N, FE | per generation: counts by stage, skipped, all t* |",
          "|---|---|---|---|---|---|---|---|"]
    for c, d in S["trace"].items():
        per_gen = all(d[x].get("bit_identical") for x in ("gen_f_best", "gen_mu_F", "gen_n", "gen_fes"))
        stage = all(d[x].get("bit_identical") for x in ("gen_evaluated", "gen_accepted", "gen_improved",
                                                         "gen_skipped", "theta", "theta_accepted"))
        L.append(f"| {c} | {d['generations_py']} / {d['generations_oct']} | {d['draws_left_unused']} | "
                 f"{same(d['x'])} | {same(d['f'])} | {'equal' if d['fes'].get('bit_identical') else 'DIFFERENT'} | "
                 f"{'bit-identical' if per_gen else 'DIFFERENT'} | {'identical' if stage else 'DIFFERENT'} |")
    L += ["",
          "Options of the cases (`cases.json`): defaults; CR = None with chain = 1 and chain_from = 0.3;",
          "lpsr = False, adapt = False, n_init = 40, n_min = 6, p = 0.2, CR = 0.7; an objective that is NaN",
          "for x_1 > 4 with chain_from = 0; D = 2 with n_init = 10. The budgets 9999 and 500 are not",
          "multiples of the population size and test the truncated last generation.",
          "",
          "## (c) Independent runs (native random numbers of each language)",
          "",
          f"{st['runs']} runs per function and language, D = {st['dim']}, box [{st['lb']}, {st['ub']}]^D, "
          f"max_fes = {st['max_fes']}; error = f_best - f*. Python: NumPy PCG64 seeded with the run number;",
          "Octave: `rng(seed, 'twister')`. Two-sided Mann-Whitney U test (SciPy).",
          "",
          "| function | median error Py [Q1, Q3] | median error Oct [Q1, Q3] | mean Py | mean Oct | U | p |",
          "|---|---|---|---|---|---|---|"]
    for f, d in S["stat"].items():
        L.append(f"| {f} | {d['median_py']:.4g} [{d['q25_py']:.4g}, {d['q75_py']:.4g}] | "
                 f"{d['median_oct']:.4g} [{d['q25_oct']:.4g}, {d['q75_oct']:.4g}] | {d['mean_py']:.4g} | "
                 f"{d['mean_oct']:.4g} | {d['mannwhitney_U']:.1f} | {d['p_value']:.3g} |")
    L += ["",
          "This comparison uses the native random number generators of the two languages; the exact",
          "equivalence of the two implementations is established by (b).",
          ""]
    path = os.path.join(os.path.dirname(HERE), "VERIFICATION.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print("wrote", path)

if __name__ == "__main__":
    {"prepare": prepare, "compare": compare}[sys.argv[1]]()
