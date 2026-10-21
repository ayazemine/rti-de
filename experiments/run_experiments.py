#!/usr/bin/env python3
"""
Run every experiment of the RTI-DE paper in parallel, with resume.

Algorithms
----------
RTI-DE   RTIDE/optimizer.py (the proposed algorithm)
DE       benchmarks/de_reference.py (DE/rand/1/bin, F = 0.5, CR = 0.9, N = --agents)
RIME, PSO, GWO, WOA   mealpy 3.0.3 Original* classes with their default parameters (N = --agents)
LSHADE   benchmarks/lshade_reference.py (N from 18 D to 4)
Variants of RTI-DE, each run into its own directory with the settings of the main runs:
  ABLATION     one component removed or replaced   --algorithms <names> --out results_cache/paper_ablation
  SENSITIVITY  one parameter changed               --algorithms <names> --out results_cache/paper_sensitivity
  (the names and the complete commands are printed by scripts/make_tables.py --only F G)

Suites
------
cec2022               CEC2022 F1-F12 (the organisers' C code), D = 20
cec2017               CEC2017 F1, F3-F30 (the organisers' C code), D = 100
composite             Laminate stacking sequence, max buckling load (Le Riche & Haftka 1993)
composite_contiguity  Same, with the <=4-contiguous-plies constraint (PENALTY in laminate_buckling.py)
uav                   3-D UAV path planning over terrain with threats (Phung & Ha 2021)

Unit of work = one (suite, function, algorithm, run). Each finished unit is written
to its own file, so the job can be stopped at any time (Ctrl+C) and continued by
running the same command again: finished runs are skipped.

    <out>/<suite>/config.json                    settings + one entry per session (git, code hash, versions)
    <out>/<suite>/runs/<F>/<ALGO>/run_XX.pkl     one finished run
    <out>/<suite>/runs/<F>/<ALGO>/run_XX.error.json   a failed run (retried on the next start)
    <out>/<suite>/results.pkl                    all runs, collected by the summary step
    <out>/<suite>/*.csv                          summary / comparison / statistics tables
    <out>/run_log.txt                            one line per finished run

Every run has a budget of function evaluations (FEs), counted by an objective wrapper:
--budget fe    the suite's --max-fes (one number for all suites, or per suite:
               cec2022=1000000 uav=150000);
--budget iter  N (T + 1) with N = --agents and T = --iterations, the evaluations of a
               population of N over T iterations.
The reported value of a run is the best value among its first budget evaluations (mealpy
finishes its epoch, so its algorithms can evaluate a few points more; they are not counted),
and the stored solution x is the point at which it was reached. The mealpy algorithms get the
number of epochs that fills the budget, (budget - N) // N, so that their schedules run to the end
(each of them evaluates one point per agent and epoch).

On the CEC suites a run ends as soon as its error f - F* is below 1e-8 (termination rule of the
CEC protocols; such a run is reported as error 0 in any case). --no-target-stop switches this
off. The FE-based trace of best-so-far improvements is stored, so that convergence can be
plotted over FEs for all algorithms alike. Run r of every algorithm uses seed = seed + r
(the same seeds for every algorithm; the runs of two algorithms are compared as independent
samples with the Mann-Whitney U test, because their random numbers are not shared).

CEC functions
-------------
Both CEC suites are evaluated by the official C code (benchmarks/cec_official_functions.py,
compiled once with g++; sources and data in benchmarks/official_cec/). opfunu is not used:
its cec2017/cec2022 modules do not reproduce the official values. The source of the CEC
functions is recorded in config.json ("cec_source"). CEC2017 uses the numbering of the C code:
F1, F3..F30 (F2 is the deleted function), error = f - F* with F* = 100 * function number.

Examples (from experiments/)
--------
  # CEC suites of the paper: 1,000,000 evaluations, 30 runs
  python -W ignore run_experiments.py --suites cec2022 cec2017 --budget fe --max-fes 1000000

  # engineering problems: N = 50, T = 1000, i.e. 50,050 evaluations
  python -W ignore run_experiments.py --suites composite composite_contiguity uav

  # only CEC2022, 10 worker processes
  python -W ignore run_experiments.py --suites cec2022 --workers 10

  # what would run, with a rough duration estimate, without running anything
  python -W ignore run_experiments.py --dry-run

  # smoke test into a throw-away directory
  python -W ignore run_experiments.py --runs 1 --iterations 5 --out results_cache/smoke

  # tables only, from whatever has finished so far
  python -W ignore run_experiments.py --summarize-only
"""

import os

# One BLAS/OpenMP thread per worker process; parallelism comes from the processes.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import pickle
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))           # experiments/ of the repository
REPO_ROOT = os.path.dirname(ROOT)                           # repository root
PACKAGE_DIR = os.path.join(REPO_ROOT, "python")             # holds the RTIDE package
OPTIMIZER_PATH = os.path.join(PACKAGE_DIR, "RTIDE", "optimizer.py")
for _p in (PACKAGE_DIR, ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "applications"),
           os.path.join(ROOT, "applications", "composite"), os.path.join(ROOT, "applications", "uav")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALL_ALGORITHMS = ["RTI-DE", "DE", "LSHADE", "RIME", "PSO", "GWO", "WOA"]
PROPOSED = ALL_ALGORITHMS[0]
# Variants of the proposed algorithm: name -> (options of rti_de, description).
# "n_init_per_dim" is turned into n_init = value * D.
ABLATION = {
    "RTI-DE-noChain": (dict(chain=0), "rejected-trial chain removed"),
    "RTI-DE-halfStep": (dict(chain=1), "parabola point of the chain removed"),
    "RTI-DE-chainAlways": (dict(chain_from=0.0), "chain used from the first evaluation"),
    "RTI-DE-noLPSR": (dict(lpsr=False), "population size kept constant"),
    "RTI-DE-fixedF": (dict(adapt=False), "adaptation of the step factor removed"),
    "RTI-DE-noCR": (dict(CR=None), "binomial crossover removed"),
}
SENSITIVITY = {
    "RTI-DE-cf50": (dict(chain_from=0.5), "chain\\_from = 0.5"),
    "RTI-DE-cf90": (dict(chain_from=0.9), "chain\\_from = 0.9"),
    "RTI-DE-p05": (dict(p=0.05), "p = 0.05"),
    "RTI-DE-p20": (dict(p=0.2), "p = 0.2"),
    "RTI-DE-n10D": (dict(n_init_per_dim=10), "n\\_init = 10 D"),
    "RTI-DE-n25D": (dict(n_init_per_dim=25), "n\\_init = 25 D"),
    "RTI-DE-cr70": (dict(CR=0.7), "CR = 0.7"),
    "RTI-DE-cr95": (dict(CR=0.95), "CR = 0.95"),
}
VARIANTS = {**ABLATION, **SENSITIVITY}
# Development only: further variants as JSON {name: options} in the environment variable
# RTIDE_DEV_VARIANTS (inherited by the worker processes). Not used for the results of the paper.
VARIANTS.update({k: (v, "development variant") for k, v in
                 json.loads(os.environ.get("RTIDE_DEV_VARIANTS", "{}")).items()})
MEALPY = {"RIME": ("mealpy.physics_based.RIME", "OriginalRIME"),      # name -> (module, class)
          "PSO": ("mealpy.swarm_based.PSO", "OriginalPSO"),
          "GWO": ("mealpy.swarm_based.GWO", "OriginalGWO"),
          "WOA": ("mealpy.swarm_based.WOA", "OriginalWOA")}
# Worst-case function evaluations per epoch, in units of N, of mealpy algorithms that evaluate
# more than once per agent (none in the present list; mealpy 3.0.3 HHO would need 3).
FES_PER_EPOCH = {}
CODE_FILES = {"optimizer_sha256": (os.pardir, "python", "RTIDE", "optimizer.py"),
              "de_reference_sha256": ("benchmarks", "de_reference.py"),
              "lshade_reference_sha256": ("benchmarks", "lshade_reference.py")}
ERROR_FLOOR = 1e-8
CEC_SUITES = ("cec2022", "cec2017")
ENG_SUITES = ("composite", "composite_contiguity", "uav")
ALL_SUITES = CEC_SUITES + ENG_SUITES
CEC2017_LABELS = ["F1"] + [f"F{k}" for k in range(3, 31)]

# Settings that must be identical for all runs stored in one suite directory.
LOCKED_SETTINGS = ("suite", "dim", "agents", "iterations", "budget", "max_fes", "seed_base",
                   "target_stop", "mealpy", "cec_source") + tuple(CODE_FILES)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

def cec_source(suite):
    """Identifies the CEC implementation in config.json (None for the engineering problems)."""
    if suite not in CEC_SUITES:
        return None
    from cec_official_functions import source_sha256
    return f"official C code, sha256 {source_sha256(suite)}"


def function_ids(suite):
    if suite == "cec2022":
        from cec2022_functions import CEC2022_INFO
        return [info.fid for info in CEC2022_INFO]
    if suite == "cec2017":
        return list(CEC2017_LABELS)
    return [suite]


_OBJ_CACHE = {}


def load_objective(suite, fid, dim):
    """dict(vec, scalar, lb, ub, dim, f_star, maximize, source) for one function."""
    key = (suite, fid, dim)
    if key in _OBJ_CACHE:
        return _OBJ_CACHE[key]
    if suite in CEC_SUITES:
        if suite == "cec2022":
            from cec2022_functions import get_cec2022_function
            f = get_cec2022_function(fid, ndim=dim)
        else:
            from cec_official_functions import get_official_function
            f = get_official_function("cec2017", fid, dim)
        source = f"official {suite} func_num {f.num}"
        obj = dict(vec=f.evaluate_batch, scalar=f.evaluate,
                   lb=np.asarray(f.lb, float), ub=np.asarray(f.ub, float), dim=dim,
                   f_star=float(f.f_bias), x_star=np.asarray(f.x_global, float),
                   maximize=False, source=source)
    else:
        from run_engineering import load_problem
        p = load_problem(suite)
        obj = dict(vec=p["objective"], scalar=p["scalar"], lb=np.asarray(p["lb"], float),
                   ub=np.asarray(p["ub"], float), dim=p["dim"], f_star=None, x_star=None,
                   maximize=p["maximize"], reference=p["reference"], source=suite)
    _OBJ_CACHE[key] = obj
    return obj


class Counter:
    """Objective wrapper: counts FEs, treats non-finite candidates as infeasible
    (f = +inf, counted in ``invalid``; the same rule for every algorithm) and keeps
    an FE-based trace of best-so-far improvements [(fe, best), ...]."""

    def __init__(self, vec, scalar):
        self.vec, self.scalar = vec, scalar
        self.n, self.invalid, self.best, self.trace = 0, 0, np.inf, []

    def _record(self, f):
        start = self.n - len(f)
        run = np.minimum.accumulate(np.r_[self.best, f])
        for j in np.nonzero(run[1:] < run[:-1])[0]:
            self.trace.append((start + int(j) + 1, float(run[j + 1])))
        self.best = float(run[-1])

    def vectorized(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.n += X.shape[0]
        ok = np.all(np.isfinite(X), axis=1)
        f = np.full(X.shape[0], np.inf)
        if ok.any():
            f[ok] = self.vec(X[ok])
        self.invalid += int((~ok).sum())
        self._record(f)
        return f

    def single(self, x):
        x = np.asarray(x, dtype=float)
        self.n += 1
        if not np.all(np.isfinite(x)):
            self.invalid += 1
            f = np.inf
        else:
            f = float(self.scalar(x))
        self._record(np.array([f]))
        return f


class TargetReached(Exception):
    """Raised by BudgetCounter to end a run whose error is already below ERROR_FLOOR."""


class BudgetCounter(Counter):
    """Counter that keeps the best point among the first ``max_fes`` evaluations (``best_f``,
    ``best_x``) and, when ``f_target`` is given, ends the run as soon as best_f <= f_target."""

    def __init__(self, vec, scalar, max_fes, f_target=None):
        super().__init__(vec, scalar)
        self.max_fes, self.f_target = max_fes, f_target
        self.best_f, self.best_x = np.inf, None

    def _check(self, X, f):
        k = min(len(f), self.max_fes - (self.n - len(f)))      # evaluations of this batch within the budget
        if k > 0:
            j = int(np.argmin(f[:k]))
            if f[j] < self.best_f:
                self.best_f, self.best_x = float(f[j]), np.array(X[j], dtype=float)
        if self.f_target is not None and self.best_f <= self.f_target:
            raise TargetReached

    def vectorized(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = super().vectorized(X)
        self._check(X, f)
        return f

    def single(self, x):
        f = super().single(x)
        self._check(np.atleast_2d(np.asarray(x, dtype=float)), np.array([f]))
        return f


def budget_of(settings):
    """Evaluation budget of a run for a suite's config settings."""
    if settings["budget"] == "fe":
        return settings["max_fes"]
    return settings["agents"] * (settings["iterations"] + 1)


def epochs_for_budget(max_fes, n, algo=None):
    """Largest number of epochs T of a mealpy algorithm whose worst-case evaluations stay within
    the budget: N + T c N <= max_fes, c = FES_PER_EPOCH (1 unless listed)."""
    return max(1, (max_fes - n) // (FES_PER_EPOCH.get(algo, 1) * n))


# ---------------------------------------------------------------------------
# One run (executed in a worker process)
# ---------------------------------------------------------------------------

def _atomic_dump(obj, path, as_json=False):
    tmp = path + ".tmp"
    if as_json:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())                       # data on disk before the rename (power loss)
    else:
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, path)


def _readable_result(path):
    """True if a finished run file can be loaded (a power loss can leave an empty/corrupt file)."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh).get("status") == "ok"
    except Exception:
        return False


RTIDE_TRACE_BINS = 200                           # equal parts of the budget in the stored trace of RTI-DE
RTIDE_THETA_EDGES = np.linspace(-1.0, 0.25, 51)  # the factor of a third point lies in [-1, 1/4)


def summarize_rtide_trace(log, max_fes, bins=RTIDE_TRACE_BINS):
    """Compresses the per-generation records of ``rti_de(trace=log)`` for the run file.

    A generation belongs to the part of the budget in which its last evaluation falls. Per part:
    ``generations``; ``evaluated``, ``accepted``, ``improved`` (sums, columns: first trial point,
    half step, parabola point); ``skipped`` (sum); ``mu_F`` and ``n`` after the last generation of
    the part (nan when the part holds no generation). Over the run: histograms of the factor t* of
    the third points that were accepted and rejected (``theta_edges``; ``theta_outside`` counts
    the factors outside the edges, which should not occur)."""
    fes = np.array([g["fes"] for g in log], dtype=np.int64)
    part = np.minimum((fes * bins - 1) // max_fes, bins - 1)

    def summed(key, width):
        out = np.zeros((bins, width), dtype=np.int32)                # counts are at most the budget
        np.add.at(out, part, np.array([g[key] for g in log], dtype=np.int64).reshape(len(log), width))
        return out

    mu_F, n = np.full(bins, np.nan), np.full(bins, np.nan)
    mu_F[part] = [g["mu_F"] for g in log]            # generations are in order: the last one of a part stays
    n[part] = [g["n"] for g in log]
    theta = np.concatenate([g["theta"] for g in log]) if log else np.zeros(0)
    acc = np.concatenate([g["theta_accepted"] for g in log]).astype(bool) if log else np.zeros(0, bool)
    lo, hi = RTIDE_THETA_EDGES[0], RTIDE_THETA_EDGES[-1]
    return dict(fe_edges=np.linspace(0, max_fes, bins + 1), generations=np.bincount(part, minlength=bins),
                evaluated=summed("evaluated", 3), accepted=summed("accepted", 3), improved=summed("improved", 3),
                skipped=summed("skipped", 1)[:, 0], mu_F=mu_F, n=n, theta_edges=RTIDE_THETA_EDGES,
                theta_accepted=np.histogram(theta[acc], RTIDE_THETA_EDGES)[0],
                theta_rejected=np.histogram(theta[~acc], RTIDE_THETA_EDGES)[0],
                theta_outside=int(((theta < lo) | (theta > hi)).sum()))


def _optimize(t, obj, cnt, n, max_fes, log=None):
    """Runs the algorithm on the counted objective; the result is read from the counter.
    ``log``: list for the per-generation records of RTI-DE and its variants (see rti_de)."""
    algo = t["algo"]
    if algo == PROPOSED or algo in VARIANTS:
        from RTIDE import rti_de
        opts = dict(VARIANTS[algo][0]) if algo in VARIANTS else {}
        if "n_init_per_dim" in opts:
            opts["n_init"] = opts.pop("n_init_per_dim") * len(obj["lb"])
        rti_de(cnt.vectorized, obj["lb"], obj["ub"], max_fes, seed=t["seed"], trace=log, **opts)
    elif algo == "LSHADE":
        from lshade_reference import lshade
        lshade(cnt.vectorized, obj["lb"], obj["ub"], max_fes, seed=t["seed"])
    elif algo == "DE":
        from de_reference import de
        de(cnt.vectorized, obj["lb"], obj["ub"], max_fes, seed=t["seed"], n=n)
    else:
        import importlib
        from mealpy import FloatVar
        mod, cls = MEALPY[algo]
        model = getattr(importlib.import_module(mod), cls)(epoch=epochs_for_budget(max_fes, n, algo), pop_size=n)
        problem = {"obj_func": cnt.single, "bounds": FloatVar(lb=tuple(obj["lb"]), ub=tuple(obj["ub"])),
                   "minmax": "min", "log_to": None}
        model.solve(problem, mode="single", seed=t["seed"], termination={"max_fe": max_fes})


def run_task(t):
    warnings.filterwarnings("ignore")
    base = dict(suite=t["suite"], fid=t["fid"], algo=t["algo"], run=t["run"], seed=t["seed"])
    t0 = time.perf_counter()
    cnt = None
    try:
        obj = load_objective(t["suite"], t["fid"], t["dim"])
        n, max_fes = t["agents"], t["max_fes"]
        target = obj["f_star"] + ERROR_FLOOR if t["target_stop"] and obj["f_star"] is not None else None
        cnt = BudgetCounter(obj["vec"], obj["scalar"], max_fes, target)
        at_target, log = False, []
        try:
            _optimize(t, obj, cnt, n, max_fes, log)
        except TargetReached:
            at_target = True
        f = cnt.best_f
        trace = np.asarray(cnt.trace, float).reshape(-1, 2)
        res = dict(base, status="ok", f=float(f),
                   error=(float(f) - obj["f_star"]) if obj["f_star"] is not None else None,
                   x=cnt.best_x, trace=trace[trace[:, 0] <= max_fes],
                   fes=min(cnt.n, max_fes), fes_evaluated=cnt.n, max_fes=max_fes,
                   stopped_at_target=at_target, invalid=cnt.invalid, time=time.perf_counter() - t0)
        if log:                                           # RTI-DE and its variants only
            res["rtide_trace"] = summarize_rtide_trace(log, max_fes)
        _atomic_dump(res, t["path"])
        if os.path.exists(t["err_path"]):
            os.remove(t["err_path"])
        return dict(base, status="ok", f=res["f"], fes=res["fes"], time=res["time"])
    except Exception as e:                                # recorded, never dropped silently
        err = dict(base, status="error", error=repr(e), traceback=traceback.format_exc(),
                   fes=cnt.n if cnt else 0, time=time.perf_counter() - t0)
        _atomic_dump(err, t["err_path"], as_json=True)
        return dict(base, status="error", f=np.nan, fes=err["fes"], time=err["time"], message=repr(e))


# ---------------------------------------------------------------------------
# Config, provenance, task list
# ---------------------------------------------------------------------------

def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def provenance():
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "host": socket.gethostname(), "cpu_count": os.cpu_count()}
    for mod in ("numpy", "scipy", "mealpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        out["git_commit"] = git("rev-parse", "--short", "HEAD") or None
        out["optimizer_py_uncommitted_changes"] = bool(git("status", "--porcelain", OPTIMIZER_PATH))
    except Exception:
        out["git_commit"] = None
    return out


def suite_dir_name(suite, dim):
    return f"{suite}_D{dim}" if suite in CEC_SUITES else suite


def prepare_suite(suite, dim, args, prov):
    """Create/verify <out>/<suite>/config.json. Refuses to mix different settings or code."""
    sdir = os.path.join(args.out, suite_dir_name(suite, dim))
    os.makedirs(sdir, exist_ok=True)
    settings = {"suite": suite, "dim": dim, "agents": args.agents,
                "iterations": args.iterations if args.budget == "iter" else None,
                "budget": args.budget, "max_fes": args.fes.get(suite) if args.budget == "fe" else None,
                "seed_base": args.seed, "target_stop": suite in CEC_SUITES and not args.no_target_stop,
                "mealpy": prov.get("mealpy"),
                **{k: _sha256(os.path.join(ROOT, *p)) for k, p in CODE_FILES.items()}}
    settings["budget_fes"] = budget_of(settings)
    settings["population"] = {"RTI-DE, L-SHADE": f"{18 * dim} to 4", "all others": args.agents}
    settings["mealpy_epochs"] = {a: epochs_for_budget(settings["budget_fes"], args.agents, a) for a in MEALPY}
    if suite in CEC_SUITES:
        settings["cec_source"] = cec_source(suite)
    if suite == "composite_contiguity":
        import laminate_buckling
        settings["contiguity_penalty"] = laminate_buckling.PENALTY
    cfg_path = os.path.join(sdir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        diff = {k: (cfg["settings"].get(k), settings.get(k)) for k in LOCKED_SETTINGS
                if cfg["settings"].get(k) != settings.get(k)}
        if diff:
            msg = "\n".join(f"    {k}: stored={a!r}  now={b!r}" for k, (a, b) in diff.items())
            sys.exit(f"ERROR: {cfg_path} was created with different settings or code:\n{msg}\n"
                     f"  Results must not be mixed. Use another --out directory "
                     f"(or restore the original settings/code).")
    else:
        cfg = {"settings": settings, "sessions": []}
    cfg["sessions"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "command": " ".join(sys.argv), "runs_requested": args.runs,
                            "algorithms": args.algorithms, "workers": args.workers, **prov})
    _atomic_dump(cfg, cfg_path, as_json=True)
    return sdir


def run_paths(sdir, fid, algo, run):
    d = os.path.join(sdir, "runs", fid, algo)
    return os.path.join(d, f"run_{run:02d}.pkl"), os.path.join(d, f"run_{run:02d}.error.json")


def check_and_time(suite, fid, dim, n_eval=20):
    """Self-check for CEC functions (f(x*) = F*) and a per-evaluation time (s)."""
    obj = load_objective(suite, fid, dim)
    if obj["x_star"] is not None:
        err = obj["scalar"](obj["x_star"]) - obj["f_star"]
        if abs(err) > 1e-6:
            sys.exit(f"ERROR: {suite} {fid}: f(x*) - F* = {err:.3e} (data problem).")
    X = np.random.default_rng(0).uniform(obj["lb"], obj["ub"], (n_eval, obj["dim"]))
    t0 = time.perf_counter()
    obj["vec"](X)
    return (time.perf_counter() - t0) / n_eval, obj["dim"]


def expected_fes(args, suite):
    return args.fes[suite] if args.budget == "fe" else args.agents * (args.iterations + 1)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def collect(sdir):
    """{fid: {algo: [run dicts sorted by run]}}, {fid: {algo: [error dicts]}}."""
    ok, errors = {}, {}
    root = os.path.join(sdir, "runs")
    if not os.path.isdir(root):
        return ok, errors
    for fid in sorted(os.listdir(root), key=lambda s: (len(s), s)):
        for algo in sorted(os.listdir(os.path.join(root, fid))):
            d = os.path.join(root, fid, algo)
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if name.endswith(".pkl"):
                    with open(p, "rb") as fh:
                        ok.setdefault(fid, {}).setdefault(algo, []).append(pickle.load(fh))
                elif name.endswith(".error.json"):
                    with open(p, encoding="utf-8") as fh:
                        errors.setdefault(fid, {}).setdefault(algo, []).append(json.load(fh))
    for fid in ok:
        for algo in ok[fid]:
            ok[fid][algo].sort(key=lambda r: r["run"])
    return ok, errors


def _cliffs_delta(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(np.mean(np.sign(a - b)))


def _cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def _mannwhitney(a, b):
    """Two-sided Mann-Whitney U test of two independent samples; nan when all values are equal."""
    from scipy import stats
    big = 1e300                                            # infeasible (inf) -> worst finite value
    a = np.where(np.isfinite(a), a, big)
    b = np.where(np.isfinite(b), b, big)
    if np.all(a == a[0]) and np.all(b == a[0]):
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _verdict(p, a, b):
    """'+' if a is significantly lower (better) than b, '-' if higher, '≈' otherwise.
    The direction is the sign of Cliff's delta, i.e. of U - n_a n_b / 2, the statistic of the
    test itself (medians can tie, e.g. both 0 after the 1e-8 threshold)."""
    if not (p < 0.05):
        return "≈"
    big = 1e300                                            # as in _mannwhitney
    d = _cliffs_delta(np.where(np.isfinite(a), a, big), np.where(np.isfinite(b), b, big))
    return "+" if d < 0 else ("-" if d > 0 else "≈")


def summarize_suite(sdir, suite, runs_expected):
    import pandas as pd
    from scipy import stats

    ok, errors = collect(sdir)
    if not ok:
        print(f"  {sdir}: no finished runs yet.")
        return
    with open(os.path.join(sdir, "results.pkl"), "wb") as fh:
        pickle.dump({"runs": ok, "errors": errors}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    is_cec = suite in CEC_SUITES
    maximize = suite.startswith("composite")
    value_name = "error (f - F*)" if is_cec else ("lambda_cb" if maximize else "cost")

    def cec_error(runs):                   # CEC2017/CEC2022 reports: errors below 1e-8 count as 0
        e = np.array([r["error"] for r in runs])        # (raw values stay in the run files)
        return np.where(e < 1e-8, 0.0, e)

    def values(runs):                      # reported values (error, lambda_cb or cost)
        if is_cec:
            return cec_error(runs)
        f = np.array([r["f"] for r in runs])
        return -f if maximize else f

    def losses(runs):                      # lower is better, for ranks and tests
        return cec_error(runs) if is_cec else np.array([r["f"] for r in runs])

    fids = [f for f in function_ids(suite) if f in ok]
    # algorithms of this directory: the comparison list, or the proposed algorithm with its variants
    present = {a for f in ok for a in ok[f]}
    algos = [a for a in ALL_ALGORITHMS + list(VARIANTS) if a in present]
    rows, incomplete = [], []
    for fid in fids:
        for algo in algos:
            runs = ok[fid].get(algo, [])
            n_err = len(errors.get(fid, {}).get(algo, []))
            if len(runs) < runs_expected or n_err:
                incomplete.append(f"{fid}/{algo}: {len(runs)}/{runs_expected} ok, {n_err} failed")
            if not runs:
                continue
            v = values(runs)
            fin = v[np.isfinite(v)]
            rows.append({"function": fid, "algorithm": algo, "runs_ok": len(runs), "runs_failed": n_err,
                         "feasible_runs": int(np.isfinite(v).sum()),
                         "mean": fin.mean() if fin.size else np.nan,
                         "std": fin.std(ddof=1) if fin.size > 1 else np.nan,
                         "median": np.median(fin) if fin.size else np.nan,
                         "best": (fin.max() if maximize else fin.min()) if fin.size else np.nan,
                         "worst": (fin.min() if maximize else fin.max()) if fin.size else np.nan,
                         "mean_FEs": np.mean([r["fes"] for r in runs]),
                         "min_FEs": int(np.min([r["fes"] for r in runs])),
                         "max_FEs": int(np.max([r["fes"] for r in runs])),
                         "mean_invalid_candidates": np.mean([r["invalid"] for r in runs]),
                         "mean_time_s": np.mean([r["time"] for r in runs])})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(sdir, "summary.csv"), index=False)

    # Complete functions only: every algorithm has the same runs 0..R-1.
    complete = [fid for fid in fids
                if all(len(ok[fid].get(a, [])) >= runs_expected for a in algos)]
    if complete:
        cmp_rows, ranks = [], []
        for fid in complete:
            means = {a: np.mean(values(ok[fid][a][:runs_expected])) for a in algos}
            stds = {a: np.std(values(ok[fid][a][:runs_expected]), ddof=1) for a in algos}
            cmp_rows += [{"function": fid, "metric": "mean", **means}, {"function": fid, "metric": "std", **stds}]
            ranks.append(stats.rankdata([np.mean(losses(ok[fid][a][:runs_expected])) for a in algos]))
        mean_rank = np.mean(ranks, axis=0)
        cmp_rows.append({"function": f"({len(complete)} functions)", "metric": "Friedman mean rank",
                         **dict(zip(algos, np.round(mean_rank, 3)))})
        if len(complete) >= 2:
            chi2, p = stats.friedmanchisquare(*np.array(ranks).T)
            cmp_rows.append({"function": "", "metric": f"Friedman chi2={chi2:.3f} p={p:.3e}"})
        pd.DataFrame(cmp_rows).to_csv(os.path.join(sdir, "comparison_table.csv"), index=False)

        others = [a for a in algos if a != PROPOSED]
        st_rows, wtl = [], {a: [0, 0, 0] for a in others}
        for fid in (complete if PROPOSED in algos else []):
            m = losses(ok[fid][PROPOSED][:runs_expected])
            for comp in others:
                c = losses(ok[fid][comp][:runs_expected])
                p = _mannwhitney(m, c)
                delta = _cliffs_delta(m, c)          # < 0: values of the proposed algorithm tend to be lower
                verdict = _verdict(p, m, c)
                wtl[comp]["+≈-".index(verdict)] += 1
                st_rows.append({"function": fid, "competitor": comp, "mannwhitney_p": p,
                                "cliffs_delta": delta, "cohens_d": _cohens_d(m, c), "verdict": verdict})
        pd.DataFrame(st_rows).to_csv(os.path.join(sdir, "stats_vs_proposed.csv"), index=False)
        pd.DataFrame([{"competitor": k, "proposed_better(+)": v[0], "no_difference(≈)": v[1],
                       "proposed_worse(-)": v[2]} for k, v in wtl.items()]).to_csv(
            os.path.join(sdir, "wtl_vs_proposed.csv"), index=False)

    if not is_cec:
        from run_engineering import load_problem, summarize
        prob = load_problem(suite)
        eng = summarize({a: ok[suite][a] for a in algos if a in ok.get(suite, {})}, prob)
        pd.DataFrame(eng).to_csv(os.path.join(sdir, "engineering_summary.csv"), index=False)

    print(f"\n  {sdir}: {value_name}; {len(complete)}/{len(function_ids(suite))} functions complete "
          f"({runs_expected} runs x {len(algos)} algorithms).")
    if complete:
        print("  Friedman mean rank (per-function means):",
              ", ".join(f"{a} {r:.2f}" for a, r in sorted(zip(algos, mean_rank), key=lambda x: x[1])))
    if incomplete:
        print(f"  Incomplete or failed ({len(incomplete)}):")
        for line in incomplete[:20]:
            print("    " + line)
        if len(incomplete) > 20:
            print(f"    ... and {len(incomplete) - 20} more (see summary.csv)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", choices=ALL_SUITES, default=list(ALL_SUITES))
    ap.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS + list(VARIANTS), default=ALL_ALGORITHMS)
    ap.add_argument("--functions", nargs="+", default=None, help="e.g. F1 F3 (CEC suites only)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--agents", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--budget", choices=("iter", "fe"), default="iter")
    ap.add_argument("--max-fes", nargs="+", default=None,
                    help="with --budget fe: one number for all suites, or suite=number entries")
    ap.add_argument("--dim-cec2022", type=int, default=20)
    ap.add_argument("--dim-cec2017", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="run r uses seed + r")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join("results_cache", "paper"))
    ap.add_argument("--dry-run", action="store_true", help="list the work and a rough duration, run nothing")
    ap.add_argument("--summarize-only", action="store_true", help="only rebuild the tables")
    ap.add_argument("--no-target-stop", action="store_true",
                    help="CEC suites: spend the whole budget even when the error is already below 1e-8")
    args = ap.parse_args()
    if args.budget == "iter" and args.max_fes:
        ap.error("--max-fes is only used with --budget fe")
    args.fes = {}
    for item in args.max_fes or []:
        suite, _, num = item.rpartition("=")
        try:
            num = int(float(num))
        except ValueError:
            ap.error(f"--max-fes: cannot read {item!r}")
        for s in ([suite] if suite else args.suites):
            if s not in ALL_SUITES:
                ap.error(f"--max-fes: unknown suite {s!r}")
            args.fes[s] = num
    if args.budget == "fe":
        missing = [s for s in args.suites if s not in args.fes]
        if missing:
            ap.error(f"--budget fe needs --max-fes for: {', '.join(missing)}")
    return args


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.out = os.path.abspath(args.out)
    dims = {"cec2022": args.dim_cec2022, "cec2017": args.dim_cec2017}

    suites = []
    for suite in args.suites:
        dim = dims.get(suite)
        fids = function_ids(suite)
        if args.functions and suite in CEC_SUITES:
            unknown = set(args.functions) - set(fids)
            if unknown and len(args.suites) == 1:
                sys.exit(f"ERROR: unknown {suite} functions {sorted(unknown)}")
            fids = [f for f in fids if f in args.functions]
        suites.append((suite, dim, fids))

    if args.summarize_only:
        for suite, dim, _ in suites:
            sdir = os.path.join(args.out, suite_dir_name(suite, dim if dim else 0))
            if os.path.isdir(sdir):
                summarize_suite(sdir, suite, args.runs)
        return

    prov = provenance()
    print(f"RTI-DE experiments -> {args.out}")
    print(f"  budget={args.budget}  agents={args.agents}  "
          + (f"iterations={args.iterations}" if args.budget == "iter" else f"max_fes={args.fes}")
          + f"  runs={args.runs}  seed={args.seed}+r  workers={args.workers}")
    print(f"  git {prov.get('git_commit')}  optimizer.py uncommitted changes: "
          f"{prov.get('optimizer_py_uncommitted_changes')}  mealpy {prov.get('mealpy')}")

    tasks, skipped, cost = [], 0, {}
    for suite, dim, fids in suites:
        if not fids:
            print(f"  {suite}: no functions selected, skipped")
            continue
        for fid in fids:
            eval_s, fdim = check_and_time(suite, fid, dim)
            dim_used = fdim
            for algo in args.algorithms:
                cost[(suite, fid, algo)] = expected_fes(args, suite) * (eval_s + (2e-4 if algo in MEALPY else 2e-5))
        sdir = (os.path.join(args.out, suite_dir_name(suite, dim)) if args.dry_run
                else prepare_suite(suite, dim_used if suite not in CEC_SUITES else dim, args, prov))
        for fid in fids:
            for algo in args.algorithms:
                for r in range(args.runs):
                    path, err_path = run_paths(sdir, fid, algo, r)
                    if os.path.exists(path):
                        if _readable_result(path):
                            skipped += 1
                            continue
                        print(f"  WARNING: unreadable result {path} - it will be run again")
                        if not args.dry_run:
                            os.remove(path)
                    tasks.append(dict(suite=suite, fid=fid, algo=algo, run=r, seed=args.seed + r,
                                      dim=dim_used if suite not in CEC_SUITES else dim, agents=args.agents,
                                      max_fes=expected_fes(args, suite),
                                      target_stop=suite in CEC_SUITES and not args.no_target_stop,
                                      path=path, err_path=err_path, cost=cost[(suite, fid, algo)]))
        print(f"  {suite_dir_name(suite, dim if dim else 0):<22} {len(fids):>2} functions x "
              f"{len(args.algorithms)} algorithms x {args.runs} runs")

    total_cost = sum(t["cost"] for t in tasks)
    print(f"  to run: {len(tasks)} runs ({skipped} already finished)."
          f"  Rough estimate: ~{total_cost / 3600:.1f} CPU-hours, "
          f"~{total_cost / 3600 / max(1, args.workers):.1f} h with {args.workers} workers "
          f"(evaluation time plus a rough allowance for the overhead of the algorithms).")
    if args.dry_run or not tasks:
        if not tasks and not args.dry_run:
            for suite, dim, _ in suites:
                sdir = os.path.join(args.out, suite_dir_name(suite, dim if dim else 0))
                summarize_suite(sdir, suite, args.runs)
        return

    tasks.sort(key=lambda t: -t["cost"])                 # longest first: better load balance
    for t in tasks:
        os.makedirs(os.path.dirname(t["path"]), exist_ok=True)

    log_path = os.path.join(args.out, "run_log.txt")
    done_cost, n_err, t0 = 0.0, 0, time.time()
    ex = ProcessPoolExecutor(max_workers=args.workers)
    try:
        futs = {ex.submit(run_task, {k: v for k, v in t.items() if k != "cost"}): t for t in tasks}
        with open(log_path, "a", encoding="utf-8") as log:
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:                   # worker crash (e.g. killed process)
                    r = dict(status="error", f=np.nan, fes=0, time=0.0, message=repr(e))
                done_cost += t["cost"]
                line = (f"{time.strftime('%H:%M:%S')} {t['suite']} {t['fid']} {t['algo']} run{t['run']:02d} "
                        f"{r['status']} f={r['f']:.6e} FEs={r['fes']} {r['time']:.1f}s")
                log.write(line + "\n")
                log.flush()
                if r["status"] != "ok":
                    n_err += 1
                    print(f"\nERROR: {t['suite']} {t['fid']} {t['algo']} run {t['run']}: {r.get('message')}")
                elapsed = time.time() - t0
                eta = elapsed / done_cost * (total_cost - done_cost) if done_cost else 0
                print(f"\r  {i}/{len(tasks)} runs  errors={n_err}  elapsed {elapsed / 3600:.2f} h  "
                      f"ETA ~{eta / 3600:.2f} h   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping: finished runs are saved; run the same command again to continue.")
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)
    ex.shutdown()
    print(f"\nFinished in {(time.time() - t0) / 3600:.2f} h, {n_err} failed run(s)"
          + (" - they are retried when the command is run again." if n_err else "."))

    for suite, dim, _ in suites:
        sdir = os.path.join(args.out, suite_dir_name(suite, dim if dim else 0))
        summarize_suite(sdir, suite, args.runs)


if __name__ == "__main__":
    main()
