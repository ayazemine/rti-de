#!/usr/bin/env python3
"""
Paper tables for RTI-DE, from the results of ``run_experiments.py`` (sections F and G: from the
ablation and the parameter-sensitivity runs; ``--only F G`` prints the commands when they are missing).

Every table is written three times:

    outputs/tables/<name>.csv    raw numbers, full precision
    outputs/tables/<name>.tex    booktabs table for \\input in a LaTeX document
    outputs/tables/<name>.json   sources, settings, git commit, notes

The .tex files need \\usepackage{booktabs,longtable,graphicx} in the preamble.
Nothing is typed in by hand: results come from the run files, problem data
from the code. Bold = best (lowest) mean of a row, computed.

Usage (from experiments/)
-------------------------
    python -W ignore scripts/make_tables.py --list
    python -W ignore scripts/make_tables.py                    # all tables
    python -W ignore scripts/make_tables.py --only B1 C1 E2
    python -W ignore scripts/make_tables.py --dry-run          # build everything, write nothing
"""

import argparse
import inspect
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as fl  # noqa: E402
from figlib import ALGORITHMS, ERROR_FLOOR, PROPOSED, display, rx  # noqa: E402

COMPETITORS = ALGORITHMS[1:]
SYM_TEX = {"+": "$+$", "-": "$-$", "≈": r"$\approx$"}
MID = None                                   # a None row is a \midrule


# ---------------------------------------------------------------------------
# Formatting and statistics
# ---------------------------------------------------------------------------

def sci(x):
    if x is None or np.isnan(x):
        return "---"
    if np.isinf(x):
        return r"$\infty$"
    return "0" if x == 0 else f"{x:.2E}"


def fixed(x, nd=2):
    return "---" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def pval(p):
    if p is None or np.isnan(p):
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.2g}"


def shown(algos):
    """Printed names of a list of algorithms (table headers)."""
    return [display(a) for a in algos]


def bold(s):
    return r"\textbf{" + s + "}"


def tt(s):
    return r"\texttt{" + str(s).replace("_", r"\_").replace("%", r"\%").replace("^", r"\textasciicircum{}") + "}"


def holm(p):
    """Holm step-down adjusted p-values; NaN entries stay NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        out[i] = min(1.0, running)
    return out


def verdict(p, a, b):
    """'+' if a (the proposed algorithm) is significantly better (lower) than b, '-' if worse, '≈' otherwise."""
    return rx._verdict(p, a, b)


def compare(base, others):
    """Mann-Whitney U test of ``base`` against each array in ``others`` (dict). Returns {name: dict}."""
    names = list(others)
    p = np.array([rx._mannwhitney(base, others[k]) for k in names])
    ph = holm(p)
    return {k: {"p": p[i], "p_holm": ph[i], "verdict": verdict(p[i], base, others[k]),
                "verdict_holm": verdict(ph[i], base, others[k]),
                "cliffs_delta": rx._cliffs_delta(base, others[k]),
                "cohens_d": rx._cohens_d(base, others[k])} for i, k in enumerate(names)}


def holm_sizes(st, fids, names):
    """{fid: size of the Holm family}: the comparisons whose test is defined (finite p, see holm)."""
    return {f: int(sum(np.isfinite(st[f][n]["p"]) for n in names)) for f in fids}


def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def holm_smaller(sizes, m):
    """'5 on F3; 4 on F5 and F9' for the functions with fewer than m comparisons ('' if there are none)."""
    small = {}
    for f, k in sizes.items():
        if k < m:
            small.setdefault(k, []).append(f)
    return "; ".join((f"{k} on " if k else "no test on ") + join_and(fs)
                     for k, fs in sorted(small.items(), reverse=True))


HOLM_UNDEFINED = "or fewer where the test is not defined because all runs of both algorithms give the same value"


def holm_family_text(sizes, m):
    smaller = holm_smaller(sizes, m)
    return f"{m} comparisons per function" + (f", {HOLM_UNDEFINED}: {smaller}" if smaller else "")


def wtl_text(counts):
    return "/".join(str(c) for c in counts)


def count_wtl(verdicts):
    return [sum(v == s for v in verdicts) for s in "+≈-"]


def mean_ranks(M):
    """Average rank of each column of a (functions, algorithms) matrix of means (1 = best)."""
    from scipy import stats
    R = np.array([stats.rankdata(row) for row in M])
    return R.mean(axis=0), R


def friedman_cd(M):
    from scipy import stats
    avg, R = mean_ranks(M)
    n, k = M.shape
    chi2, p = stats.friedmanchisquare(*R.T)
    q = float(stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2.0))
    return avg, float(chi2), float(p), q * np.sqrt(k * (k + 1) / (6.0 * n))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Writer:
    def __init__(self, out_dir, dry_run):
        self.out_dir, self.dry_run, self.written = out_dir, dry_run, []

    def write(self, name, csv, header, rows, caption, notes=(), suites=(), meta=None, align=None,
              extra_sources=()):
        """csv: DataFrame; header/rows: TeX-ready strings (a None row is a \\midrule)."""
        # a None row is a \midrule, a str row is a raw TeX line (e.g. a \multicolumn heading)
        body_rows = [r for r in rows if isinstance(r, list)]
        ncol = len(header)
        if any(len(r) != ncol for r in body_rows):
            raise RuntimeError(f"{name}: a row does not have {ncol} cells.")
        tex = self._tex(name, header, rows, caption, notes, align or ("l" + "r" * (ncol - 1)),
                        long=len(body_rows) > 42)
        if self.dry_run:
            print(f"    [dry run] {name}: {len(csv)} csv rows, {len(body_rows)} x {ncol} tex cells, not written")
            return
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, name)
        csv.to_csv(base + ".csv", index=False)
        with open(base + ".tex", "w", encoding="utf-8") as fh:
            fh.write(tex)
        info = {"table": name, "caption": caption, "notes": list(notes),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(sys.argv),
                "git_commit": fl._git("rev-parse", "--short", "HEAD"),
                "optimizer_sha256": rx._sha256(fl.OPTIMIZER),
                "sources": [{"suite": s.name, "directory": os.path.relpath(s.dir, fl.ROOT),
                             "run_files": s.n_files, "settings": s.settings} for s in suites]
                           + [{"file": os.path.relpath(p, fl.ROOT)} for p in extra_sources],
                "details": meta or {}}
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, ensure_ascii=False, default=fl._jsonable)
        self.written.append(name)
        print(f"    wrote {os.path.relpath(base, fl.ROOT)}.csv/.tex/.json")

    @staticmethod
    def _tex(name, header, rows, caption, notes, align, long):
        head = " & ".join(header) + r" \\"
        lines = [r"\midrule" if r is MID else (r if isinstance(r, str) else " & ".join(r) + r" \\")
                 for r in rows]
        note = ""
        if notes:
            note = "\\par\\smallskip\\noindent\\parbox{\\linewidth}{\\footnotesize " + " ".join(notes) + "}\n"
        top = f"% Generated by scripts/make_tables.py - do not edit by hand ({name})\n"
        if long:
            return (top + "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
                    f"\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}\\label{{tab:{name}}}\\\\\n"
                    f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n\\toprule\n{head}\n\\midrule\n\\endhead\n"
                    "\\bottomrule\n\\endfoot\n" + "\n".join(lines) + "\n\\end{longtable}\n" + note + "}\n")
        tab = (f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n" + "\n".join(lines)
               + "\n\\bottomrule\n\\end{tabular}")
        if len(header) > 7:
            tab = "\\resizebox{\\linewidth}{!}{%\n" + tab + "}"
        return (top + "\\begin{table}[htbp]\n\\centering\\footnotesize\n"
                f"\\caption{{{caption}}}\\label{{tab:{name}}}\n" + tab + "\n" + note + "\\end{table}\n")


def df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared computations
# ---------------------------------------------------------------------------

_CACHE = {}


def values(S, fid, algo, budget=None):
    key = (S.name, fid, algo, budget)
    if key not in _CACHE:
        _CACHE[key] = S.final(fid, algo) if budget is None else S.at_fes(fid, algo, budget)
    return _CACHE[key]


def suite_stats(S, budget=None):
    """{fid: {competitor: stats}} of the proposed algorithm against every competitor."""
    key = ("stats", S.name, budget)
    if key not in _CACHE:
        _CACHE[key] = {fid: compare(values(S, fid, PROPOSED, budget),
                                    {a: values(S, fid, a, budget) for a in COMPETITORS}) for fid in S.fids}
    return _CACHE[key]


def means_matrix(S, budget=None):
    return np.array([[values(S, fid, a, budget).mean() for a in ALGORITHMS] for fid in S.fids])


TITLE = {"cec2022": "CEC2022 ($D = 20$)", "cec2017": "CEC2017 ($D = 100$)"}
PAIRED = "Two-sided Wilcoxon rank-sum (Mann-Whitney $U$) test on the runs, $\\alpha = 0.05$."
SYMBOLS = (f"$+$: {PROPOSED} significantly better; $-$: significantly worse; $\\approx$: no significant difference "
           "(decided on the unadjusted $p$-value).")
FLOOR_NOTE = f"Errors $f - F^*$ below {ERROR_FLOOR:g} are reported as 0."


def cec2017_category(fid):
    k = int(fid[1:])
    return "Unimodal" if k <= 3 else "Multimodal" if k <= 10 else "Hybrid" if k <= 20 else "Composition"


# ---------------------------------------------------------------------------
# A. Experimental setup
# ---------------------------------------------------------------------------

def table_A1(ctx):
    from cec2022_functions import CEC2022_INFO, SEARCH_RANGE
    rng = f"[{SEARCH_RANGE[0]:g}, {SEARCH_RANGE[1]:g}]"
    rows = [{"No": i.fid, "Function": i.name, "Category": i.category, "F*": i.f_star, "Search range": rng}
            for i in CEC2022_INFO]
    ctx.w.write("A1_cec2022_functions", df(rows), ["No.", "Function", "Category", "$F^*$", "Search range"],
                [[r["No"], r["Function"], r["Category"], f"{r['F*']:g}", f"${rng}^D$"] for r in rows],
                "CEC2022 benchmark functions.", align="lllrl",
                notes=["Names and $F^*$ as in the CEC2022 technical report (Kumar et al., 2021); "
                       "functions evaluated by the official C code (cec22_test_func.cpp)."],
                extra_sources=[os.path.join(fl.ROOT, "benchmarks", "cec2022_functions.py")])


def table_A2(ctx):
    from cec_official_functions import get_official_function
    S = ctx.load("cec2017")
    if not S.settings.get("cec_source"):
        raise RuntimeError(f"{S.dir}: config.json has no cec_source entry; the runs must use the "
                           "official CEC2017 code.")
    D = S.settings["dim"]
    rows = []
    for fid in S.fids:
        f = get_official_function("cec2017", fid, D)
        rows.append({"No": fid, "Function": f.name, "Category": cec2017_category(fid), "F*": f.f_bias,
                     "Search range": f"[{float(f.lb[0]):g}, {float(f.ub[0]):g}]"})
    ctx.w.write("A2_cec2017_functions", df(rows), ["No.", "Function", "Category", "$F^*$", "Search range"],
                [[r["No"], r["Function"], r["Category"], f"{r['F*']:g}",
                  "$" + r["Search range"] + "^D$"] for r in rows],
                "CEC2017 benchmark functions used in this study.", align="lllrl", suites=[S],
                notes=["Numbering of the official C code (cec17_test_func.cpp), which evaluates the functions; "
                       "$F^* = 100 \\times$ the function number. "
                       "Names from Table I of the CEC2017 technical report (Awad et al., 2016)."],
                meta={"category_rule": "F1-F3 unimodal, F4-F10 multimodal, F11-F20 hybrid, F21-F30 composition "
                                       "(Table I of the CEC2017 technical report, in the numbering of the C code)"})


def table_A3(ctx):
    names = ["cec2022", "cec2017", "composite", "composite_contiguity", "uav"]
    suites = [ctx.load(n) for n in names]
    ses = [s.config["sessions"][-1] for s in suites]
    spec = [("Dimension $D$", lambda s, e: s.settings["dim"]),
            ("Population size", lambda s, e: "; ".join(f"{a}: {n}" for a, n in s.settings["population"].items())),
            ("Budget (FEs)", lambda s, e: f"{s.budget:,}".replace(",", "\\,")),
            ("Budget set by", lambda s, e: "MaxFES" if s.settings["budget"] == "fe"
             else f"$N(T + 1)$, $N = {s.settings['agents']}$, $T = {s.settings['iterations']}$"),
            ("Stop at error $< 10^{-8}$", lambda s, e: "yes" if s.settings["target_stop"] else "no"),
            ("Independent runs", lambda s, e: s.n_runs),
            ("Seeds", lambda s, e: f"{s.settings['seed_base']}--{s.settings['seed_base'] + s.n_runs - 1}"),
            ("Functions", lambda s, e: len(s.fids)),
            ("Python / numpy / scipy", lambda s, e: f"{e.get('python')} / {e.get('numpy')} / {e.get('scipy')}"),
            ("mealpy", lambda s, e: str(e.get("mealpy"))),
            ("CPU cores / workers", lambda s, e: f"{e.get('cpu_count')} / {e.get('workers')}")]
    head = ["Setting", "CEC2022", "CEC2017", "Laminate", "Laminate (contiguity)", "UAV"]
    rows, csv = [], []
    for label, fn in spec:
        vals = [str(fn(s, e)) for s, e in zip(suites, ses)]
        rows.append([label] + vals)
        csv.append(dict(zip(head, [label] + vals)))
    ctx.w.write("A3_settings", df(csv), head, rows, "Experimental settings.", align="llllll", suites=suites,
                notes=["The reported value of a run is the best value among its first Budget function "
                       "evaluations. RTI-DE and L-SHADE reduce their population linearly from the first to the "
                       "second value; the other algorithms keep theirs."],
                meta={"platform": ses[0].get("platform"), "host": ses[0].get("host"),
                      "git_commit": ses[0].get("git_commit")})


def table_A4(ctx):
    import importlib
    import mealpy
    from RTIDE import rti_de
    from de_reference import de
    from lshade_reference import lshade

    def defaults(fn, skip):
        return [(k, p.default) for k, p in inspect.signature(fn).parameters.items()
                if k not in skip and p.default is not inspect.Parameter.empty]

    # trace: a debugging hook of rti_de (records the run, changes nothing), not an algorithm parameter
    skip = ("self", "func", "lb", "ub", "max_fes", "seed", "epoch", "pop_size", "kwargs", "trace")
    rows = []
    for algo, fn, impl in ((PROPOSED, rti_de, "RTIDE/optimizer.py"), ("DE", de, "benchmarks/de_reference.py"),
                           ("LSHADE", lshade, "benchmarks/lshade_reference.py")):
        for k, v in defaults(fn, skip):
            if algo == "DE" and k == "n":
                continue                             # the population size is a run setting (Table A3)
            rows.append({"Algorithm": algo, "Parameter": k, "Value": "18 D" if v is None else str(v),
                         "Implementation": impl})
    rows.insert(len(defaults(rti_de, skip)), {"Algorithm": PROPOSED, "Parameter": "F", "Value": "N(mu_F, 0.15^2) in [0.05, 1], mu_F(0) = 0.5",
                                              "Implementation": "RTIDE/optimizer.py"})
    for a, (mod, cls_name) in rx.MEALPY.items():
        cls = getattr(importlib.import_module(mod), cls_name)
        impl = f"mealpy {mealpy.__version__} {cls.__module__.split('.', 1)[1]}.{cls.__name__}"
        pars = defaults(cls.__init__, skip)
        if not pars:
            rows.append({"Algorithm": a, "Parameter": "—", "Value": "—", "Implementation": impl})
        rows += [{"Algorithm": a, "Parameter": k, "Value": str(v), "Implementation": impl} for k, v in pars]
    order = {a: i for i, a in enumerate(ALGORITHMS)}
    rows.sort(key=lambda r: order[r["Algorithm"]])
    tex, last = [], None
    for r in rows:
        if last is not None and r["Algorithm"] != last:
            tex.append(MID)
        par = "---" if r["Parameter"] == "—" else tt(r["Parameter"])
        val = "---" if r["Value"] == "—" else tt(r["Value"])
        tex.append([display(r["Algorithm"]) if r["Algorithm"] != last else "", par, val, tt(r["Implementation"])])
        last = r["Algorithm"]
    ctx.w.write("A4_parameters", df(rows), ["Algorithm", "Parameter", "Value", "Implementation"], tex,
                "Parameter settings of the algorithms.", align="llll",
                notes=["Values are the defaults of the functions and classes that were run, read from the code. "
                       "\\texttt{n\\_init} = 18 D is the initial population size; population sizes are in "
                       "Table~\\ref{tab:A3_settings}."],
                meta={"warning": "library defaults are not claimed to equal the settings of the original papers"},
                extra_sources=[fl.OPTIMIZER, os.path.join(fl.ROOT, "benchmarks", "de_reference.py"),

                               os.path.join(fl.ROOT, "benchmarks", "lshade_reference.py")])


def table_A5(ctx):
    names = ["cec2022", "cec2017", "composite", "composite_contiguity", "uav"]
    suites = [ctx.load(n) for n in names]
    csv, tex = [], []
    for a in ALGORITHMS:
        row = [display(a)]
        for S in suites:
            fes = np.concatenate([S.fes(f, a) for f in S.fids])
            b = S.budget
            csv.append({"suite": S.name, "algorithm": a, "mean_FEs": fes.mean(), "min_FEs": int(fes.min()),
                        "max_FEs": int(fes.max()), "budget": b, "mean_over_budget": fes.mean() / b})
            row.append(f"{fes.mean():.0f} ({fes.mean() / b:.2f}$\\times$)")
        tex.append(row)
    ctx.w.write("A5_function_evaluations", df(csv),
                ["Algorithm", "CEC2022", "CEC2017", "Laminate", "Laminate (contiguity)", "UAV"], tex,
                "Mean number of function evaluations per run (ratio to the budget in parentheses).",
                suites=suites,
                notes=["All algorithms have the same budget; a run on a CEC function uses fewer evaluations "
                       "when it stops at an error below $10^{-8}$. Minimum and maximum values are in the CSV file."])


# ---------------------------------------------------------------------------
# B / C. Benchmark results
# ---------------------------------------------------------------------------

def comparison(ctx, suite, name, budget=None):
    S = ctx.load(suite)
    metrics = [("Best", np.min), ("Median", np.median), ("Worst", np.max), ("Mean", np.mean),
               ("Std", lambda v: np.std(v, ddof=1))]
    if budget is not None:
        metrics = metrics[3:]
    csv, tex = [], []
    for fid in S.fids:
        vals = {a: values(S, fid, a, budget) for a in ALGORITHMS}
        means = np.array([vals[a].mean() for a in ALGORITHMS])
        best = np.isclose(means, means.min(), rtol=1e-12, atol=0.0)
        for j, (label, fn) in enumerate(metrics):
            cells = [fn(vals[a]) for a in ALGORITHMS]
            csv.append({"function": fid, "metric": label, **dict(zip(ALGORITHMS, cells))})
            txt = [sci(c) for c in cells]
            if label == "Mean":
                txt = [bold(t) if b else t for t, b in zip(txt, best)]
            tex.append([fid if j == 0 else "", label] + txt)
        tex.append(MID)
    from scipy import stats
    avg, _ = mean_ranks(means_matrix(S, budget))
    final_rank = stats.rankdata(avg, method="min").astype(int)
    csv.append({"function": f"({len(S.fids)} functions)", "metric": "Friedman mean rank", **dict(zip(ALGORITHMS, avg))})
    csv.append({"function": "", "metric": "Rank", **dict(zip(ALGORITHMS, final_rank))})
    tex.append(["", "Friedman mean rank"] + [bold(f"{r:.2f}") if r == avg.min() else f"{r:.2f}" for r in avg])
    tex.append(["", "Rank"] + [str(r) for r in final_rank])
    when = "final results" if budget is None else f"results after {budget:,} function evaluations".replace(",", "\\,")
    ctx.w.write(name, df(csv), ["Function", "Metric"] + shown(ALGORITHMS), tex,
                f"Comparison on {TITLE[suite]}: {when} of {S.n_runs} runs.", align="ll" + "r" * len(ALGORITHMS),
                suites=[S], notes=[FLOOR_NOTE, "Bold: lowest mean. Ranks are computed from the mean error of each "
                                               "function (1 = best) and averaged over the functions."],
                meta={"budget_FEs": budget, "std": "sample standard deviation (ddof = 1)"})


def statistics(ctx, suite, name):
    S = ctx.load(suite)
    st = suite_stats(S)
    csv, tex = [], []
    for fid in S.fids:
        for a in COMPETITORS:
            csv.append({"function": fid, "competitor": a, **st[fid][a]})
        tex.append([fid] + [f"{pval(st[fid][a]['p'])} {SYM_TEX[st[fid][a]['verdict']]}" for a in COMPETITORS])
    tex.append(MID)
    raw = {a: count_wtl([st[f][a]["verdict"] for f in S.fids]) for a in COMPETITORS}
    hol = {a: count_wtl([st[f][a]["verdict_holm"] for f in S.fids]) for a in COMPETITORS}
    tex.append(["$+/\\approx/-$"] + [wtl_text(raw[a]) for a in COMPETITORS])
    tex.append(["$+/\\approx/-$ (Holm)"] + [wtl_text(hol[a]) for a in COMPETITORS])
    sizes = holm_sizes(st, S.fids, COMPETITORS)
    ctx.w.write(name, df(csv), ["Function"] + shown(COMPETITORS), tex,
                f"{PROPOSED} against each competitor on {TITLE[suite]}: $p$-values of the Wilcoxon rank-sum test and decisions.",
                suites=[S], notes=[PAIRED, SYMBOLS, "The last row gives the counts when the decision is taken on "
                                   f"the Holm-adjusted $p$-value ({holm_family_text(sizes, len(COMPETITORS))}); "
                                   "the adjusted values are in the CSV file."],
                meta={"wtl_raw": raw, "wtl_holm": hol, "holm_family_size": sizes})


def effect_sizes(ctx, suite, name):
    S = ctx.load(suite)
    st = suite_stats(S)
    tex = [[fid] + [f"{fixed(st[fid][a]['cohens_d'])} / {fixed(st[fid][a]['cliffs_delta'])}" for a in COMPETITORS]
           for fid in S.fids]
    csv = [{"function": fid, "competitor": a, "cohens_d": st[fid][a]["cohens_d"],
            "cliffs_delta": st[fid][a]["cliffs_delta"]} for fid in S.fids for a in COMPETITORS]
    ctx.w.write(name, df(csv), ["Function"] + shown(COMPETITORS), tex,
                f"Effect sizes of {PROPOSED} against each competitor on {TITLE[suite]}: Cohen's $d$ / Cliff's $\\delta$.",
                suites=[S], notes=[f"Negative values: the error of {PROPOSED} is lower. ``---'': not defined "
                                   "(zero pooled variance)."])


def runtime(ctx, suite, name):
    S = ctx.load(suite)
    T = np.array([[np.mean([r["time"] for r in S.runs[fid][a]]) for a in ALGORITHMS] for fid in S.fids])
    fes = [np.mean([S.fes(fid, a).mean() for fid in S.fids]) for a in ALGORITHMS]
    csv = [{"function": fid, **dict(zip(ALGORITHMS, T[i]))} for i, fid in enumerate(S.fids)]
    csv.append({"function": "mean time (s)", **dict(zip(ALGORITHMS, T.mean(axis=0)))})
    csv.append({"function": "mean FEs", **dict(zip(ALGORITHMS, fes))})
    tex = [[fid] + [fixed(t) for t in T[i]] for i, fid in enumerate(S.fids)]
    tex += [MID, ["Mean time (s)"] + [fixed(t) for t in T.mean(axis=0)], ["Mean FEs"] + [f"{f:.0f}" for f in fes]]
    ses = S.config["sessions"][-1]
    ctx.w.write(name, df(csv), ["Function"] + shown(ALGORITHMS), tex,
                f"Mean wall-clock time per run (s) on {TITLE[suite]}.", suites=[S],
                notes=[f"Measured with {ses.get('workers')} runs in parallel on a {ses.get('cpu_count')}-core "
                       f"machine; times include the function evaluations and are not comparable across machines."])


def category_summary(ctx):
    S = ctx.load("cec2017")
    _, R = mean_ranks(means_matrix(S))
    cats = ["Unimodal", "Multimodal", "Hybrid", "Composition"]
    csv, tex = [], []
    for c in cats + ["All"]:
        idx = [i for i, f in enumerate(S.fids) if c == "All" or cec2017_category(f) == c]
        r = R[idx].mean(axis=0)
        csv.append({"category": c, "functions": len(idx), **dict(zip(ALGORITHMS, r))})
        if c == "All":
            tex.append(MID)
        tex.append([c, str(len(idx))] + [bold(f"{v:.2f}") if v == r.min() else f"{v:.2f}" for v in r])
    ctx.w.write("C6_cec2017_categories", df(csv), ["Category", "Functions"] + shown(ALGORITHMS), tex,
                "Mean rank by function category on CEC2017 ($D = 100$).", align="lr" + "r" * len(ALGORITHMS),
                suites=[S], notes=["Ranks from the mean final error of each function (1 = best). Bold: best "
                                   "mean rank of the row."])


# ---------------------------------------------------------------------------
# D. Overall statistics
# ---------------------------------------------------------------------------

def friedman_summary(ctx):
    csv, tex, suites = [], [], []
    for suite in ("cec2022", "cec2017"):
        S = ctx.load(suite)
        suites.append(S)
        avg, chi2, p, cd = friedman_cd(means_matrix(S))
        csv.append({"suite": suite, "functions": len(S.fids), "friedman_chi2": chi2, "friedman_p": p,
                    "nemenyi_cd": cd, **dict(zip(ALGORITHMS, avg))})
        cells = []
        for a, r in zip(ALGORITHMS, avg):
            s = f"{r:.2f}"
            if a != PROPOSED and abs(r - avg[0]) >= cd:
                s += "$^\\dagger$"
            cells.append(bold(s) if r == avg.min() else s)
        tex.append([TITLE[suite], str(len(S.fids)), f"{chi2:.2f}", pval(p), f"{cd:.2f}"] + cells)
    ctx.w.write("D1_friedman", df(csv), ["Suite", "$n$", "$\\chi^2$", "$p$", "CD"] + shown(ALGORITHMS),
                tex, "Friedman test, Nemenyi critical difference (CD, $\\alpha = 0.05$) and mean ranks.",
                align="lrrrr" + "r" * len(ALGORITHMS), suites=suites,
                notes=[f"$^\\dagger$: mean rank differs from that of {PROPOSED} by at least CD. Bold: best mean rank."])


def wtl_summary(ctx):
    cols, suites = [], []
    for suite in ("cec2022", "cec2017"):
        S = ctx.load(suite)
        suites.append(S)
        st = suite_stats(S)
        cols.append((suite, {a: count_wtl([st[f][a]["verdict"] for f in S.fids]) for a in COMPETITORS},
                     {a: count_wtl([st[f][a]["verdict_holm"] for f in S.fids]) for a in COMPETITORS}))
    csv = [{"competitor": a, **{c: wtl_text(d[a]) for c, d, _ in cols},
            **{c + "_holm": wtl_text(h[a]) for c, _, h in cols}} for a in COMPETITORS]
    tex = [[display(a)] + [x for _, d, h in cols for x in (wtl_text(d[a]), wtl_text(h[a]))] for a in COMPETITORS]
    ctx.w.write("D2_win_tie_loss", df(csv),
                ["Competitor", "CEC2022", "CEC2022 (Holm)", "CEC2017", "CEC2017 (Holm)"], tex,
                f"Functions on which {PROPOSED} is significantly better / not different / significantly worse "
                "($+/\\approx/-$).", suites=suites,
                notes=[PAIRED, "Holm: decision on the Holm-adjusted $p$-value."])


def signed_rank(ctx):
    """Wilcoxon signed-rank test over the functions (pairs: the mean errors of two algorithms on
    the same function), the multiple-problem analysis of Derrac et al. (2011)."""
    from scipy import stats
    csv, tex, suites = [], [], []
    for suite in ("cec2022", "cec2017"):
        S = ctx.load(suite)
        suites.append(S)
        M = means_matrix(S)
        rows = []
        for j, a in enumerate(ALGORITHMS):
            if a == PROPOSED:
                continue
            d = M[:, 0] - M[:, j]                       # < 0: the proposed algorithm has the lower mean error
            nz = d[d != 0]
            ranks = stats.rankdata(np.abs(nz))
            p = float(stats.wilcoxon(M[:, 0], M[:, j]).pvalue) if nz.size else np.nan
            rows.append((a, float(ranks[nz < 0].sum()), float(ranks[nz > 0].sum()), int((d == 0).sum()), p))
        adj = holm([r[4] for r in rows])
        for i, ((a, r_plus, r_minus, ties, p), q) in enumerate(zip(rows, adj)):
            csv.append({"suite": suite, "competitor": a, "functions": len(S.fids), "R_plus": r_plus,
                        "R_minus": r_minus, "ties": ties, "p": p, "p_holm": q})
            tex.append([TITLE[suite] if i == 0 else "", display(a), f"{r_plus:.1f}", f"{r_minus:.1f}", str(ties),
                        pval(p), pval(q)])
        tex.append(MID)
    ctx.w.write("D3_signed_rank", df(csv), ["Suite", "Competitor", "$R^+$", "$R^-$", "Ties", "$p$", "$p$ (Holm)"],
                tex[:-1], f"Wilcoxon signed-rank test of {PROPOSED} against each competitor over the functions.",
                align="llrrrrr", suites=suites,
                notes=[f"Pairs: the mean errors of the two algorithms on each function. $R^+$: sum of the ranks "
                       f"of the functions on which {PROPOSED} has the lower mean error; $R^-$: the higher. "
                       f"Holm: adjusted for the {len(ALGORITHMS) - 1} comparisons of a suite."])


# ---------------------------------------------------------------------------
# E. Engineering problems
# ---------------------------------------------------------------------------

STACK_TEX = {0: "$0_2$", 1: "$\\pm45$", 2: "$90_2$"}


def seq_tex(genes):
    return "[" + "/".join(STACK_TEX[int(g)] for g in genes) + "]$_s$"


def best_run(S, algo):
    runs = S.runs[S.name][algo]
    f = np.array([r["f"] for r in runs])
    return runs[int(np.argmin(f))] if np.isfinite(f).any() else None


def table_E1(ctx):
    import laminate_buckling as lam
    import path_planning as pp
    rows = [("Laminate", "Material", "graphite-epoxy"),
            ("Laminate", "$E_1$, $E_2$, $G_{12}$ (Msi)", f"{lam.E1 / 1e6:g}, {lam.E2 / 1e6:g}, {lam.G12 / 1e6:g}"),
            ("Laminate", "$\\nu_{12}$", f"{lam.NU12:g}"),
            ("Laminate", "Plies, ply thickness (in)", f"{lam.N_PLIES}, {lam.PLY_T:g}"),
            ("Laminate", "Plate $a \\times b$ (in)", f"{lam.A_LEN:g} $\\times$ {lam.B_LEN:g}"),
            ("Laminate", "Loads $N_x$, $N_y$ (lb/in)", f"{lam.NX:g}, {lam.NY:g}"),
            ("Laminate", "Design variables", f"{lam.DIM} two-ply stacks, each $0_2$, $\\pm45$ or $90_2$"),
            ("Laminate", "Buckling modes $p, q$", f"{lam.MODES[0]}--{lam.MODES[-1]}"),
            ("Laminate", "Contiguity limit, penalty", f"{2 * lam.MAX_CONTIGUOUS_STACKS} plies, {lam.PENALTY:g} per violating stack"),
            ("UAV", "Terrain map (cells)", f"{pp.MAPSIZE_X} $\\times$ {pp.MAPSIZE_Y}"),
            ("UAV", "Start; goal", f"({pp.START[0]:g}, {pp.START[1]:g}, {pp.START[2]:g}); "
                                    f"({pp.END[0]:g}, {pp.END[1]:g}, {pp.END[2]:g})"),
            ("UAV", "Path nodes, design variables", f"{pp.N_NODES}, {pp.DIM}"),
            ("UAV", "Altitude above ground", f"{pp.ZMIN:g}--{pp.ZMAX:g}"),
            ("UAV", "UAV size, danger distance", f"{pp.DRONE_SIZE:g}, {pp.DANGER_DIST:g}"),
            ("UAV", "Maximum turning, climb angle", f"{pp.TURNING_MAX:g}$^\\circ$, {pp.CLIMB_MAX:g}$^\\circ$"),
            ("UAV", "Cost weights $b_1, \\dots, b_4$", f"{pp.B1:g}, {pp.B2:g}, {pp.B3:g}, {pp.B4:g}")]
    rows += [("UAV", f"Threat {i + 1}: centre, radius", f"({t[0]:g}, {t[1]:g}), {t[3]:g}")
             for i, t in enumerate(pp.THREATS)]
    tex, last = [], None
    for p, k, v in rows:
        if last is not None and p != last:
            tex.append(MID)
        tex.append([p if p != last else "", k, v])
        last = p
    ctx.w.write("E1_problem_data", df([{"problem": p, "parameter": k, "value": v} for p, k, v in rows]),
                ["Problem", "Parameter", "Value"], tex, "Data of the engineering design problems.", align="lll",
                notes=["Laminate: Le Riche (1994, Ch. 3); Ahmid et al. (2019). UAV: Phung and Ha (2021); "
                       "lengths are in the map units of the original model."],
                extra_sources=[os.path.join(fl.ROOT, "applications", "composite", "laminate_buckling.py"),
                               os.path.join(fl.ROOT, "applications", "uav", "path_planning.py")])


def table_E2(ctx):
    csv, tex, suites, refs = [], [], [], {}
    for suite, label in (("composite", "Unconstrained"), ("composite_contiguity", "With contiguity constraint")):
        S = ctx.load(suite)
        suites.append(S)
        ref = rx.load_objective(suite, suite, None)["reference"]
        refs[suite] = ref
        st = compare(S.final(suite, PROPOSED), {a: S.final(suite, a) for a in COMPETITORS})
        tex.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{label}}} (exact optimum {ref:.4f})}} \\\\")
        lams = {a: -S.final(suite, a) for a in ALGORITHMS}
        top = max(v.mean() for v in lams.values())
        for a in ALGORITHMS:
            v = lams[a]
            hits = int(np.sum(np.abs(v - ref) < 1e-6))
            gap = float(np.mean((ref - v) / ref) * 100.0)
            s = st.get(a)
            csv.append({"variant": suite, "algorithm": a, "best": v.max(), "mean": v.mean(), "std": v.std(ddof=1),
                        "worst": v.min(), "runs_at_optimum": hits, "runs": len(v), "mean_gap_%": gap,
                        "mean_FEs": S.fes(suite, a).mean(),
                        "mannwhitney_p_vs_proposed": s["p"] if s else np.nan, "verdict": s["verdict"] if s else ""})
            mean = f"{v.mean():.2f}"
            tex.append([display(a), f"{v.max():.2f}", bold(mean) if v.mean() == top else mean, f"{v.std(ddof=1):.2f}",
                        f"{v.min():.2f}", f"{hits}/{len(v)}", f"{gap:.3f}", f"{S.fes(suite, a).mean():.0f}",
                        "---" if s is None else f"{pval(s['p'])} {SYM_TEX[s['verdict']]}"])
        tex.append(MID)
    head =["Algorithm", "Best", "Mean", "Std", "Worst", "Optimum hit", "Mean gap (\\%)", "Mean FEs", f"$p$ vs {PROPOSED}"]
    ctx.w.write("E2_composite_results", df(csv), head, tex[:-1],
                "Laminate stacking-sequence design: critical buckling load factor $\\lambda_{cb}$ over the runs.",
                suites=suites, meta={"reference_optimum": refs},
                notes=["Higher is better. Exact optima from an exhaustive search of all $3^{16}$ designs. With the "
                       "contiguity constraint the values include the static penalty of infeasible designs.",
                       PAIRED, SYMBOLS])


def table_E3(ctx):
    import laminate_buckling as lam
    with open(os.path.join(fl.ROOT, "applications", "composite", "reference_optimum.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    csv, tex, suites = [], [], []
    for suite, key, label in (("composite", "free", "Unconstrained"),
                              ("composite_contiguity", "contiguity", "Contiguity")):
        S = ctx.load(suite)
        suites.append(S)
        r = best_run(S, PROPOSED)
        g = lam.decode(r["x"])
        entries = [(f"{PROPOSED} (best run)", g), ("Exact optimum", np.array(ref[key]["genes"]))]
        for src, genes in entries:
            lcb, viol = float(lam.buckling_load_factor(genes)), int(lam.contiguity_violation(genes))
            csv.append({"variant": suite, "source": src, "sequence": lam.describe(genes), "lambda_cb": lcb,
                        "contiguity_violations": viol})
            tex.append([label, src, seq_tex(genes), f"{lcb:.4f}", str(viol)])
        tex.append(MID)
    ctx.w.write("E3_composite_layups", df(csv), ["Variant", "Design", "Stacking sequence", "$\\lambda_{cb}$", "Violations"],
                tex[:-1], f"Best stacking sequences found by {PROPOSED} and the exact optima.", align="lllrr", suites=suites,
                notes=[f"The unconstrained problem has {ref['free']['n_optimal_designs']} optimal designs with the "
                       f"same $\\lambda_{{cb}}$. Violations: stacks exceeding the contiguity limit."])


def table_E4(ctx):
    S = ctx.load("uav")
    st = compare(S.final("uav", PROPOSED), {a: S.final("uav", a) for a in COMPETITORS})
    csv, tex = [], []
    means = {a: (lambda v: v[np.isfinite(v)].mean() if np.isfinite(v).any() else np.nan)(S.final("uav", a))
             for a in ALGORITHMS}
    top = np.nanmin(list(means.values()))
    for a in ALGORITHMS:
        v = S.final("uav", a)
        fin = v[np.isfinite(v)]
        inv = np.mean([r["invalid"] for r in S.runs["uav"][a]])
        g = lambda fn: float(fn(fin)) if fin.size else np.nan
        s = st.get(a)
        csv.append({"algorithm": a, "best": g(np.min), "mean": g(np.mean),
                    "std": float(fin.std(ddof=1)) if fin.size > 1 else np.nan, "worst": g(np.max),
                    "feasible_runs": int(fin.size), "runs": len(v), "mean_FEs": S.fes("uav", a).mean(),
                    "mean_invalid_candidates": inv, "mannwhitney_p_vs_proposed": s["p"] if s else np.nan,
                    "verdict": s["verdict"] if s else ""})
        mean = fixed(g(np.mean), 1)
        tex.append([display(a), fixed(g(np.min), 1), bold(mean) if g(np.mean) == top else mean,
                    fixed(csv[-1]["std"], 1), fixed(g(np.max), 1), f"{fin.size}/{len(v)}",
                    f"{S.fes('uav', a).mean():.0f}", f"{inv:.0f}",
                    "---" if s is None else f"{pval(s['p'])} {SYM_TEX[s['verdict']]}"])
    ctx.w.write("E4_uav_results", df(csv),
                ["Algorithm", "Best", "Mean", "Std", "Worst", "Feasible", "Mean FEs", "Invalid", f"$p$ vs {PROPOSED}"], tex,
                "UAV path planning: cost $F$ over the runs.", suites=[S],
                notes=["Best, mean, standard deviation and worst are computed over the feasible runs only; "
                       "``Feasible'' gives the number of runs that found a collision-free path. In the test, "
                       "a run without a feasible path counts as the worst value. ``Invalid'': mean number of "
                       "non-finite candidates per run, evaluated as infeasible for every algorithm.",
                       PAIRED, SYMBOLS])


def table_E5(ctx):
    import path_planning as pp
    S = ctx.load("uav")
    csv, tex = [], []
    for a in ALGORITHMS:
        r = best_run(S, a)
        ok = r is not None and np.all(np.isfinite(r["x"])) and \
            abs(pp.cost(r["x"]) - r["f"]) <= 1e-9 * max(1.0, abs(r["f"]))
        if not ok:
            csv.append({"algorithm": a, "run": None if r is None else r["run"], "note":
                        "no feasible run" if r is None else "stored path does not reproduce its stored cost"})
            tex.append([display(a)] + ["---"] * 5)
            continue
        J = pp.cost_components(r["x"])
        csv.append({"algorithm": a, "run": r["run"], "J1_length": J[0], "J2_threat": J[1], "J3_altitude": J[2],
                    "J4_smooth": J[3], "cost": r["f"]})
        tex.append([display(a)] + [fixed(j, 2) for j in J] + [fixed(r["f"], 1)])
    ctx.w.write("E5_uav_cost_components", df(csv),
                ["Algorithm", "$J_1$ (length)", "$J_2$ (threat)", "$J_3$ (altitude)", "$J_4$ (smoothness)", "$F$"],
                tex, "Cost components of the best path found by each algorithm.", suites=[S],
                notes=[f"$F = {pp.B1:g}J_1 + {pp.B2:g}J_2 + {pp.B3:g}J_3 + {pp.B4:g}J_4$, recomputed from the "
                       f"stored best solution."])


def table_E6(ctx):
    import laminate_buckling as lam
    S = ctx.load("composite")
    with open(os.path.join(fl.ROOT, "applications", "composite", "reference_optimum.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["free"]
    entries = [("Ahmid et al. (2019), Table 3", np.array(g), float(v)) for g, v in lam.PUBLISHED]
    entries.append(("Exhaustive search (this study)", np.array(ref["genes"]), None))
    entries.append((f"{PROPOSED}, best run (this study)", lam.decode(best_run(S, PROPOSED)["x"]), None))
    csv, tex = [], []
    for src, genes, pub in entries:
        calc = float(lam.buckling_load_factor(genes))
        csv.append({"source": src, "sequence": lam.describe(genes), "lambda_cb_published": pub,
                    "lambda_cb_recomputed": calc, "difference": None if pub is None else calc - pub})
        tex.append([src, seq_tex(genes), "---" if pub is None else f"{pub:.2f}", f"{calc:.4f}",
                    "---" if pub is None else f"{calc - pub:+.4f}"])
    ctx.w.write("E6_composite_literature", df(csv),
                ["Source", "Stacking sequence", "Published $\\lambda_{cb}$", "Recomputed $\\lambda_{cb}$", "Difference"],
                tex, "Laminate design without the contiguity constraint: published designs and this study.",
                align="llrrr", suites=[S],
                notes=["Recomputed: the model of this study applied to the listed sequence. Published values are "
                       "given to two decimals in the source. No published value was found for the contiguity "
                       "constraint as defined here."],
                extra_sources=[os.path.join(fl.ROOT, "applications", "composite", "laminate_buckling.py")])


# ---------------------------------------------------------------------------
# F. Ablation
# ---------------------------------------------------------------------------

def load_variants(ctx, suite, group, directory):
    """(S, A, names, vals) for one suite, or None if the runs of this group do not exist yet.
    vals[fid][name]: errors of the full algorithm (from the main results) and of each variant."""
    S = ctx.load(suite)
    sdir = os.path.join(directory, fl.SUITE_DIRS[suite])
    if not os.path.isdir(os.path.join(sdir, "runs")):
        return None
    A = fl.load(suite, directory, algorithms=list(group))
    same = ("dim", "budget", "max_fes", "iterations", "agents", "seed_base", "target_stop", "cec_source")
    diff = [k for k in same if A.settings.get(k) != S.settings.get(k)]
    if diff or A.n_runs != S.n_runs or A.fids != S.fids:
        sys.exit(f"ERROR: {sdir}: settings differ from the main results ({diff or 'runs / functions'}); "
                 f"rerun the variants with the settings of {S.dir}.")
    names = [PROPOSED] + list(group)
    vals = {fid: {v: (S if v == PROPOSED else A).final(fid, v) for v in names} for fid in S.fids}
    return S, A, names, vals


def variant_study(ctx, group, directory, prefix, what, caption, column):
    """Summary table <prefix>1 and per-function tables <prefix>2 of a group of RTI-DE variants."""
    data = {s: load_variants(ctx, s, group, directory) for s in ("cec2022", "cec2017")}
    data = {s: d for s, d in data.items() if d is not None}
    if not data:
        S = ctx.load("cec2022")
        budget = (f"--budget fe --max-fes {S.settings['max_fes']}" if S.settings["budget"] == "fe"
                  else f"--agents {S.settings['agents']} --iterations {S.settings['iterations']}")
        print(f"    no {what} runs in {directory} (tables {prefix}1, {prefix}2 skipped); run\n"
              f"      python -W ignore run_experiments.py --suites cec2022 cec2017 {budget} --runs {S.n_runs} "
              f"--algorithms {' '.join(group)} --out {os.path.relpath(directory, fl.ROOT)}")
        return
    names = [PROPOSED] + list(group)
    text = {PROPOSED: "---", **{v: d for v, (_, d) in group.items()}}
    summary = {v: [] for v in names}
    csv1, suites = {v: {"variant": v, column.lower(): text[v].replace("\\", "")} for v in names}, []
    sizes = {}
    for suite, (S, A, _, vals) in data.items():
        suites += [S, A]
        avg, _ = mean_ranks(np.array([[vals[f][v].mean() for v in names] for f in S.fids]))
        st = {f: compare(vals[f][PROPOSED], {v: vals[f][v] for v in names[1:]}) for f in S.fids}
        sizes[suite] = holm_sizes(st, S.fids, names[1:])
        for i, v in enumerate(names):
            w = "---" if v == PROPOSED else wtl_text(count_wtl([st[f][v]["verdict"] for f in S.fids]))
            wh = "---" if v == PROPOSED else wtl_text(count_wtl([st[f][v]["verdict_holm"] for f in S.fids]))
            summary[v] += [bold(f"{avg[i]:.2f}") if avg[i] == avg.min() else f"{avg[i]:.2f}", w, wh]
            csv1[v].update({f"{suite}_mean_rank": avg[i], f"{suite}_wtl": w.replace("---", ""),
                            f"{suite}_wtl_holm": wh.replace("---", "")})
        csv2, tex2 = [], []
        for f in S.fids:
            for label, fn in (("Mean", np.mean), ("Std", lambda x: np.std(x, ddof=1))):
                csv2.append({"function": f, "metric": label, **{v: fn(vals[f][v]) for v in names}})
                tex2.append([f if label == "Mean" else "", label] + [sci(fn(vals[f][v])) for v in names])
            csv2.append({"function": f, "metric": "verdict", **{v: st[f][v]["verdict"] for v in names[1:]}})
            tex2.append(["", "Decision", "---"] + [SYM_TEX[st[f][v]["verdict"]] for v in names[1:]])
            tex2.append(MID)
        ctx.w.write(f"{prefix}2_{what}_{suite}", df(csv2), ["Function", "Metric"] + names, tex2[:-1],
                    f"{what.capitalize()} on {TITLE[suite]}: error of each variant and decision against {PROPOSED}.",
                    align="ll" + "r" * len(names), suites=[S, A],
                    notes=[FLOOR_NOTE, f"$+$: {PROPOSED} is significantly better than the variant.", PAIRED])
    head = ["Variant", column]
    for suite in data:
        t_ = "CEC2022" if suite == "cec2022" else "CEC2017"
        head += [f"{t_} rank", f"{t_} $+/\\approx/-$", f"{t_} (Holm)"]
    tex1 = [[v, text[v]] + summary[v] for v in names]
    m = len(names) - 1
    smaller = [f"{'CEC2022' if s == 'cec2022' else 'CEC2017'}: {t}"
               for s, t in ((s, holm_smaller(sizes[s], m)) for s in data) if t]
    family = f"{m} comparisons per function" + (f", {HOLM_UNDEFINED}: {'; '.join(smaller)}" if smaller else "")
    ctx.w.write(f"{prefix}1_{what}_summary", df(list(csv1.values())), head, tex1, caption,
                align="ll" + "rrr" * len(data), suites=suites,
                notes=[f"Rank: mean rank among the {len(names)} variants (1 = best). $+/\\approx/-$: functions on "
                       f"which {PROPOSED} is significantly better / not different / significantly worse than "
                       f"the variant; Holm: decision on the Holm-adjusted $p$-value ({family}).", PAIRED],
                meta={"holm_family_size": sizes})


def ablation(ctx):
    variant_study(ctx, rx.ABLATION, ctx.args.ablation, "F", "ablation",
                  f"Ablation study: each variant removes or replaces one component of {PROPOSED}.", "Change")


def sensitivity(ctx):
    variant_study(ctx, rx.SENSITIVITY, ctx.args.sensitivity, "G", "sensitivity",
                  f"Parameter sensitivity: each variant changes one parameter of {PROPOSED}.", "Parameter")


# ---------------------------------------------------------------------------
# Registry and command line
# ---------------------------------------------------------------------------

TABLES = {
    "A1": (table_A1, "CEC2022 functions"),
    "A2": (table_A2, "CEC2017 functions"),
    "A3": (table_A3, "experimental settings"),
    "A4": (table_A4, "algorithm parameters"),
    "A5": (table_A5, "function evaluations per algorithm"),
    "B1": (lambda c: comparison(c, "cec2022", "B1_cec2022_comparison"), "CEC2022 Best/Median/Worst/Mean/Std + ranks"),
    "B2": (lambda c: statistics(c, "cec2022", "B2_cec2022_ranksum"), "CEC2022 Wilcoxon rank-sum (Mann-Whitney) p and decisions"),
    "B3": (lambda c: effect_sizes(c, "cec2022", "B3_cec2022_effect_sizes"), "CEC2022 Cohen's d / Cliff's delta"),
    "B4": (lambda c: runtime(c, "cec2022", "B4_cec2022_runtime"), "CEC2022 runtime and FEs"),
    "C1": (lambda c: comparison(c, "cec2017", "C1_cec2017_comparison"), "CEC2017 Best/Median/Worst/Mean/Std + ranks"),
    "C2": (lambda c: statistics(c, "cec2017", "C2_cec2017_ranksum"), "CEC2017 Wilcoxon rank-sum (Mann-Whitney) p and decisions"),
    "C3": (lambda c: effect_sizes(c, "cec2017", "C3_cec2017_effect_sizes"), "CEC2017 Cohen's d / Cliff's delta"),
    "C4": (lambda c: runtime(c, "cec2017", "C4_cec2017_runtime"), "CEC2017 runtime and FEs"),
    "C6": (category_summary, "CEC2017 mean rank by category"),
    "D1": (friedman_summary, "Friedman test and Nemenyi CD"),
    "D2": (wtl_summary, "win/tie/loss summary"),
    "D3": (signed_rank, "Wilcoxon signed-rank test over the functions"),
    "E1": (table_E1, "engineering problem data"),
    "E2": (table_E2, "composite results"),
    "E3": (table_E3, "composite best layups"),
    "E4": (table_E4, "UAV results"),
    "E5": (table_E5, "UAV cost components"),
    "E6": (table_E6, "composite: comparison with the literature"),
    "F": (ablation, "ablation summary (F1) and per-function results (F2); needs the ablation runs"),
    "G": (sensitivity, "parameter sensitivity summary (G1) and per-function results (G2); needs the sensitivity runs"),
}


class Context:
    def __init__(self, args):
        self.args = args
        self.w = Writer(args.out, args.dry_run)

    def load(self, suite):
        return fl.load(suite, self.args.results)


def main():
    ap = argparse.ArgumentParser(description="Tables for the RTI-DE paper (CSV + LaTeX + provenance).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ID", help="tables to build (default: all); see --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results", default=os.path.join(fl.ROOT, "results_cache", "paper"))
    ap.add_argument("--ablation", default=os.path.join(fl.ROOT, "results_cache", "paper_ablation"))
    ap.add_argument("--sensitivity", default=os.path.join(fl.ROOT, "results_cache", "paper_sensitivity"))
    ap.add_argument("--out", default=os.path.join(fl.ROOT, "outputs", "tables"))
    ap.add_argument("--dry-run", action="store_true", help="build every table in memory, write no file")
    args = ap.parse_args()
    if args.list:
        for k, (_, text) in TABLES.items():
            print(f"  {k:<4} {text}")
        return
    names = args.only or list(TABLES)
    unknown = [n for n in names if n not in TABLES]
    if unknown:
        sys.exit(f"ERROR: unknown table id(s) {unknown}; use --list.")
    warnings.filterwarnings("ignore")
    ctx = Context(args)
    print(f"RTI-DE paper tables -> {args.out}{'  (dry run: nothing is written)' if args.dry_run else ''}")
    t0 = time.time()
    for n in names:
        print(f"  [{n}] {TABLES[n][1]}")
        TABLES[n][0](ctx)
    print(f"Done in {time.time() - t0:.0f} s: {len(ctx.w.written)} table(s) written.")


if __name__ == "__main__":
    main()
