"""
Shared helpers for the paper figures (``scripts/make_figures.py``).

Conventions:

* vector output only: every figure is written as PDF and SVG;
* data only from files: results are read from ``results_cache/paper/<suite>/``
  (written by ``run_experiments.py``); nothing is typed in by hand;
* provenance: a ``<figure>.json`` next to every figure lists the result files,
  the run settings, the git commit and the hash of ``RTIDE/optimizer.py``;
* stale results are refused: the hash stored with the results must match the
  current ``RTIDE/optimizer.py``;
* failed runs are never dropped silently: a suite with ``*.error.json`` files
  or missing runs stops the script.
"""

import json
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_experiments as rx  # noqa: E402  (also puts benchmarks/ and applications/ on sys.path)

ALGORITHMS = list(rx.ALL_ALGORITHMS)
PROPOSED = rx.PROPOSED
OPTIMIZER = rx.OPTIMIZER_PATH
SUITE_DIRS = {"cec2022": "cec2022_D20", "cec2017": "cec2017_D100", "composite": "composite",
              "composite_contiguity": "composite_contiguity", "uav": "uav"}
SUITE_TITLES = {"cec2022": "CEC2022 (D = 20)", "cec2017": "CEC2017 (D = 100)"}

# CEC2017 functions shown in the main text: evenly spaced within each category,
# fixed before looking at the results.
CEC2017_MAIN = ["F1", "F3", "F4", "F7", "F10", "F11", "F14", "F17", "F20", "F21", "F25", "F30"]

ERROR_FLOOR = 1e-8          # CEC reports: errors below 1e-8 count as zero; drawn at this value on log axes

# One fixed colour, line style and marker per algorithm in every figure
# (blue, green, cyan and purple of Paul Tol's "bright" scheme, plus black and grey; the line styles and
# markers differ as well, so the curves do not depend on colour alone). The proposed algorithm is the
# thick dark-red line.
STYLE = {
    "RTI-DE": dict(color="#C00000", ls="-", marker="o", lw=1.8, zorder=10),
    "DE": dict(color="#4477AA", ls="--", marker="s", lw=0.9, zorder=5),
    "LSHADE": dict(color="#228833", ls=":", marker="^", lw=1.1, zorder=5),
    "RIME": dict(color="#000000", ls="-.", marker="X", lw=0.9, zorder=6),
    "PSO": dict(color="#66CCEE", ls="-", marker="v", lw=0.9, zorder=5),
    "GWO": dict(color="#AA3377", ls="--", marker="P", lw=0.9, zorder=5),
    "WOA": dict(color="#888888", ls="-.", marker="<", lw=0.9, zorder=5),
}
# Names as printed in figures and tables; the keys above and in the result files stay unchanged.
DISPLAY_NAME = {"LSHADE": "L-SHADE"}


def display(algo):
    """Printed name of an algorithm (its internal key if no other name is defined)."""
    return DISPLAY_NAME.get(algo, algo)


THREAT = "#6A3D9A"          # threat cylinders of the UAV scenario (not the colour of any algorithm)

SINGLE_COL = 3.5            # inches
DOUBLE_COL = 7.2


def apply_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.markersize": 3, "grid.linewidth": 0.3, "grid.color": "#D0D0D0",
        "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.unicode_minus": False,
    })


def line_style(algo, markers=False):
    s = dict(STYLE[algo])
    if not markers:
        s.pop("marker")
    return s


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Suite:
    """All finished runs of one suite: ``runs[fid][algo]`` is a list of run dicts."""

    def __init__(self, name, results_dir, algorithms=None):
        self.name = name
        self.algos = list(algorithms or ALGORITHMS)
        self.dir = os.path.join(results_dir, SUITE_DIRS[name])
        cfg_path = os.path.join(self.dir, "config.json")
        if not os.path.exists(cfg_path):
            sys.exit(f"ERROR: {cfg_path} not found. Run run_experiments.py first.")
        with open(cfg_path, encoding="utf-8") as fh:
            self.config = json.load(fh)
        self.settings = self.config["settings"]
        now = rx._sha256(OPTIMIZER)
        if self.settings["optimizer_sha256"] != now:
            sys.exit(f"ERROR: {self.dir} was produced with a different RTIDE/optimizer.py\n"
                     f"  stored {self.settings['optimizer_sha256']}\n  now    {now}\n"
                     f"  The results do not describe the current algorithm; rerun the experiments.")
        self.runs, errors = rx.collect(self.dir)
        self.fids = [f for f in rx.function_ids(name) if f in self.runs]
        self.n_runs = max(s["runs_requested"] for s in self.config["sessions"])
        problems = [f"{fid}/{a}: {len(errors[fid][a])} failed run(s)" for fid in errors for a in errors[fid]]
        missing = [f for f in rx.function_ids(name) if f not in self.runs]
        problems += [f"{fid}: no runs" for fid in missing]
        for fid in self.fids:
            for a in self.algos:
                k = len(self.runs[fid].get(a, []))
                if k != self.n_runs:
                    problems.append(f"{fid}/{a}: {k}/{self.n_runs} runs")
        if problems:
            sys.exit(f"ERROR: {self.dir} is incomplete; figures are not drawn from partial data:\n  "
                     + "\n  ".join(problems[:30]))
        self.is_cec = name in rx.CEC_SUITES
        self.n_files = sum(len(self.runs[f][a]) for f in self.fids for a in self.algos)
        self.budget = rx.budget_of(self.settings)

    # --- values -----------------------------------------------------------
    def f_star(self, fid):
        """F* of a CEC function, recovered from a stored run (error = f - F*)."""
        r = self.runs[fid][self.algos[0]][0]
        return r["f"] - r["error"]

    def final(self, fid, algo):
        """Final values, lower is better: CEC error (below 1e-8 -> 0) or raw objective f."""
        rr = self.runs[fid][algo]
        if self.is_cec:
            e = np.array([r["error"] for r in rr])
            return np.where(e < ERROR_FLOOR, 0.0, e)
        return np.array([r["f"] for r in rr])

    def fes(self, fid, algo):
        return np.array([r["fes"] for r in self.runs[fid][algo]])

    def at_fes(self, fid, algo, budget):
        """Best value found within ``budget`` FEs for each run (inf if nothing finite yet)."""
        off = self.f_star(fid) if self.is_cec else 0.0
        out = np.empty(len(self.runs[fid][algo]))
        for i, r in enumerate(self.runs[fid][algo]):
            out[i] = trace_on_grid(r["trace"], np.array([budget]))[0] - off
        if self.is_cec:
            out = np.where(out < ERROR_FLOOR, 0.0, out)
        return out

    def curves(self, fid, algo, grid):
        """(runs, len(grid)) best-so-far values on an FE grid; a run keeps its final value
        after it stops. CEC suites return the error f - F*."""
        off = self.f_star(fid) if self.is_cec else 0.0
        return np.array([trace_on_grid(r["trace"], grid) - off for r in self.runs[fid][algo]])


def trace_on_grid(trace, grid):
    """Step function of an FE trace [(fe, best), ...] on ``grid`` (inf before the first entry)."""
    if len(trace) == 0:
        return np.full(len(grid), np.inf)
    idx = np.searchsorted(trace[:, 0], grid, side="right") - 1
    return np.where(idx >= 0, trace[np.maximum(idx, 0), 1], np.inf)


def fe_grid(lo, hi, points=160):
    return np.unique(np.round(np.geomspace(max(lo, 1), hi, points)).astype(int))


_SUITES = {}


def load(name, results_dir, algorithms=None):
    key = (name, os.path.abspath(results_dir))
    if key not in _SUITES:
        t0 = time.time()
        print(f"  loading {name} ...", end="", flush=True)
        _SUITES[key] = Suite(name, results_dir, algorithms)
        s = _SUITES[key]
        print(f" {s.n_files} runs, {len(s.fids)} function(s), {time.time() - t0:.0f} s")
    return _SUITES[key]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


class Output:
    """Writes <name>.pdf, <name>.svg and <name>.json; with ``dry_run`` nothing is written."""

    def __init__(self, out_dir, dry_run=False):
        self.out_dir, self.dry_run, self.written = out_dir, dry_run, []

    def save(self, fig, name, suites=(), notes=None, extra_sources=()):
        fig.canvas.draw()                        # run the layout even in a dry run
        if self.dry_run:
            print(f"    [dry run] {name}: built, not written")
            plt.close(fig)
            return
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, name)
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".svg")
        plt.close(fig)
        meta = {
            "figure": name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "optimizer_sha256": rx._sha256(OPTIMIZER),
            "sources": [{"suite": s.name, "directory": os.path.relpath(s.dir, ROOT),
                         "run_files": s.n_files, "settings": s.settings,
                         "runs_per_algorithm": s.n_runs} for s in suites]
                       + [{"file": os.path.relpath(p, ROOT)} for p in extra_sources],
            "notes": notes or {},
        }
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=_jsonable)
        self.written.append(base)
        print(f"    wrote {os.path.relpath(base, ROOT)}.pdf/.svg/.json")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def legend_below(fig, handles, labels, ncol=9, y=0.0):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, handlelength=2.6, columnspacing=1.2)
