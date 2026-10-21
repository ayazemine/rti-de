"""
Engineering design problems of the RTI-DE paper: problem registry and summary rows.

Problems
--------
composite             Laminate stacking sequence, max buckling load (Le Riche & Haftka 1993)
composite_contiguity  Same, with the ≤4-contiguous-plies constraint (static penalty)
uav                   3-D UAV path planning over terrain with threats (Phung & Ha 2021)

The runs are made by ``run_experiments.py`` (suites of the same names), which imports
``load_problem`` and ``summarize`` from here:

    python -W ignore run_experiments.py --suites composite composite_contiguity uav
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "composite"), os.path.join(HERE, "uav")):
    if p not in sys.path:
        sys.path.insert(0, p)

PROBLEMS = ("composite", "composite_contiguity", "uav")


def load_problem(name):
    """Return dict(objective(X)->(n,), scalar(x)->float, lb, ub, dim, maximize, reference, describe)."""
    if name.startswith("composite"):
        import laminate_buckling as lb
        contiguity = name == "composite_contiguity"
        ref_path = os.path.join(HERE, "composite", "reference_optimum.json")
        reference = None
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as f:
                ref = json.load(f)
            reference = ref["contiguity" if contiguity else "free"]["lambda_cb"]

        def objective(X):
            return lb.objective(X, contiguity=contiguity)

        def describe(x):
            g = lb.decode(x)
            return {"sequence": lb.describe(g), "genes": g.tolist(),
                    "lambda_cb": float(lb.buckling_load_factor(g)),
                    "contiguity_violations": int(lb.contiguity_violation(g))}

        return dict(objective=objective, scalar=lambda x: float(objective(np.atleast_2d(x))[0]),
                    lb=lb.LB, ub=lb.UB, dim=lb.DIM, maximize=True, reference=reference,
                    describe=describe)
    if name == "uav":
        import path_planning as pp

        def describe(x):
            J = pp.cost_components(x)
            return {"J1_length": J[0], "J2_threat": J[1], "J3_altitude": J[2], "J4_smooth": J[3]}

        return dict(objective=pp.objective, scalar=pp.cost, lb=pp.LB, ub=pp.UB, dim=pp.DIM,
                    maximize=False, reference=None, describe=describe)
    raise ValueError(f"Unknown problem {name!r}; choose from {PROBLEMS}.")


def summarize(results, prob):
    """One row per algorithm from {algorithm: [run dicts]} (keys f, x, fes, invalid, time)."""
    rows = []
    for algo, runs in results.items():
        f = np.array([r["f"] for r in runs])
        feas = np.isfinite(f)
        vals = -f if prob["maximize"] else f        # report λ_cb (maximized) as a positive value
        v = vals[feas]
        best_idx = int(np.nanargmax(np.where(feas, vals, -np.inf)) if prob["maximize"]
                       else np.nanargmin(np.where(feas, vals, np.inf)))
        row = {"algorithm": algo, "runs": len(runs), "feasible_rate": feas.mean(),
               "best": (v.max() if prob["maximize"] else v.min()) if v.size else np.nan,
               "mean": v.mean() if v.size else np.nan, "std": v.std() if v.size else np.nan,
               "median": np.median(v) if v.size else np.nan,
               "worst": (v.min() if prob["maximize"] else v.max()) if v.size else np.nan,
               "mean_FEs": np.mean([r["fes"] for r in runs]),
               "mean_invalid_candidates": np.mean([r["invalid"] for r in runs]),
               "mean_time_s": np.mean([r["time"] for r in runs])}
        if prob["reference"] is not None and v.size:
            row["hits_optimum"] = int(np.sum(np.isclose(v, prob["reference"], rtol=0, atol=1e-6)))
            row["mean_gap_%"] = 100 * (prob["reference"] - v.mean()) / prob["reference"]
        row["best_design"] = json.dumps(prob["describe"](runs[best_idx]["x"]), ensure_ascii=False)
        rows.append(row)
    return rows
