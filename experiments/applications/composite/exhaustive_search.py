"""
Exhaustive enumeration of all 3^16 = 43,046,721 stacking sequences of the
64-ply benchmark, to obtain the exact global optimum of λ_cb with and without
the contiguity constraint. The result is the reference value used to judge
optimizer results (gap to the true optimum).

Run from experiments/:
    python applications/composite/exhaustive_search.py
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laminate_buckling as lb  # noqa: E402

CHUNK_BITS = 10           # enumerate the last 10 genes in a block, loop over the first 6


def main():
    t0 = time.time()
    tail = np.array(list(itertools.product(range(3), repeat=CHUNK_BITS)), dtype=np.int8)
    n_head = lb.N_STACKS - CHUNK_BITS
    best = {"free": (-np.inf, None, 0), "contiguity": (-np.inf, None, 0)}
    n_opt = {"free": 0, "contiguity": 0}
    n_feasible = 0
    for head in itertools.product(range(3), repeat=n_head):
        G = np.hstack([np.tile(np.array(head, dtype=np.int8), (len(tail), 1)), tail])
        lam = lb.buckling_load_factor(G)
        feas = lb.contiguity_violation(G) == 0
        n_feasible += int(feas.sum())
        for key, mask in (("free", np.ones_like(feas)), ("contiguity", feas)):
            if not mask.any():
                continue
            vals = np.where(mask, lam, -np.inf)
            m = vals.max()
            if m > best[key][0] + 1e-9:
                best[key] = (m, G[vals.argmax()].tolist(), 0)
                n_opt[key] = int(np.sum(np.isclose(vals, m, rtol=0, atol=1e-6)))
            elif abs(m - best[key][0]) <= 1e-9:
                n_opt[key] += int(np.sum(np.isclose(vals, m, rtol=0, atol=1e-6)))
    out = {
        "n_designs": 3 ** lb.N_STACKS,
        "n_feasible_contiguity": n_feasible,
        "free": {"lambda_cb": best["free"][0], "genes": best["free"][1],
                 "sequence": lb.describe(best["free"][1]), "n_optimal_designs": n_opt["free"]},
        "contiguity": {"lambda_cb": best["contiguity"][0], "genes": best["contiguity"][1],
                       "sequence": lb.describe(best["contiguity"][1]),
                       "n_optimal_designs": n_opt["contiguity"]},
        "modes": f"p, q = {lb.MODES[0]}..{lb.MODES[-1]}",
        "seconds": round(time.time() - t0, 1),
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_optimum.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
