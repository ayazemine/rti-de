"""
L-SHADE, written from the description in Tanabe and Fukunaga (2014), "Improving the search
performance of SHADE using linear population size reduction", IEEE CEC 2014, pp. 1658-1665.

Why not mealpy's ``SHADE.L_SHADE``: in mealpy 3.0.3 that class computes a reduced population
size but keeps evolving all ``pop_size`` agents, so the population is never reduced, and it
stores the accepted trial instead of the replaced parent in the archive.

Settings of the paper (its Table II, values tuned with ParamILS on CEC2013 F21-F25 at D = 10
and 30):
    N_init = 18 D, N_min = 4, memory size H = 6, p = 0.11, archive size = 2.6 N,
    M_F = M_CR = 0.5 initially.
Operators: current-to-pbest/1 with an external archive, binomial crossover, F ~ Cauchy(M_F, 0.1)
(redrawn while <= 0, truncated to 1), CR ~ N(M_CR, 0.1) clipped to [0, 1] (0 when the memory
entry is terminal), weighted Lehmer mean for both memories, a violated bound replaced by the
midpoint between the bound and the parent, linear population size reduction in the number of
evaluations.

This is an independent implementation written from the description in the paper, not the
authors' C++ code.
"""

import numpy as np


def lshade(func, lb, ub, max_fes, seed=None, n_init=None, n_min=4, H=6, p=0.11, arc_rate=2.6):
    """Minimise ``func`` (vectorised: (m, D) array -> m values). Returns (x_best, f_best, fes)."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    dim = lb.size
    n_init = int(n_init or 18 * dim)
    n = n_init

    def evaluate(X):
        f = np.asarray(func(X), dtype=float)
        return np.where(np.isnan(f), np.inf, f)

    X = rng.uniform(lb, ub, (n, dim))
    fit = evaluate(X)
    fes = n
    M_F, M_CR, k = np.full(H, 0.5), np.full(H, 0.5), 0      # M_CR = nan marks the terminal value
    archive = np.empty((0, dim))

    while fes < max_fes:
        order = np.argsort(fit, kind="stable")
        r = rng.integers(H, size=n)
        CR = np.clip(rng.normal(np.nan_to_num(M_CR[r]), 0.1), 0.0, 1.0)
        CR[np.isnan(M_CR[r])] = 0.0
        F = M_F[r] + 0.1 * rng.standard_cauchy(n)
        while True:
            bad = F <= 0.0
            if not bad.any():
                break
            F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        F = np.minimum(F, 1.0)

        pbest = order[rng.integers(max(2, int(round(p * n))), size=n)]
        idx = np.arange(n)
        r1 = rng.integers(n - 1, size=n)
        r1 += r1 >= idx
        union = np.vstack([X, archive])
        r2 = rng.integers(union.shape[0], size=n)
        while True:
            bad = (r2 == idx) | (r2 == r1)
            if not bad.any():
                break
            r2[bad] = rng.integers(union.shape[0], size=int(bad.sum()))

        V = X + F[:, None] * (X[pbest] - X) + F[:, None] * (X[r1] - union[r2])
        V = np.where(V < lb, (lb + X) / 2.0, V)
        V = np.where(V > ub, (ub + X) / 2.0, V)
        cross = rng.random((n, dim)) < CR[:, None]
        cross[idx, rng.integers(dim, size=n)] = True
        U = np.where(cross, V, X)

        m = min(n, max_fes - fes)                # the last generation is cut at the budget
        f_new = np.full(n, np.inf)
        f_new[:m] = evaluate(U[:m])
        fes += m

        better = f_new < fit
        accept = f_new <= fit
        accept[m:] = False
        if better.any():
            archive = np.vstack([archive, X[better]])
            gain = fit[better] - f_new[better]
            gain = np.where(np.isfinite(gain), gain, 0.0)
            w = gain / gain.sum() if gain.sum() > 0 else np.full(gain.size, 1.0 / gain.size)
            s_f, s_cr = F[better], CR[better]
            M_F[k] = np.sum(w * s_f ** 2) / np.sum(w * s_f)
            if np.isnan(M_CR[k]) or s_cr.max() == 0.0:
                M_CR[k] = np.nan
            else:
                M_CR[k] = np.sum(w * s_cr ** 2) / np.sum(w * s_cr)
            k = (k + 1) % H
        X[accept], fit[accept] = U[accept], f_new[accept]

        n_next = max(n_min, int(round((n_min - n_init) / max_fes * fes + n_init)))
        if n_next < n:
            keep = np.argsort(fit, kind="stable")[:n_next]
            X, fit, n = X[keep], fit[keep], n_next
        arc_max = int(round(arc_rate * n))
        if archive.shape[0] > arc_max:
            archive = archive[rng.permutation(archive.shape[0])[:arc_max]]

    best = int(np.argmin(fit))
    return X[best].copy(), float(fit[best]), fes
