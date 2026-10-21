"""
Classic differential evolution, DE/rand/1/bin, written from the description in Storn and Price
(1997), "Differential evolution - a simple and efficient heuristic for global optimization over
continuous spaces", Journal of Global Optimization 11, 341-359.

    v = x_r0 + F (x_r1 - x_r2),  r0, r1, r2 mutually different and different from i
    u_j = v_j if rand_j < CR or j = j_rand, else x_j
    u replaces x when f(u) <= f(x)

Fixed settings: F = 0.5, CR = 0.9, population size N (given by the caller, constant). The
original paper does not prescribe a bound handling; a component outside the box is set to the
bound here.

This is an independent implementation written from the description in the paper.
"""

import numpy as np


def de(func, lb, ub, max_fes, seed=None, n=50, F=0.5, CR=0.9):
    """Minimise ``func`` (vectorised: (m, D) array -> m values). Returns (x_best, f_best, fes)."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    dim = lb.size
    idx = np.arange(n)

    def evaluate(X):
        f = np.asarray(func(X), dtype=float)
        return np.where(np.isnan(f), np.inf, f)

    X = rng.uniform(lb, ub, (n, dim))
    fit = evaluate(X)
    fes = n

    while fes < max_fes:
        r = np.argsort(rng.random((n, n - 1)), axis=1)[:, :3]      # three different indices != i
        r += r >= idx[:, None]
        V = np.clip(X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]]), lb, ub)
        cross = rng.random((n, dim)) < CR
        cross[idx, rng.integers(dim, size=n)] = True
        U = np.where(cross, V, X)

        m = min(n, max_fes - fes)                # the last generation is cut at the budget
        f_new = np.full(n, np.inf)
        f_new[:m] = evaluate(U[:m])
        fes += m
        accept = f_new <= fit
        accept[m:] = False
        X[accept], fit[accept] = U[accept], f_new[accept]

    best = int(np.argmin(fit))
    return X[best].copy(), float(fit[best]), fes
