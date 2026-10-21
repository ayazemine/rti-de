"""
RTI-DE: Rejected-Trial Interpolation Differential Evolution.

A differential evolution variant in which the evaluation of a rejected trial point is used
instead of being discarded.

Components
----------
1. Trial point
       v = x + F (x_e - x) + F (x_r1 - x_r2),
   x_e drawn from the best p N agents, r1 != r2 != i: the DE/current-to-pbest/1 mutation of JADE
   (Zhang and Sanderson, 2009) without the external archive, followed by the binomial crossover
   of classic DE (Storn and Price, 1997) with the fixed rate CR: component j of the trial point
   is v_j with probability CR and for one random j, and x_j otherwise. The displacement is
   d = trial point - x.
2. Selection: v replaces x when f(v) <= f(x).
3. Rejected-trial chain, used after the fraction ``chain_from`` of the evaluation budget.
   An agent whose trial point was rejected does not draw a new displacement at once:
       second point   x + d / 2
       third point    x + t* d, t* the minimiser of the parabola through the three collinear
                      points x, x + d / 2, x + d, limited to [-1, 1]; t* = -1/2 when the parabola
                      is not convex. The point is skipped when t* is closer than 0.02 to a point
                      already evaluated (0, 1/2 or 1).
   An accepted point ends the chain; after the third point a new displacement is drawn.
4. Step factor: F ~ N(mu_F, 0.15) clipped to [0.05, 1], mu_F = 0.5 at the start. After a
   generation in which first trial points were accepted,
       mu_F <- 0.9 mu_F + 0.1 L,   L = sum(w F^2) / sum(w F),
   over the accepted first trial points, w proportional to the improvement (the adaptation rule
   of JADE with the improvement weights of SHADE; points of the chain do not enter).
5. Population size: reduced linearly in the evaluations used from n_init to n_min, the worst
   agents being removed (linear population size reduction of L-SHADE, Tanabe and Fukunaga, 2014).

Bounds: a component of v outside the box is set to the midpoint between the bound and x (before
the crossover), and d is the displacement actually made; the third point is clipped to the box.

CR = 0.9 was chosen on a development setting (CEC2017, D = 30, 300,000 evaluations, seeds 0-4,
among 0.5, 0.7, 0.9 and no crossover) and confirmed on seeds 5-14 on 2026-10-10.
"""

import numpy as np


def rti_de(func, lb, ub, max_fes, seed=None, n_init=None, n_min=4, p=0.11, chain=2, chain_from=0.7,
           CR=0.9, lpsr=True, adapt=True, trace=None):
    """Minimise ``func`` (vectorised: (k, D) array -> k values) within ``max_fes`` evaluations.

    n_init      initial population size (default 18 D)
    n_min       final population size
    p           fraction of the population from which x_e is drawn
    chain       points of the rejected-trial chain: 2 (half step and parabola), 1 (half step
                only) or 0 (no chain); values below 2 are for the ablation
    chain_from  fraction of the budget after which the chain is used
    CR          rate of the binomial crossover (None: no crossover; for the ablation)

    Switches for the ablation (the algorithm is the default):
    lpsr        False: the population keeps its initial size
    adapt       False: mu_F stays at 0.5

    trace       None, or a list to which one dict is appended per generation (the run itself is
                not changed: no random number is drawn and nothing is evaluated for the trace):
                  fes, n, mu_F, f_best     evaluations used, population size, mean step factor and
                                           best value after the generation (n before the reduction)
                  evaluated, accepted,     candidates of the generation by stage [first trial point,
                  improved                 half step, parabola point]: evaluated, accepted
                                           (f <= f of the agent), strictly better
                  skipped                  third points left out by the skip rule
                  theta, theta_accepted    factor t* of every third point evaluated and whether it
                                           was accepted
                The list is filled as the run goes, so it is valid when ``func`` ends the run early.

    Returns (x_best, f_best, evaluations used)."""
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
    mu_F = 0.5
    stage = np.zeros(n, dtype=int)               # 0: first trial point, 1: half step, 2: parabola
    D = np.zeros((n, dim))                       # displacement of the running chain
    T = np.ones(n)                               # factor of d in the current candidate
    F = np.full(n, 0.5)                          # step factor with which d was drawn
    f_one, f_half = np.zeros(n), np.zeros(n)     # values at x + d and x + d / 2

    while fes < max_fes:
        # --- third point: minimiser of the parabola through (0, fit), (1/2, f_half), (1, f_one)
        s2 = np.nonzero(stage == 2)[0]
        n_skip = 0
        if s2.size:
            d_one, d_half = f_one[s2] - fit[s2], f_half[s2] - fit[s2]
            with np.errstate(all="ignore"):
                beta = (d_half - 0.5 * d_one) / -0.25
                t = -(d_one - beta) / (2.0 * beta)
            t = np.where(np.isfinite(t) & (beta > 0), t, -0.5)
            ok = np.isfinite(d_one) & np.isfinite(d_half)
            t = np.clip(np.where(ok, t, 0.0), -1.0, 1.0)
            ok &= (np.abs(t) >= 0.02) & (np.abs(t - 0.5) >= 0.02) & (np.abs(t - 1.0) >= 0.02)
            T[s2[ok]] = t[ok]
            stage[s2[~ok]] = 0
            n_skip = int((~ok).sum())

        # --- second point: half step
        T[stage == 1] = 0.5

        # --- first trial point: new displacement
        s0 = np.nonzero(stage == 0)[0]
        if s0.size:
            k = s0.size
            order = np.argsort(fit, kind="stable")
            elite = X[order[rng.integers(max(2, int(round(p * n))), size=k)]]
            f = np.clip(rng.normal(mu_F, 0.15, k), 0.05, 1.0)
            r1 = rng.integers(n - 1, size=k)
            r1 += r1 >= s0
            r2 = rng.integers(n, size=k)
            while True:
                bad = (r2 == s0) | (r2 == r1)
                if not bad.any():
                    break
                r2[bad] = rng.integers(n, size=int(bad.sum()))
            V = X[s0] + f[:, None] * (elite - X[s0]) + f[:, None] * (X[r1] - X[r2])
            V = np.where(V < lb, (lb + X[s0]) / 2.0, V)
            V = np.where(V > ub, (ub + X[s0]) / 2.0, V)
            if CR is not None:
                cross = rng.random((k, dim)) < CR
                cross[np.arange(k), rng.integers(dim, size=k)] = True
                V = np.where(cross, V, X[s0])
            D[s0], F[s0], T[s0] = V - X[s0], f, 1.0

        U = np.clip(X + T[:, None] * D, lb, ub)
        k = min(n, max_fes - fes)                # the last generation is cut at the budget
        f_new = np.full(n, np.inf)
        f_new[:k] = evaluate(U[:k])
        fes += k

        better = f_new < fit
        accept = f_new <= fit
        accept[k:] = False

        # --- step factor: accepted first trial points only
        counted = better & (stage == 0)
        if adapt and counted.any():
            gain = fit[counted] - f_new[counted]
            gain = np.where(np.isfinite(gain), gain, 0.0)
            w = gain / gain.sum() if gain.sum() > 0 else np.full(gain.size, 1.0 / gain.size)
            s_f = F[counted] * np.abs(T[counted])
            den = np.sum(w * s_f)
            lehmer = np.sum(w * s_f ** 2) / den if den > 0 else 0.0
            mu_F = float(np.clip(0.9 * mu_F + 0.1 * lehmer, 0.05, 1.0))

        if trace is not None:
            done = np.arange(n) < k
            by_stage = [done & (stage == s) for s in (0, 1, 2)]
            trace.append(dict(
                fes=fes, n=n, mu_F=mu_F, f_best=float(min(fit.min(), f_new[:k].min())),
                evaluated=[int(m.sum()) for m in by_stage],
                accepted=[int((m & accept).sum()) for m in by_stage],
                improved=[int((m & better).sum()) for m in by_stage],
                skipped=n_skip, theta=T[by_stage[2]].copy(), theta_accepted=accept[by_stage[2]].copy()))

        # --- selection and next stage of the chain
        rejected = ~accept
        rejected[k:] = False
        f_one = np.where(rejected & (stage == 0), f_new, f_one)
        f_half = np.where(rejected & (stage == 1), f_new, f_half)
        X[accept], fit[accept] = U[accept], f_new[accept]
        nxt = np.where(rejected, stage + 1, 0)
        nxt[nxt > chain] = 0
        if fes < chain_from * max_fes:
            nxt[:] = 0
        nxt[k:] = stage[k:]
        stage = nxt

        # --- linear population size reduction
        n_next = max(n_min, int(round((n_min - n_init) / max_fes * fes + n_init))) if lpsr else n
        if n_next < n:
            keep = np.argsort(fit, kind="stable")[:n_next]
            X, fit, stage, D, T, F = X[keep], fit[keep], stage[keep], D[keep], T[keep], F[keep]
            f_one, f_half = f_one[keep], f_half[keep]
            n = n_next

    best = int(np.argmin(fit))
    return X[best].copy(), float(fit[best]), fes
