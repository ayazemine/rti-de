#!/usr/bin/env python3
"""
Paper figures for RTI-DE (vector PDF + SVG), from the results of ``run_experiments.py``.

Every figure reads the result files in ``results_cache/paper/<suite>/`` and writes

    outputs/figures/fig_<name>.pdf   outputs/figures/fig_<name>.svg   outputs/figures/fig_<name>.json

where the JSON records the sources, settings, git commit and the numbers that
belong in the caption (budget, critical difference, floor values, ...).

Usage (from experiments/)
-------------------------
    python -W ignore scripts/make_figures.py --list
    python -W ignore scripts/make_figures.py                      # all figures
    python -W ignore scripts/make_figures.py --only conv_cec2022 cd_cec2017
    python -W ignore scripts/make_figures.py --dry-run            # build everything, write nothing

All figures only read result files.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as fl  # noqa: E402
from figlib import ALGORITHMS, DOUBLE_COL, ERROR_FLOOR, PROPOSED, STYLE, THREAT, display, plt, rx  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def algo_handles(markers=True):
    return [Line2D([0], [0], label=display(a), **fl.line_style(a, markers)) for a in ALGORITHMS]


def panel_grid(n_panels, ncols=3, row_h=1.7, width=DOUBLE_COL):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, row_h * nrows + 0.35),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    return fig, list(axes.flat[:n_panels])


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def finite_median(a):
    """Column-wise median; a column is NaN unless more than half of the runs are finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.median(a, axis=0)
    return np.where(np.isfinite(m), m, np.nan)


# ---------------------------------------------------------------------------
# Convergence (median best-so-far error against function evaluations)
# ---------------------------------------------------------------------------

def draw_convergence(ax, S, fid, floor):
    n = S.settings["agents"]
    for a in ALGORITHMS:
        grid = fl.fe_grid(n, int(S.fes(fid, a).max()))
        med = finite_median(S.curves(fid, a, grid))
        if floor is not None:
            med = np.maximum(med, floor)
        ax.plot(grid, med, **fl.line_style(a))
        ax.plot(grid[-1], med[-1], ls="none", marker=STYLE[a]["marker"], color=STYLE[a]["color"],
                ms=3, zorder=STYLE[a]["zorder"])
    ax.set_xscale("log")
    ax.grid(True, which="major")


def convergence(ctx, suite, fids, name):
    S = ctx.load(suite)
    fig, axes = panel_grid(len(fids))
    for ax, fid in zip(axes, fids):
        draw_convergence(ax, S, fid, ERROR_FLOOR)
        ax.set_yscale("log")
        ax.set_title(fid, pad=2)
    fig.supxlabel("Function evaluations", fontsize=7)
    fig.supylabel(r"Median error $f - F^*$", fontsize=7)
    # Legend below the figure, under the shared x label (an "outside" legend of the constrained
    # layout would be placed on top of the figure-level x label); savefig uses bbox="tight".
    fig.legend(handles=algo_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=9, frameon=False)
    ctx.out.save(fig, name, [S], notes={
        "curve": f"median over {S.n_runs} runs of the best value evaluated so far, minus F*",
        "x_axis": "function evaluations counted by run_experiments.Counter",
        "end_marker": "each curve ends at the largest number of FEs the algorithm used; "
                      "a finished run keeps its final value",
        "floor": f"errors below {ERROR_FLOOR:g} are drawn at {ERROR_FLOOR:g}",
        "functions": fids,
        "mean_FEs": {a: float(np.mean([S.fes(f, a).mean() for f in fids])) for a in ALGORITHMS},
    })


# ---------------------------------------------------------------------------
# Box plots
# ---------------------------------------------------------------------------

def draw_box(ax, data, log=True, labels=True):
    shown = [d[np.isfinite(d)] for d in data]
    if log:
        shown = [np.maximum(d, ERROR_FLOOR) for d in shown]
    bp = ax.boxplot(shown, patch_artist=True, widths=0.62, whis=1.5,
                    flierprops=dict(marker=".", markersize=2, markeredgewidth=0.3),
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                    boxprops=dict(linewidth=0.5))
    for patch, a in zip(bp["boxes"], ALGORITHMS):
        patch.set_facecolor(STYLE[a]["color"])
        patch.set_alpha(0.75)
    for fly, a in zip(bp["fliers"], ALGORITHMS):
        fly.set_markeredgecolor(STYLE[a]["color"])
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(1, len(ALGORITHMS) + 1))
    ax.set_xticklabels([display(a) for a in ALGORITHMS] if labels else [], rotation=90)
    ax.grid(True, axis="y", which="major")


def boxplots(ctx, suite, fids, name, budget=None):
    S = ctx.load(suite)
    fig, axes = panel_grid(len(fids), row_h=1.8)
    for ax, fid in zip(axes, fids):
        data = [S.final(fid, a) if budget is None else S.at_fes(fid, a, budget) for a in ALGORITHMS]
        draw_box(ax, data)
        ax.set_title(fid, pad=2)
    what = "final error" if budget is None else f"error after {budget:,} function evaluations"
    fig.supylabel(rf"{what[0].upper() + what[1:]} $f - F^*$", fontsize=7)
    ctx.out.save(fig, name, [S], notes={
        "values": f"{what} of each of the {S.n_runs} runs",
        "floor": f"errors below {ERROR_FLOOR:g} count as 0 and are drawn at {ERROR_FLOOR:g}",
        "box": "median, quartiles, whiskers at 1.5 IQR, points beyond the whiskers",
        "functions": fids, "budget_FEs": budget,
    })


# ---------------------------------------------------------------------------
# Critical difference diagram (Friedman + Nemenyi; Demšar 2006)
# ---------------------------------------------------------------------------

def critical_difference(ctx, suite):
    from scipy import stats
    S = ctx.load(suite)
    k, n = len(ALGORITHMS), len(S.fids)
    means = np.array([[np.mean(S.final(fid, a)) for a in ALGORITHMS] for fid in S.fids])
    ranks = np.array([stats.rankdata(row) for row in means])
    avg = ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*ranks.T)
    q = float(stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2.0))     # Nemenyi q_0.05
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

    order = np.argsort(avg)
    srt = avg[order]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and srt[j + 1] - srt[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))

    half = (k + 1) // 2
    step, y0 = 0.30, 0.22 * len(cliques) + 0.45
    height = y0 + step * half + 0.9
    fig = plt.figure(figsize=(DOUBLE_COL * 0.8, 0.42 * height + 0.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0.0, k + 1.0)
    ax.set_ylim(-(y0 + step * half + 0.1), 0.95)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.08], color="black", lw=0.8)
        ax.text(r, 0.13, str(r), ha="center", va="bottom")
    ax.plot([1, 1 + cd], [0.62, 0.62], color="black", lw=1.2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.57, 0.67], color="black", lw=0.8)
    ax.text(1 + cd / 2, 0.70, f"CD = {cd:.2f}", ha="center", va="bottom")
    for m, (i, j) in enumerate(cliques):
        y = -0.18 - 0.22 * m
        ax.plot([srt[i] - 0.04, srt[j] + 0.04], [y, y], color="black", lw=2.2, solid_capstyle="butt")
    for pos, idx in enumerate(order):
        a, r = ALGORITHMS[idx], avg[idx]
        left = pos < half
        level = pos if left else k - 1 - pos
        y = -(y0 + step * level)
        xe = 0.75 if left else k + 0.25
        col = STYLE[a]["color"]
        ax.plot([r, r, xe], [0, y, y], color=col, lw=1.4 if a == PROPOSED else 0.8)
        ax.text(xe - 0.06 if left else xe + 0.06, y, f"{display(a)} ({r:.2f})", ha="right" if left else "left",
                va="center", color=col, fontweight="bold" if a == PROPOSED else "normal")
    ctx.out.save(fig, f"fig_cd_{suite}", [S], notes={
        "ranks": "per function, algorithms ranked by mean final error over the runs (1 = best), then averaged",
        "average_rank": dict(zip(ALGORITHMS, np.round(avg, 4))),
        "friedman_chi2": float(chi2), "friedman_p": float(p), "functions": n, "algorithms": k,
        "post_hoc": "Nemenyi, alpha = 0.05", "q_alpha": q, "critical_difference": float(cd),
        "groups_not_significantly_different": [[ALGORITHMS[order[t]] for t in range(i, j + 1)]
                                               for i, j in cliques],
        "reference": "Demšar (2006), JMLR 7, 1-30",
    })


# ---------------------------------------------------------------------------
# ECDF of the function evaluations needed to reach target errors
# ---------------------------------------------------------------------------

N_TARGETS = 21


def ecdf(ctx, suite):
    S = ctx.load(suite)
    n = S.settings["agents"]
    hits = {a: [] for a in ALGORITHMS}
    targets_used = {}
    for fid in S.fids:
        fs = S.f_star(fid)
        start = [fl.trace_on_grid(r["trace"], np.array([n]))[0] - fs
                 for a in ALGORITHMS for r in S.runs[fid][a]]
        best = min(r["trace"][-1, 1] - fs for a in ALGORITHMS for r in S.runs[fid][a])
        hi, lo = float(np.median(start)), max(float(best), ERROR_FLOOR)
        targets = np.geomspace(hi, lo, N_TARGETS) if hi > lo else np.full(N_TARGETS, lo)
        targets_used[fid] = [hi, lo]
        for a in ALGORITHMS:
            for r in S.runs[fid][a]:
                err = r["trace"][:, 1] - fs
                err = np.where(err < ERROR_FLOOR, 0.0, err)             # non-increasing
                idx = np.searchsorted(-err, -targets, side="left")       # first entry with err <= target
                fe = np.where(idx < len(err), r["trace"][np.minimum(idx, len(err) - 1), 0], np.inf)
                hits[a].append(fe)
    total = len(S.fids) * S.n_runs * N_TARGETS
    fig, ax = plt.subplots(figsize=(fl.SINGLE_COL * 1.25, 2.9), layout="constrained")
    final = {}
    for a in ALGORITHMS:
        h = np.sort(np.concatenate(hits[a]))
        h = h[np.isfinite(h)]
        end = max(int(S.fes(f, a).max()) for f in S.fids)
        x = np.r_[h, end]
        y = np.r_[np.arange(1, len(h) + 1), len(h)] / total
        ax.step(x, y, where="post", **fl.line_style(a))
        ax.plot(x[-1], y[-1], ls="none", marker=STYLE[a]["marker"], color=STYLE[a]["color"], ms=3)
        final[a] = float(len(h) / total)
    ax.set_xscale("log")
    ax.set_xlim(left=n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Fraction of (function, target, run) reached")
    ax.grid(True, which="major")
    ax.legend(handles=algo_handles(), loc="upper left", frameon=False, ncol=2)
    ctx.out.save(fig, f"fig_ecdf_{suite}", [S], notes={
        "target_rule": f"per function, {N_TARGETS} error targets spaced log-uniformly from the median error "
                       f"after the first {n} evaluations (all algorithms and runs) down to the best error "
                       f"reached by any algorithm in any run (not below {ERROR_FLOOR:g}); fixed before plotting",
        "targets_high_low": targets_used, "triples": total, "fraction_reached": final,
        "reference": "Hansen et al. (2021), doi:10.1080/10556788.2020.1808977",
    })


# ---------------------------------------------------------------------------
# CEC2022 landscapes at D = 2 (F6-F8 are not defined at D = 2)
# ---------------------------------------------------------------------------

_LAND = {}


def landscape(fid, points=121):
    if fid not in _LAND:
        from cec2022_functions import get_cec2022_function
        f = get_cec2022_function(fid, ndim=2, allow_2d=True)
        xs = np.linspace(float(f.lb[0]), float(f.ub[0]), points)
        ys = np.linspace(float(f.lb[1]), float(f.ub[1]), points)
        Z = np.array([[f.evaluate(np.array([x, y])) for x in xs] for y in ys]) - float(f.f_bias)
        _LAND[fid] = (xs, ys, np.log10(np.maximum(Z, 0.0) + 1.0), np.asarray(f.x_global, float))
    return _LAND[fid]


def draw_landscape(ax, fid, filled=True):
    xs, ys, L, x_star = landscape(fid)
    if filled:
        cs = ax.contourf(xs, ys, L, levels=14, cmap="viridis")
        ax.contour(xs, ys, L, levels=14, colors="white", linewidths=0.15)
    else:
        cs = ax.contour(xs, ys, L, levels=10, colors="0.6", linewidths=0.3)
    ax.plot(x_star[0], x_star[1], marker="*", color=STYLE[PROPOSED]["color"], ms=5, mec="white", mew=0.3,
            ls="none", zorder=20)
    ax.set_aspect("equal")
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    return cs


def landscapes(ctx):
    from cec2022_functions import PLOT_2D_FUNCTIONS
    fig, axes = panel_grid(len(PLOT_2D_FUNCTIONS), row_h=2.2)
    for ax, fid in zip(axes, PLOT_2D_FUNCTIONS):
        cs = draw_landscape(ax, fid)
        ax.set_title(fid, pad=2)
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    fig.supxlabel(r"$x_1$", fontsize=7)
    fig.supylabel(r"$x_2$", fontsize=7)
    ctx.out.save(fig, "fig_landscape_cec2022", notes={
        "functions": list(PLOT_2D_FUNCTIONS),
        "not_shown": "F6, F7, F8 (hybrid functions) are not defined for D = 2",
        "colour": "log10(f - F* + 1) on a 121 x 121 grid of the official CEC2022 C code at D = 2",
        "star": "global optimum x* (shift vector of the official data)",
    })


# ---------------------------------------------------------------------------
# Engineering problems
# ---------------------------------------------------------------------------

def best_run(S, algo):
    runs = S.runs[S.name][algo]
    f = np.array([r["f"] for r in runs])
    if not np.isfinite(f).any():
        return None
    return runs[int(np.argmin(f))]


GENE_FACE = {0: "#4C72B0", 1: "#DDCC77", 2: "#CC6677"}
GENE_HATCH = {0: "---", 1: "xxx", 2: "|||"}
GENE_TEXT = {0: r"$0_2$", 1: r"$\pm45$", 2: r"$90_2$"}


def draw_plate(ax, lam):
    a, b = lam.A_LEN, lam.B_LEN
    ax.add_patch(Rectangle((0, 0), a, b, fc="#F2F2F2", ec="black", lw=1.0))
    ax.add_patch(Rectangle((0.25, 0.25), a - 0.5, b - 0.5, fill=False, ec="0.4", lw=0.5, ls=(0, (3, 2))))
    ax.text(a / 2, b - 0.9, "simply supported on all four edges", ha="center", va="top", fontsize=6, color="0.3")
    arrow = dict(arrowstyle="-|>", mutation_scale=6, lw=0.8, color="black")
    for y in np.linspace(1, b - 1, 5):
        ax.add_patch(FancyArrowPatch((-2.2, y), (-0.15, y), **arrow))
        ax.add_patch(FancyArrowPatch((a + 2.2, y), (a + 0.15, y), **arrow))
    for x in np.linspace(1.5, a - 1.5, 8):
        ax.add_patch(FancyArrowPatch((x, -2.2), (x, -0.15), **arrow))
        ax.add_patch(FancyArrowPatch((x, b + 2.2), (x, b + 0.15), **arrow))
    ax.text(-2.4, b / 2, rf"$N_x$ = {lam.NX:g} lb/in", rotation=90, ha="right", va="center")
    ax.text(a + 2.4, b / 2, r"$N_x$", rotation=90, ha="left", va="center")
    ax.text(a / 2, b + 2.4, rf"$N_y$ = {lam.NY:g} lb/in", ha="center", va="bottom")
    ax.text(a / 2, -2.4, r"$N_y$", ha="center", va="top")
    # fibre orientations of the three stack types
    cx, cy, L = a / 2, b / 2 - 0.6, 2.6
    for g, angles in lam.STACKS.items():
        for ang in sorted(set(angles)):
            t = np.radians(ang)
            ax.plot([cx - L * np.cos(t), cx + L * np.cos(t)], [cy - L * np.sin(t), cy + L * np.sin(t)],
                    color=GENE_FACE[g], lw=1.6, solid_capstyle="round")
    ax.text(cx + L + 0.3, cy, "0°", va="center", fontsize=6)
    ax.text(cx, cy + L + 0.2, "90°", ha="center", va="bottom", fontsize=6)
    ax.text(cx + 0.74 * L + 0.2, cy + 0.74 * L, "+45°", fontsize=6)
    ax.text(cx + 0.74 * L + 0.2, cy - 0.74 * L, "−45°", va="top", fontsize=6)
    ax.add_patch(FancyArrowPatch((0.6, 0.6), (3.6, 0.6), **arrow))
    ax.add_patch(FancyArrowPatch((0.6, 0.6), (0.6, 3.6), **arrow))
    ax.text(3.8, 0.6, r"$x$", va="center")
    ax.text(0.6, 3.8, r"$y$", ha="center", va="bottom")
    ax.annotate("", (0, -4.6), (a, -4.6), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(a / 2, -4.8, rf"$a$ = {a:g} in ({a * 25.4:.0f} mm)", ha="center", va="top")
    ax.annotate("", (a + 5.0, 0), (a + 5.0, b), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(a + 5.3, b / 2, rf"$b$ = {b:g} in ({b * 25.4:.0f} mm)", rotation=90, ha="left", va="center")
    ax.set_xlim(-5.5, a + 7.5)
    ax.set_ylim(-7.0, b + 4.5)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_stack(ax, lam, genes, title):
    full = list(genes) + list(genes)[::-1]                # outer surface -> mid-plane -> outer surface
    n = len(full)
    for i, g in enumerate(full):
        ax.add_patch(Rectangle((0, n - 1 - i), 1, 1, fc=GENE_FACE[g], ec="black", lw=0.3,
                               hatch=GENE_HATCH[g], alpha=0.9))
        ax.text(1.06, n - 0.5 - i, GENE_TEXT[g], va="center", ha="left", fontsize=4.6)
    ax.plot([-0.12, 1.5], [n / 2, n / 2], color="black", lw=0.8, ls=(0, (4, 2)))
    ax.text(-0.16, n / 2, "mid-plane", ha="right", va="center", fontsize=6, rotation=90)
    ax.set_xlim(-0.45, 1.6)
    ax.set_ylim(-0.3, n + 0.3)
    ax.set_title(title, pad=3)
    ax.axis("off")


def composite_figure(ctx):
    import laminate_buckling as lam
    suites = [ctx.load("composite"), ctx.load("composite_contiguity")]
    designs, notes = [], {}
    for S in suites:
        r = best_run(S, PROPOSED)
        genes = lam.decode(r["x"])
        contig = S.name == "composite_contiguity"
        if abs(float(lam.objective(r["x"], contiguity=contig)) - r["f"]) > 1e-9:
            sys.exit(f"ERROR: {S.name}: the stored best design does not reproduce its stored objective value.")
        lcb = float(lam.buckling_load_factor(genes))
        ref = rx.load_objective(S.name, S.name, None)["reference"]
        hits = int(np.sum(np.abs(-S.final(S.name, PROPOSED) - ref) < 1e-6))
        designs.append((genes, lcb, ref, contig))
        notes[S.name] = {"run": r["run"], "seed": r["seed"], "genes_outer_to_midplane": genes.tolist(),
                         "sequence": lam.describe(genes), "lambda_cb": lcb, "global_optimum": ref,
                         "contiguity_violations": int(lam.contiguity_violation(genes)),
                         "proposed_runs_at_optimum": f"{hits}/{S.n_runs}"}
    # Three separate columns (plate | layup | layup) with a clear gap, so that neither the
    # panel titles nor the property text can run into a neighbouring panel.
    plt.rcParams["hatch.linewidth"] = 0.4
    fig = plt.figure(figsize=(DOUBLE_COL, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.9, 1, 1], height_ratios=[2.5, 1.25],
                          left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.22, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    draw_plate(ax, lam)
    ax.set_title("(a) Plate, loading and fibre orientations", pad=3)
    for col, (genes, lcb, ref, contig) in enumerate(designs, 1):
        axs = fig.add_subplot(gs[:, col])
        label = "With contiguity" if contig else "Unconstrained"
        draw_stack(axs, lam, genes, f"({'bc'[col - 1]}) {label}\n"
                                    rf"$\lambda_{{cb}}$ = {lcb:.2f}" "\n" rf"(optimum {ref:.2f})")
    axt = fig.add_subplot(gs[1, 0])
    axt.axis("off")
    h = lam.N_PLIES * lam.PLY_T
    props = (f"Graphite-epoxy:  $E_1$ = {lam.E1 / 1e6:g} Msi,  $E_2$ = {lam.E2 / 1e6:g} Msi,  "
             f"$G_{{12}}$ = {lam.G12 / 1e6:g} Msi,  $\\nu_{{12}}$ = {lam.NU12:g}\n"
             f"{lam.N_PLIES} plies of {lam.PLY_T:g} in ({lam.PLY_T * 25.4:.3f} mm), total thickness {h:g} in\n"
             f"Symmetric, balanced laminate; {lam.N_STACKS} design variables,\n"
             f"one two-ply stack each (listed from the outer surface to the mid-plane)\n"
             f"(b), (c): best stacking sequences found by {PROPOSED}")
    axt.text(0.02, 1.0, props, va="top", ha="left", fontsize=6.5, linespacing=1.45)
    axt.legend(handles=[Patch(fc=GENE_FACE[g], ec="black", lw=0.3, hatch=GENE_HATCH[g], label=GENE_TEXT[g])
                        for g in (0, 1, 2)], loc="lower left", ncol=3, frameon=False,
               bbox_to_anchor=(0.0, 0.0), handlelength=1.6, title="Stack type", title_fontsize=6.5,
               alignment="left")
    notes["drawing"] = "schematic, not to scale; stacking sequences, lambda_cb and plate data are read from " \
                       "the result files and applications/composite/laminate_buckling.py"
    notes["sources"] = ("Le Riche (1994), PhD dissertation, Virginia Tech, hdl:10919/38270; "
                        "Ahmid et al. (2019), doi:10.46254/j.ieom.20190201")
    ctx.out.save(fig, "fig_composite_laminate", suites, notes=notes, extra_sources=[
        os.path.join(fl.ROOT, "applications", "composite", "laminate_buckling.py"),
        os.path.join(fl.ROOT, "applications", "composite", "reference_optimum.json")])


def uav_path(pp, x):
    nx, ny, nz = pp.spherical_to_cart(x)
    xa = np.r_[pp.START[0], nx, pp.END[0]]
    ya = np.r_[pp.START[1], ny, pp.END[1]]
    za = np.r_[pp.START[2], nz, pp.END[2]]
    return xa, ya, za + pp.terrain_height(xa, ya)


def draw_uav_symbol(ax, x, y, heading, size):
    c, s = np.cos(heading), np.sin(heading)
    for ang in (np.pi / 4, 3 * np.pi / 4):
        dx, dy = size * np.cos(ang), size * np.sin(ang)
        px, py = c * dx - s * dy, s * dx + c * dy
        ax.plot([x - px, x + px], [y - py, y + py], color="black", lw=0.9, zorder=31)
        for sgn in (-1, 1):
            ax.add_patch(Circle((x + sgn * px, y + sgn * py), size * 0.42, fc="white", ec="black",
                                lw=0.6, zorder=32))
    ax.add_patch(Circle((x, y), size * 0.22, fc="black", ec="black", zorder=33))


def uav_figure(ctx):
    import path_planning as pp
    S = ctx.load("uav")
    r = best_run(S, PROPOSED)
    if r is None:
        sys.exit(f"ERROR: uav: {PROPOSED} has no feasible run; the scenario figure cannot be drawn.")
    if abs(pp.cost(r["x"]) - r["f"]) > 1e-9 * max(1.0, abs(r["f"])):
        sys.exit("ERROR: uav: the stored best path does not reproduce its stored cost.")
    xa, ya, zabs = uav_path(pp, r["x"])
    J = pp.cost_components(r["x"])
    # best competitor (lowest best cost), drawn for comparison if its path reproduces its cost
    comp = None
    cands = [(best_run(S, a), a) for a in ALGORITHMS[1:]]
    cands = sorted([(c["f"], a, c) for c, a in cands if c is not None], key=lambda t: t[0])
    if cands and np.all(np.isfinite(cands[0][2]["x"])) and \
            abs(pp.cost(cands[0][2]["x"]) - cands[0][0]) <= 1e-9 * max(1.0, abs(cands[0][0])):
        comp = cands[0]

    # Both plots get the same box: equal height, top and bottom edges aligned. The height follows
    # from the map's aspect ratio, so the equal-aspect map fills its box exactly. The scenario data
    # go in a full-width text block below the two plots.
    W, plot_w, text_h = DOUBLE_COL, 3.0, 0.95
    plot_h = plot_w * (pp.YMAX - pp.YMIN) / (pp.XMAX - pp.XMIN)
    bottom_in = text_h + 0.50
    H = bottom_in + plot_h + 0.25
    fig = plt.figure(figsize=(W, H))
    box = lambda left_in: [left_in / W, bottom_in / H, plot_w / W, plot_h / H]
    ax = fig.add_axes(box(0.50))
    step = 4
    xs = np.arange(1, pp.MAPSIZE_X + 1)[::step]
    ys = np.arange(1, pp.MAPSIZE_Y + 1)[::step]
    cs = ax.contour(xs, ys, pp.H[::step, ::step], levels=10, colors="0.45", linewidths=0.35)
    ax.clabel(cs, fontsize=5, fmt="%d", inline_spacing=1)
    for tx, ty, _, tr in pp.THREATS:
        ax.add_patch(Circle((tx, ty), tr, fc=THREAT, alpha=0.22, ec="none"))
        ax.add_patch(Circle((tx, ty), tr, fill=False, ec=THREAT, lw=0.8))
        ax.add_patch(Circle((tx, ty), tr + pp.DRONE_SIZE + pp.DANGER_DIST, fill=False, ec=THREAT,
                            lw=0.5, ls=(0, (3, 2))))
        ax.text(tx, ty, f"R = {tr:g}", ha="center", va="center", fontsize=5, color="#3F1D6B")
    if comp is not None:
        cx, cy, _ = uav_path(pp, comp[2]["x"])
        ax.plot(cx, cy, color=STYLE[comp[1]]["color"], lw=0.9, ls="--", marker=".", ms=3, zorder=20)
    ax.plot(xa, ya, color=STYLE[PROPOSED]["color"], lw=1.6, marker="o", ms=2.6, mfc="white", mew=0.7, zorder=25)
    ax.plot(xa[0], ya[0], marker="s", ms=5, color="black", ls="none", zorder=30)
    ax.plot(xa[-1], ya[-1], marker="*", ms=8, color="black", ls="none", zorder=30)
    ax.annotate("Start", (xa[0], ya[0]), (8, -3), textcoords="offset points", fontsize=6)
    ax.annotate("Goal", (xa[-1], ya[-1]), (6, 4), textcoords="offset points", fontsize=6)
    seg = len(xa) // 2
    mx, my = (xa[seg] + xa[seg + 1]) / 2, (ya[seg] + ya[seg + 1]) / 2
    draw_uav_symbol(ax, mx, my, np.arctan2(ya[seg + 1] - ya[seg], xa[seg + 1] - xa[seg]), 16.0)
    ax.annotate("UAV", (mx, my), (-28, 10), textcoords="offset points", fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.4))
    ax.set_xlim(pp.XMIN, pp.XMAX)
    ax.set_ylim(pp.YMIN, pp.YMAX)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (map units)")
    ax.set_ylabel(r"$y$ (map units)")
    ax.set_title("(a) Terrain contours, threats and best paths", pad=3)
    handles = [Line2D([0], [0], color=STYLE[PROPOSED]["color"], lw=1.6, marker="o", ms=2.6, mfc="white",
                      label=f"{PROPOSED} (cost {r['f']:.1f})")]
    if comp is not None:
        handles.append(Line2D([0], [0], color=STYLE[comp[1]]["color"], lw=0.9, ls="--", marker=".",
                              label=f"{display(comp[1])} (cost {comp[0]:.1f})"))
    handles += [Line2D([0], [0], color=THREAT, lw=0.8, label="Threat cylinder"),
                Line2D([0], [0], color=THREAT, lw=0.5, ls=(0, (3, 2)), label="Danger zone"),
                Line2D([0], [0], color="0.45", lw=0.35, label="Terrain elevation")]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.5)

    # elevation profile along the path of the proposed algorithm
    axp = fig.add_axes(box(W - plot_w - 0.08))
    s_all, g_all, p_all = [], [], []
    s0 = 0.0
    for j in range(len(xa) - 1):
        t = np.linspace(0, 1, 40)
        px, py = xa[j] + t * (xa[j + 1] - xa[j]), ya[j] + t * (ya[j + 1] - ya[j])
        d = np.hypot(xa[j + 1] - xa[j], ya[j + 1] - ya[j])
        s_all.append(s0 + t * d)
        g_all.append(pp.terrain_height(px, py))
        p_all.append(zabs[j] + t * (zabs[j + 1] - zabs[j]))
        s0 += d
    s_all, g_all, p_all = map(np.concatenate, (s_all, g_all, p_all))
    s_nodes = np.r_[0, np.cumsum(np.hypot(np.diff(xa), np.diff(ya)))]
    axp.fill_between(s_all, g_all + pp.ZMIN, g_all + pp.ZMAX, color="#66CCEE", alpha=0.25, lw=0,
                     label=f"Allowed altitude\n({pp.ZMIN:g}–{pp.ZMAX:g} above ground)")
    axp.fill_between(s_all, 0, g_all, color="0.75", lw=0, label="Terrain")
    axp.plot(s_all, g_all, color="0.3", lw=0.6)
    axp.plot(s_all, p_all, color=STYLE[PROPOSED]["color"], lw=1.4, label=f"{PROPOSED} path")
    axp.plot(s_nodes, zabs, ls="none", marker="o", ms=2.6, mfc="white", mew=0.7, color=STYLE[PROPOSED]["color"])
    axp.set_xlim(0, s_all[-1])
    axp.set_ylim(bottom=0)
    axp.set_xlabel("Horizontal distance along the path (map units)")
    axp.set_ylabel("Elevation (map units)")
    axp.set_title(f"(b) Elevation profile of the {PROPOSED} path", pad=3)
    axp.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.5,
               handlelength=1.4)

    axt = fig.add_axes([0.50 / W, 0.0, 1.0 - 0.58 / W, text_h / H])
    axt.axis("off")
    feas = int(np.isfinite(S.final("uav", PROPOSED)).sum())
    clearance = float(np.min(p_all - g_all))
    text = (f"Terrain: Christmas Island elevation map, {pp.MAPSIZE_X} × {pp.MAPSIZE_Y} cells, "
            f"elevation {pp.H.min():.0f}–{pp.H.max():.0f}\n"
            f"Threats: {len(pp.THREATS)} cylinders, radius {pp.THREATS[:, 3].min():g}–{pp.THREATS[:, 3].max():g}; "
            f"danger distance {pp.DANGER_DIST:g}\n"
            f"UAV: size {pp.DRONE_SIZE:g}; turning angle ≤ {pp.TURNING_MAX:g}°, "
            f"climb-angle change ≤ {pp.CLIMB_MAX:g}°\n"
            f"Path: start ({pp.START[0]:g}, {pp.START[1]:g}, {pp.START[2]:g}), "
            f"goal ({pp.END[0]:g}, {pp.END[1]:g}, {pp.END[2]:g}), {pp.N_NODES} nodes\n"
            f"Best {PROPOSED} path: length $J_1$ = {J[0]:.1f}, threat $J_2$ = {J[1]:.2f},\n"
            f"altitude $J_3$ = {J[2]:.2f}, smoothness $J_4$ = {J[3]:.2f}\n"
            f"Cost $F = {pp.B1:g}J_1 + {pp.B2:g}J_2 + {pp.B3:g}J_3 + {pp.B4:g}J_4$ = {r['f']:.1f}\n"
            f"Minimum ground clearance {clearance:.1f}; feasible {PROPOSED} runs {feas}/{S.n_runs}")
    lines = text.split("\n")                       # two columns, one under each plot
    axt.text(0.0, 0.92, "\n".join(lines[:4]), va="top", ha="left", fontsize=6.2, linespacing=1.55)
    axt.text((W - plot_w - 0.58) / (W - 0.58), 0.92, "\n".join(lines[4:]), va="top", ha="left",
             fontsize=6.2, linespacing=1.55)
    ctx.out.save(fig, "fig_uav_scenario", [S], notes={
        "proposed_best": {"run": r["run"], "seed": r["seed"], "cost": r["f"],
                      "J1_length": J[0], "J2_threat": J[1], "J3_altitude": J[2], "J4_smooth": J[3],
                      "nodes_xy": np.c_[xa, ya], "absolute_altitude": zabs,
                      "min_ground_clearance": clearance, "feasible_runs": f"{feas}/{S.n_runs}"},
        "competitor_path": None if comp is None else {"algorithm": comp[1], "cost": comp[0],
                                                      "run": comp[2]["run"]},
        "terrain": f"contour lines of applications/uav/data/ChrismasTerrain.tif, every {step}th cell",
        "units": "map units of the original model (Phung & Ha 2021); no physical unit is asserted",
        "uav_symbol": "representative symbol, not to scale",
        "source": "Phung & Ha (2021), doi:10.1016/j.asoc.2021.107376",
    }, extra_sources=[os.path.join(fl.ROOT, "applications", "uav", "path_planning.py"),
                      os.path.join(fl.ROOT, "applications", "uav", "data", "ChrismasTerrain.tif")])


ENG = [("composite", "Laminate, unconstrained"), ("composite_contiguity", "Laminate, contiguity"),
       ("uav", "UAV path planning")]


def engineering_convergence(ctx):
    suites = [ctx.load(s) for s, _ in ENG]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.35), layout="constrained")
    notes = {}
    for ax, S, (_, title) in zip(axes, suites, ENG):
        fid = S.name
        ref = rx.load_objective(fid, fid, None)["reference"]
        n = S.settings["agents"]
        for a in ALGORITHMS:
            grid = fl.fe_grid(n, int(S.fes(fid, a).max()))
            med = finite_median(S.curves(fid, a, grid))
            if ref is not None:
                med = med + ref                     # f = -lambda_cb (+ penalty): gap to the exact optimum
            ax.plot(grid, med, **fl.line_style(a))
            ax.plot(grid[-1], med[-1], ls="none", marker=STYLE[a]["marker"], color=STYLE[a]["color"], ms=3)
        ax.set_xscale("log")
        if ref is not None:
            ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(r"Median gap to optimum, $\lambda^* - \lambda_{cb}$")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("Median cost $F$")
        ax.set_title(title, pad=2)
        ax.set_xlabel("Function evaluations")
        ax.grid(True, which="major")
        notes[fid] = {"reference_optimum": ref,
                      "feasible_runs": {a: f"{int(np.isfinite(S.final(fid, a)).sum())}/{S.n_runs}"
                                        for a in ALGORITHMS}}
    notes["curve"] = "median over the runs of the best value evaluated so far; a curve is drawn only where " \
                     "more than half of the runs have a finite value (infeasible = infinite cost)"
    notes["composite_contiguity"]["penalty"] = "the gap includes the static penalty of infeasible designs"
    fig.legend(handles=algo_handles(), loc="outside lower center", ncol=9, frameon=False)
    ctx.out.save(fig, "fig_engineering_convergence", suites, notes=notes)


def engineering_boxplots(ctx):
    suites = [ctx.load(s) for s, _ in ENG]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5), layout="constrained")
    notes = {}
    for ax, S, (_, title) in zip(axes, suites, ENG):
        fid = S.name
        maximize = fid.startswith("composite")
        data = [(-S.final(fid, a) if maximize else S.final(fid, a)) for a in ALGORITHMS]
        draw_box(ax, data, log=not maximize)
        feas = [int(np.isfinite(d).sum()) for d in data]
        ax.set_xticklabels([f"{display(a)} ({k}/{S.n_runs})" if k < S.n_runs else display(a)
                            for a, k in zip(ALGORITHMS, feas)], rotation=90)
        ax.set_title(title, pad=2)
        ax.set_ylabel(r"$\lambda_{cb}$ (penalised objective)" if fid == "composite_contiguity"
                      else (r"$\lambda_{cb}$" if maximize else "Cost $F$"))
        ref = rx.load_objective(fid, fid, None)["reference"]
        if ref is not None:
            ax.axhline(ref, color="black", lw=0.5, ls=(0, (4, 2)))
        notes[fid] = {"feasible_runs": dict(zip(ALGORITHMS, [f"{k}/{S.n_runs}" for k in feas])),
                      "reference_optimum": ref}
    notes["boxes"] = "final values of the runs; runs without a feasible solution (infinite cost) are not in " \
                     "the box and are reported as (feasible/total) in the tick label"
    notes["dashed_line"] = "exact global optimum (exhaustive search), composite panels"
    ctx.out.save(fig, "fig_engineering_boxplots", suites, notes=notes)


# ---------------------------------------------------------------------------
# Behaviour of RTI-DE (summary of the per-generation trace stored with its runs)
# ---------------------------------------------------------------------------

STAGES = [("First trial point", "#4477AA", "-"), ("Half step", "#228833", "--"),
          ("Parabola point", "#AA3377", "-.")]


def _rate(num, den):
    """num / den, nan where den = 0."""
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _line(ax, x, y, **style):
    """Line through the finite values of y (a part of the budget without a generation has none)."""
    ok = np.isfinite(y)
    return ax.plot(x[ok], y[ok], **style)


def _quantiles(a):
    """25 %, 50 % and 75 % quantile of each column; nan entries (runs that have stopped) are left out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanpercentile(a, [25, 50, 75], axis=0)


def behaviour(ctx, suite):
    import inspect
    from RTIDE import rti_de
    S = ctx.load(suite)
    runs = [(fid, r) for fid in S.fids for r in S.runs[fid][PROPOSED]]
    missing = sum("rtide_trace" not in r for _, r in runs)
    if missing:
        sys.exit(f"ERROR: {S.dir}: {missing} of {len(runs)} runs of {PROPOSED} have no 'rtide_trace'; "
                 f"rerun them with run_experiments.py.")
    T = [r["rtide_trace"] for _, r in runs]
    edges, th_edges = T[0]["fe_edges"], T[0]["theta_edges"]
    if any(not np.array_equal(t["fe_edges"], edges) or not np.array_equal(t["theta_edges"], th_edges) for t in T):
        sys.exit(f"ERROR: {S.dir}: the traces of {PROPOSED} do not share one grid.")
    x = 0.5 * (edges[:-1] + edges[1:])
    total = lambda key: np.sum([t[key] for t in T], axis=0).astype(float)
    ev, acc, imp = total("evaluated"), total("accepted"), total("improved")
    th_acc, th_rej = total("theta_accepted"), total("theta_rejected")
    rho = inspect.signature(rti_de).parameters["chain_from"].default

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5), layout="constrained")
    (a_share, a_rate), (a_ctrl, a_theta) = axes

    share = _rate(ev, ev.sum(axis=1, keepdims=True))
    for j, (label, col, ls) in enumerate(STAGES):
        _line(a_share, x, share[:, j], color=col, ls=ls, lw=1.1, label=label)
    a_share.set_ylim(0, 1.02)
    a_share.set_ylabel("Share of the evaluations")
    a_share.set_title("(a) Candidates by stage", pad=3)
    a_share.legend(frameon=False, loc="center left")

    for j, (label, col, ls) in enumerate(STAGES):
        _line(a_rate, x, _rate(acc[:, j], ev[:, j]), color=col, ls="-", lw=1.1)
        _line(a_rate, x, _rate(imp[:, j], ev[:, j]), color=col, ls=":", lw=0.9)
    a_rate.set_yscale("log")
    a_rate.set_ylabel("Fraction of the candidates of the stage")
    a_rate.set_title("(b) Accepted and strictly better candidates", pad=3)
    a_rate.legend(handles=[Line2D([0], [0], color="black", ls="-", lw=1.1, label=r"accepted ($f \leq f_i$)"),
                           Line2D([0], [0], color="black", ls=":", lw=0.9, label=r"strictly better ($f < f_i$)")],
                  frameon=False, loc="best")

    q_mu = _quantiles(np.array([t["mu_F"] for t in T]))
    q_n = _quantiles(np.array([t["n"] for t in T]))
    ok = np.isfinite(q_mu[1])
    a_ctrl.fill_between(x[ok], q_mu[0][ok], q_mu[2][ok], color=STYLE[PROPOSED]["color"], alpha=0.25, lw=0)
    _line(a_ctrl, x, q_mu[1], color=STYLE[PROPOSED]["color"], lw=1.2)
    a_ctrl.set_ylabel(r"Mean step factor $\mu_F$ (median, quartiles)", color=STYLE[PROPOSED]["color"])
    a_ctrl.set_ylim(0, 1)
    a_pop = a_ctrl.twinx()
    _line(a_pop, x, q_n[1], color="black", ls="--", lw=0.9)
    a_pop.set_ylabel("Population size $N$ (median)")
    a_pop.set_ylim(bottom=0)
    a_ctrl.set_title(r"(c) Mean step factor and population size", pad=3)

    centres, width = 0.5 * (th_edges[:-1] + th_edges[1:]), np.diff(th_edges)
    a_theta.bar(centres, th_rej, width=width, color="#B0B0B0", lw=0, label="rejected")
    a_theta.bar(centres, th_acc, width=width, bottom=th_rej, color=STAGES[2][1], lw=0, label="accepted")
    a_theta.axvline(0.0, color="black", lw=0.5)
    a_theta.set_xlabel(r"Factor $\theta^*$ of the parabola point")
    a_theta.set_ylabel("Number of third points")
    a_theta.set_title("(d) Third points of the chain", pad=3)
    a_theta.legend(frameon=False, loc="upper left")
    a_theta.grid(True, axis="y", which="major")

    for ax in (a_share, a_rate, a_ctrl):
        ax.axvline(rho * S.budget, color="black", lw=0.5, ls=(0, (4, 2)))
        ax.set_xlim(0, S.budget)
        ax.set_xlabel("Function evaluations")
        ax.grid(True, which="major")

    per_function = {}
    for fid in S.fids:
        tt = [r["rtide_trace"] for r in S.runs[fid][PROPOSED]]
        e, a, i = (np.sum([t[k] for t in tt], axis=(0, 1)) for k in ("evaluated", "accepted", "improved"))
        per_function[fid] = {"evaluated": e, "accepted": a, "strictly_better": i,
                             "skipped_third_points": int(np.sum([t["skipped"] for t in tt])),
                             "runs_stopped_at_target": int(sum(r["stopped_at_target"] for r in S.runs[fid][PROPOSED]))}
    ctx.out.save(fig, f"fig_behaviour_{suite}", [S], notes={
        "data": f"per-generation trace of {PROPOSED} (run_experiments.summarize_rtide_trace), summed over "
                f"{len(runs)} runs ({len(S.fids)} functions x {S.n_runs} runs); x axis in {len(x)} equal parts "
                "of the budget; a run that stopped at the target contributes nothing after it stopped",
        "panel_a": "share of the candidates evaluated in each part that were first trial points, half steps "
                   "and parabola points (pooled over functions and runs)",
        "panel_b": "per stage, candidates accepted (solid) and strictly better than their agent (dotted), "
                   "divided by the candidates of that stage evaluated in the part (pooled)",
        "panel_c": "mu_F after the last generation of each part: median and quartiles over the runs; "
                   "population size: median over the runs",
        "panel_d": "factor of every third point evaluated, accepted and rejected (pooled); third points left "
                   "out by the skip rule are not evaluated and not shown",
        "dashed_line": f"chain_from = {rho:g} of the budget ({rho * S.budget:g} function evaluations)",
        "order_of_columns": [s[0] for s in STAGES],
        "totals": {"evaluated": ev.sum(axis=0), "accepted": acc.sum(axis=0), "strictly_better": imp.sum(axis=0),
                   "skipped_third_points": int(np.sum([t["skipped"] for t in T])),
                   "third_points_accepted": int(th_acc.sum()), "third_points_rejected": int(th_rej.sum()),
                   "third_points_outside_the_histogram": int(sum(t["theta_outside"] for t in T))},
        "per_function": per_function,
    })


# ---------------------------------------------------------------------------
# Registry and command line
# ---------------------------------------------------------------------------

def _cec2017_all(fn, prefix):
    def run(ctx):
        S = ctx.load("cec2017")
        for i, part in enumerate(chunks(S.fids, 12), 1):
            fn(ctx, "cec2017", part, f"{prefix}_{i}")
    return run


FIGURES = {
    "conv_cec2022": (lambda c: convergence(c, "cec2022", c.load("cec2022").fids, "fig_conv_cec2022"),
                     "CEC2022 convergence against FEs, 4x3"),
    "conv_cec2017_main": (lambda c: convergence(c, "cec2017", fl.CEC2017_MAIN, "fig_conv_cec2017_main"),
                          "CEC2017 convergence, 12 representative functions, 4x3"),
    "conv_cec2017_all": (_cec2017_all(convergence, "fig_conv_cec2017_all"),
                         "CEC2017 convergence, all 29 functions (3 appendix figures)"),
    "box_cec2022": (lambda c: boxplots(c, "cec2022", c.load("cec2022").fids, "fig_box_cec2022"),
                    "CEC2022 box plots of the final errors, 4x3"),
    "box_cec2017_main": (lambda c: boxplots(c, "cec2017", fl.CEC2017_MAIN, "fig_box_cec2017_main"),
                         "CEC2017 box plots, 12 representative functions, 4x3"),
    "box_cec2017_all": (_cec2017_all(boxplots, "fig_box_cec2017_all"),
                        "CEC2017 box plots, all 29 functions (3 appendix figures)"),
    "cd_cec2022": (lambda c: critical_difference(c, "cec2022"), "CEC2022 critical difference diagram"),
    "cd_cec2017": (lambda c: critical_difference(c, "cec2017"), "CEC2017 critical difference diagram"),
    "ecdf_cec2022": (lambda c: ecdf(c, "cec2022"), "CEC2022 ECDF of FEs to reach the targets"),
    "ecdf_cec2017": (lambda c: ecdf(c, "cec2017"), "CEC2017 ECDF of FEs to reach the targets"),
    "behaviour_cec2022": (lambda c: behaviour(c, "cec2022"),
                          "CEC2022 behaviour of RTI-DE: chain points, acceptance, mu_F and N, factors of the third points"),
    "behaviour_cec2017": (lambda c: behaviour(c, "cec2017"),
                          "CEC2017 behaviour of RTI-DE: chain points, acceptance, mu_F and N, factors of the third points"),
    "landscape_cec2022": (landscapes, "CEC2022 landscapes at D = 2, 3x3 (F1-F5, F9-F12)"),
    "composite": (composite_figure, "Composite laminate: plate, properties and best layups"),
    "uav": (uav_figure, "UAV: terrain, threats, best path, elevation profile"),
    "eng_convergence": (engineering_convergence, "Engineering problems: convergence against FEs"),
    "eng_box": (engineering_boxplots, "Engineering problems: box plots"),
}


class Context:
    def __init__(self, args):
        self.args = args
        self.out = fl.Output(args.out, args.dry_run)

    def load(self, suite):
        return fl.load(suite, self.args.results)


def main():
    ap = argparse.ArgumentParser(description="Vector figures for the RTI-DE paper.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="figures to build (default: all); see --list")
    ap.add_argument("--list", action="store_true", help="list the figure names and exit")
    ap.add_argument("--results", default=os.path.join(fl.ROOT, "results_cache", "paper"),
                    help="directory written by run_experiments.py (default: results_cache/paper)")
    ap.add_argument("--out", default=os.path.join(fl.ROOT, "outputs", "figures"),
                    help="output directory (default: outputs/figures)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the data and build every figure in memory, but write no file")
    args = ap.parse_args()

    if args.list:
        for name, (_, text) in FIGURES.items():
            print(f"  {name:<22} {text}")
        return
    names = args.only or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f"ERROR: unknown figure name(s) {unknown}; use --list.")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    fl.apply_style()
    ctx = Context(args)
    print(f"RTI-DE paper figures -> {args.out}{'  (dry run: nothing is written)' if args.dry_run else ''}")
    t0 = time.time()
    for name in names:
        print(f"  [{name}] {FIGURES[name][1]}")
        FIGURES[name][0](ctx)
    print(f"Done in {time.time() - t0:.0f} s: {len(names)} figure group(s), "
          f"{len(ctx.out.written)} figure(s) written.")


if __name__ == "__main__":
    main()
