"""
Composite laminate stacking-sequence design for maximum buckling load.

Benchmark of Le Riche & Haftka (1993), in the 64-ply form used by
Ahmid, Dao & Le (2019): a simply supported rectangular graphite-epoxy plate
under biaxial in-plane compression. The design variables are the orientations
of the 16 two-ply stacks in one half of a symmetric, balanced laminate. The
objective is to maximize the critical buckling load factor λ_cb. The problem
data (material properties, ply thickness, 64 plies, a = 20 in, b = 10 in,
Ny/Nx = 1) agree with Le Riche (1994), Ch. 3 (PhD dissertation, Virginia Tech,
hdl:10919/38270).

Model (classical lamination theory, D16 = D26 neglected as in the references):

    λ_b(p, q) = π² [D11 (p/a)⁴ + 2 (D12 + 2 D66) (p/a)² (q/b)² + D22 (q/b)⁴]
                / [(p/a)² Nx + (q/b)² Ny]
    λ_cb = min over p, q ≥ 1 of λ_b(p, q)

Units follow the original imperial definition (psi, in, lb/in). With these
units the model reproduces all four stacking sequences and λ_cb values
published by Ahmid et al. (2019), Table 3: 3973.01 (three sequences) and
3972.50 (one sequence). See ``tests/test_composite.py``.

Sources
-------
R. Le Riche and R. T. Haftka, "Optimization of laminate stacking sequence for
buckling load maximization by genetic algorithm," AIAA Journal, 31(5),
951–956, 1993. doi:10.2514/3.11710
R. Le Riche, "Optimization of composite structures by genetic algorithms,"
PhD dissertation, Virginia Polytechnic Institute and State University,
Blacksburg, VA, 1994, Ch. 3. hdl:10919/38270
A. Ahmid, T.-M. Dao and V. N. Le, "Comparison study of discrete optimization
problem using meta-heuristic approaches: A case study," International Journal
of Industrial Engineering and Operations Management, 1(2), 97–109, 2019.
doi:10.46254/j.ieom.20190201
"""

import numpy as np

# ---------------------------------------------------------------------------
# Problem data (Le Riche & Haftka 1993; SI values in Ahmid et al. 2019, Tables 1–2;
# the values agree with Le Riche 1994, Ch. 3, Sec. 3.1.2 and Table 3.2)
# ---------------------------------------------------------------------------

E1 = 18.5e6        # psi   (127.59 GPa)
E2 = 1.89e6        # psi   (13.03 GPa)
G12 = 0.93e6       # psi   (6.41 GPa)
NU12 = 0.3
PLY_T = 0.005      # in    (0.127 mm)
N_PLIES = 64
A_LEN = 20.0       # in    (508 mm)
B_LEN = 10.0       # in    (a/b = 2)
NX = 1.0           # lb/in
NY = 1.0           # lb/in (Nx/Ny = 1)

# Each design variable is one two-ply stack: 0 → [0, 0], 1 → [+45, −45], 2 → [90, 90]
STACKS = {0: (0.0, 0.0), 1: (45.0, -45.0), 2: (90.0, 90.0)}
STACK_LABELS = {0: "0_2", 1: "±45", 2: "90_2"}
N_STACKS = N_PLIES // 4          # stacks in one half of the symmetric laminate
DIM = N_STACKS                   # 16 design variables
LB = np.zeros(DIM)
UB = np.full(DIM, 3.0)           # continuous x in [0, 3) → gene floor(x) ∈ {0, 1, 2}

MODES = np.arange(1, 11)         # half-wave numbers p, q = 1..10

# Published reference values used for validation (Ahmid et al. 2019, Table 3)
PUBLISHED = (
    ([2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1], 3973.01),
    ([2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1], 3973.01),
    ([2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2], 3973.01),
    ([2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1], 3972.50),
)


# ---------------------------------------------------------------------------
# Laminate mechanics
# ---------------------------------------------------------------------------

def _qbar(theta_deg):
    """Transformed reduced stiffnesses (Q̄11, Q̄12, Q̄22, Q̄66) of one ply."""
    nu21 = NU12 * E2 / E1
    den = 1.0 - NU12 * nu21
    q11, q22, q12, q66 = E1 / den, E2 / den, NU12 * E2 / den, G12
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        q11 * c**4 + 2 * (q12 + 2 * q66) * s**2 * c**2 + q22 * s**4,
        (q11 + q22 - 4 * q66) * s**2 * c**2 + q12 * (s**4 + c**4),
        q11 * s**4 + 2 * (q12 + 2 * q66) * s**2 * c**2 + q22 * c**4,
        (q11 + q22 - 2 * q12 - 2 * q66) * s**2 * c**2 + q66 * (s**4 + c**4),
    ])


def _stack_contributions():
    """C[k, g, :] = contribution of gene g at stack position k (outermost = 0)
    to (D11, D12, D22, D66) of the full symmetric laminate."""
    h = N_PLIES * PLY_T
    C = np.zeros((N_STACKS, 3, 4))
    for k in range(N_STACKS):
        for g, (a1, a2) in STACKS.items():
            for j, ang in enumerate((a1, a2)):
                ply = 2 * k + j
                z_top = h / 2 - ply * PLY_T
                z_bot = z_top - PLY_T
                # factor 2: the mirrored ply below the mid-plane contributes equally
                C[k, g] += 2.0 * _qbar(ang) * (z_top**3 - z_bot**3) / 3.0
    return C


_C = _stack_contributions()
_PA2 = (MODES[:, None] / A_LEN) ** 2          # (p/a)^2, shape (P, 1)
_QB2 = (MODES[None, :] / B_LEN) ** 2          # (q/b)^2, shape (1, Q)


def bending_stiffness(genes):
    """(D11, D12, D22, D66) for one gene vector or a (n, 16) array of them."""
    genes = np.asarray(genes, dtype=int)
    single = genes.ndim == 1
    G = np.atleast_2d(genes)
    D = _C[np.arange(N_STACKS)[None, :], G].sum(axis=1)   # (n, 4)
    return D[0] if single else D


def buckling_load_factor(genes):
    """Critical buckling load factor λ_cb for one or many gene vectors."""
    D = np.atleast_2d(bending_stiffness(genes))
    d11, d12, d22, d66 = (D[:, i, None, None] for i in range(4))
    num = np.pi**2 * (d11 * _PA2**2 + 2 * (d12 + 2 * d66) * _PA2 * _QB2 + d22 * _QB2**2)
    lam = (num / (_PA2 * NX + _QB2 * NY)).min(axis=(1, 2))
    return lam[0] if np.asarray(genes).ndim == 1 else lam


# ---------------------------------------------------------------------------
# Contiguity constraint (optional): at most 4 contiguous plies of the same
# orientation. With two-ply stacks this means no run of more than 2 identical
# 0_2 or 90_2 stacks, counted across the mid-plane of the symmetric laminate
# (the innermost stack is mirrored). ±45 stacks alternate +45/−45 and never
# form contiguous identical plies.
# ---------------------------------------------------------------------------

MAX_CONTIGUOUS_STACKS = 2


def contiguity_violation(genes):
    """Number of stacks exceeding the contiguity limit (0 = feasible).

    Rows of a (n, 16) array are handled independently.
    """
    G = np.atleast_2d(np.asarray(genes, dtype=int))
    full = np.concatenate([G, G[:, ::-1]], axis=1)       # outer → mid-plane → outer
    viol = np.zeros(G.shape[0], dtype=int)
    run = np.ones(G.shape[0], dtype=int)
    for j in range(1, full.shape[1]):
        same = (full[:, j] == full[:, j - 1]) & (full[:, j] != 1)
        run = np.where(same, run + 1, 1)
        viol += (same & (run > MAX_CONTIGUOUS_STACKS)).astype(int)
    return viol[0] if np.asarray(genes).ndim == 1 else viol


# ---------------------------------------------------------------------------
# Continuous interface for the optimizers (minimization)
# ---------------------------------------------------------------------------

def decode(x):
    """Continuous x ∈ [0, 3]^16 → integer genes {0, 1, 2}^16 (floor, x = 3 → 2)."""
    return np.clip(np.floor(np.asarray(x)), 0, 2).astype(int)


PENALTY = 1000.0   # λ units per violating stack (only used with contiguity=True)


def objective(x, contiguity=False):
    """Minimization objective: −λ_cb (+ PENALTY × violations if contiguity=True).

    Accepts one vector (returns float) or an (n, 16) array (returns (n,)).
    """
    genes = decode(x)
    f = -buckling_load_factor(genes)
    if contiguity:
        f = f + PENALTY * contiguity_violation(genes)
    return f


def describe(genes):
    """Human-readable stacking sequence, e.g. '[90_2/90_2/±45/...]_s'."""
    return "[" + "/".join(STACK_LABELS[int(g)] for g in genes) + "]_s"
