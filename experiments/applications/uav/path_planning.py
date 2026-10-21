"""
3-D UAV path planning over real terrain with cylindrical threats.

Python port of the problem defined by Phung & Ha (2021) in their public
MATLAB code (github.com/duongpm/SPSO: CreateModel.m, MyCost.m,
SphericalToCart.m, DistP2S.m, SPSO_MAIN.m). The port reproduces the MATLAB
logic line by line, including:

* MATLAB ``round`` (half away from zero) when reading terrain heights, and
  1-based terrain indexing;
* clipping of every node to the map/altitude limits inside the spherical →
  Cartesian conversion;
* infinite cost for collisions (threat or ground), as in the original.

Decision vector (dimension 3n, n = 10 path nodes):
    x = [r_1..r_n, psi_1..psi_n, phi_1..phi_n]
    r   ∈ [0, 2‖start − end‖ / n]       (segment length)
    psi ∈ [−π/4, π/4]                    (climb/dive angle)
    phi ∈ [phi0 − π/4, phi0 + π/4]       (heading), phi0 = atan2(dy, dx)

Note on the original code: phi0 is computed with atan2(dy, dx) (angle from
the x-axis), while SphericalToCart uses x = r cos(psi) sin(phi) and
y = r cos(psi) cos(phi) (angle from the y-axis). The port keeps this as is.

Cost (minimize): F = 5 J1 + 1 J2 + 10 J3 + 1 J4
    J1 path length, J2 threat cost, J3 altitude cost, J4 smoothness cost.

Scenario: the single scenario in the authors' CreateModel.m (Christmas Island
terrain, six cylindrical threats, start (200, 100, 150), end (800, 800, 150)).
The paper reports eight scenarios; the repository contains only this one, and
which paper scenario (if any) it corresponds to is not documented.

Source
------
M. D. Phung and Q. P. Ha, "Safety-enhanced UAV path planning with spherical
vector-based particle swarm optimization," Applied Soft Computing, 107,
107376, 2021. doi:10.1016/j.asoc.2021.107376
Terrain data and original code: BSD-style license, see data/LICENSE_SPSO.txt.
"""

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ChrismasTerrain.tif")


def _load_terrain(path=_DATA):
    from PIL import Image
    H = np.array(Image.open(path), dtype=np.float64)
    H[H < 0] = 0.0                       # CreateModel.m: H(H < 0) = 0
    return H


# ---------------------------------------------------------------------------
# Scenario (CreateModel.m)
# ---------------------------------------------------------------------------

H = _load_terrain()
MAPSIZE_Y, MAPSIZE_X = H.shape           # rows = y, columns = x
XMIN, XMAX = 1.0, float(MAPSIZE_X)
YMIN, YMAX = 1.0, float(MAPSIZE_Y)
ZMIN, ZMAX = 100.0, 200.0                # height above ground

# Threats as vertical cylinders: (x, y, z, R)
THREATS = np.array([
    [400.0, 500.0, 100.0, 80.0],
    [600.0, 200.0, 150.0, 70.0],
    [500.0, 350.0, 150.0, 80.0],
    [350.0, 200.0, 150.0, 70.0],
    [700.0, 550.0, 150.0, 70.0],
    [650.0, 750.0, 150.0, 80.0],
])

START = np.array([200.0, 100.0, 150.0])
END = np.array([800.0, 800.0, 150.0])
N_NODES = 10

# Cost constants (MyCost.m)
DRONE_SIZE = 1.0
DANGER_DIST = 10.0 * DRONE_SIZE
TURNING_MAX = 45.0                       # degrees
CLIMB_MAX = 45.0                         # degrees
B1, B2, B3, B4 = 5.0, 1.0, 10.0, 1.0

# Search space (SPSO_MAIN.m)
ANGLE_RANGE = np.pi / 4
_dir = END - START
PHI0 = np.arctan2(_dir[1], _dir[0])
R_MAX = 2.0 * np.linalg.norm(START - END) / N_NODES
DIM = 3 * N_NODES
LB = np.concatenate([np.zeros(N_NODES),
                     np.full(N_NODES, -ANGLE_RANGE),
                     np.full(N_NODES, PHI0 - ANGLE_RANGE)])
UB = np.concatenate([np.full(N_NODES, R_MAX),
                     np.full(N_NODES, ANGLE_RANGE),
                     np.full(N_NODES, PHI0 + ANGLE_RANGE)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matlab_round(v):
    """MATLAB round: halves are rounded away from zero."""
    v = np.asarray(v, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def terrain_height(x, y):
    """H(round(y), round(x)) with MATLAB's 1-based indexing."""
    r = matlab_round(y).astype(int) - 1
    c = matlab_round(x).astype(int) - 1
    return H[r, c]


def dist_point_segment(p, a, b):
    """DistP2S.m: distance from point p to segment ab in the plane."""
    p, a, b = (np.asarray(v, dtype=np.float64) for v in (p, a, b))
    d_ab = np.linalg.norm(a - b)
    d_ax = np.linalg.norm(a - p)
    d_bx = np.linalg.norm(b - p)
    if d_ab != 0:
        if np.dot(a - b, p - b) * np.dot(b - a, p - a) >= 0:
            det = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            return abs(det) / d_ab
        return min(d_ax, d_bx)
    return d_ax


def spherical_to_cart(x):
    """SphericalToCart.m: decision vector → node coordinates (x, y, z), each clipped."""
    x = np.asarray(x, dtype=np.float64)
    r, psi, phi = x[:N_NODES], x[N_NODES:2 * N_NODES], x[2 * N_NODES:]
    px, py, pz = np.empty(N_NODES), np.empty(N_NODES), np.empty(N_NODES)
    cx, cy, cz = START
    for i in range(N_NODES):
        cx = min(max(cx + r[i] * np.cos(psi[i]) * np.sin(phi[i]), XMIN), XMAX)
        cy = min(max(cy + r[i] * np.cos(psi[i]) * np.cos(phi[i]), YMIN), YMAX)
        cz = min(max(cz + r[i] * np.sin(psi[i]), ZMIN), ZMAX)
        px[i], py[i], pz[i] = cx, cy, cz
    return px, py, pz


def _atan2d(y, x):
    return np.degrees(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# Cost (MyCost.m)
# ---------------------------------------------------------------------------

def cost_components(x):
    """Return (J1, J2, J3, J4) for one decision vector.

    On a collision the remaining components are not computed and are
    returned as inf (the overall cost is inf either way, as in MyCost.m).
    """
    if not np.all(np.isfinite(x)):
        raise ValueError("Decision vector contains NaN or inf.")
    nx, ny, nz = spherical_to_cart(x)
    xa = np.concatenate([[START[0]], nx, [END[0]]])
    ya = np.concatenate([[START[1]], ny, [END[1]]])
    za = np.concatenate([[START[2]], nz, [END[2]]])
    N = xa.size
    z_abs = za + terrain_height(xa, ya)

    # J1: path length
    J1 = float(np.sum(np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2 + np.diff(z_abs) ** 2)))

    # J2: threats (projected segments vs. cylinder axes)
    J2 = 0.0
    for tx, ty, _, tr in THREATS:
        for j in range(N - 1):
            d = dist_point_segment((tx, ty), (xa[j], ya[j]), (xa[j + 1], ya[j + 1]))
            if d > tr + DRONE_SIZE + DANGER_DIST:
                continue
            if d < tr + DRONE_SIZE:
                return J1, np.inf, np.inf, np.inf
            J2 += (tr + DRONE_SIZE + DANGER_DIST) - d

    # J3: altitude (node heights above ground; z < 0 would be a crash)
    if np.any(nz < 0):
        return J1, J2, np.inf, np.inf
    J3 = float(np.sum(np.abs(nz - (ZMAX + ZMIN) / 2.0)))

    # J4: smoothness (turning and climb-angle changes)
    J4 = 0.0
    for i in range(N - 2):
        seg1 = np.zeros(3)
        for j in range(i, -1, -1):
            seg1 = np.array([xa[j + 1] - xa[j], ya[j + 1] - ya[j], 0.0])
            if np.count_nonzero(seg1) != 0:
                break
        seg2 = np.zeros(3)
        for j in range(i, N - 2):
            seg2 = np.array([xa[j + 2] - xa[j + 1], ya[j + 2] - ya[j + 1], 0.0])
            if np.count_nonzero(seg2) != 0:
                break
        climb1 = _atan2d(z_abs[i + 1] - z_abs[i], np.linalg.norm(seg1))
        climb2 = _atan2d(z_abs[i + 2] - z_abs[i + 1], np.linalg.norm(seg2))
        turning = _atan2d(np.linalg.norm(np.cross(seg1, seg2)), np.dot(seg1, seg2))
        if abs(turning) > TURNING_MAX:
            J4 += abs(turning)
        if abs(climb2 - climb1) > CLIMB_MAX:
            J4 += abs(climb2 - climb1)
    return J1, J2, J3, J4


def cost(x):
    """Overall cost F = 5 J1 + J2 + 10 J3 + J4 (inf for infeasible paths)."""
    J1, J2, J3, J4 = cost_components(x)
    if not np.isfinite(J2) or not np.isfinite(J3):
        return np.inf
    return B1 * J1 + B2 * J2 + B3 * J3 + B4 * J4


def objective(X):
    """Vectorized wrapper: (n, DIM) → (n,), or one vector → float."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return cost(X)
    return np.array([cost(row) for row in X])
