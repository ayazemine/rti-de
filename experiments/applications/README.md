# Engineering design test problems

Two engineering problems of the RTI-DE paper, both taken from the literature and ported so that they reproduce the original definitions.

| Problem | Module | D | Type | Reference solution |
|---|---|---|---|---|
| Composite laminate, max buckling load | `composite/laminate_buckling.py` | 16 | discrete (continuous → floor), maximize | exact global optimum by exhaustive search |
| Same, with ply-contiguity constraint | same, `contiguity=True` | 16 | discrete, constrained (static penalty) | exact global optimum by exhaustive search |
| UAV 3-D path planning | `uav/path_planning.py` | 30 | continuous, infeasible = ∞, minimize | none (no exact optimum known) |

## Running

From `experiments/`:

```bash
python -W ignore run_experiments.py --suites composite composite_contiguity uav
python -m unittest discover -s tests -v
```

The runs are made by `run_experiments.py` (RTI-DE, DE, L-SHADE, RIME, PSO, GWO, WOA); `applications/run_engineering.py` holds the problem registry and the summary rows. Each suite gets `config.json` (settings, code hashes, package versions), one file per run and the summary tables in `results_cache/paper/<problem>/`. All algorithms use the same seeds (`seed = seed_base + run`). Every algorithm has the same budget of function evaluations, N (T + 1) with `--agents` N and `--iterations` T (default 50 and 1000, i.e. 50,050).

## 1. Composite laminate stacking sequence (buckling)

**Problem.** A simply supported rectangular graphite-epoxy plate under biaxial compression. The laminate is symmetric and balanced with 64 plies, built from two-ply stacks {0₂, ±45, 90₂}. The 16 stack orientations of one half are the design variables. The objective is to maximize the critical buckling load factor λ_cb (classical lamination theory, D16 = D26 neglected, modes p, q = 1..10).

**Data** (Le Riche & Haftka 1993; SI values in Ahmid et al. 2019, Tables 1–2):
- E1 = 18.5 Msi (127.59 GPa), E2 = 1.89 Msi (13.03 GPa), G12 = 0.93 Msi (6.41 GPa), ν12 = 0.3
- ply thickness 0.005 in (0.127 mm); a = 20 in (508 mm), a/b = 2; Nx = Ny = 1 lb/in

**Validation.** The model reproduces all four stacking sequences and λ_cb values in Ahmid et al. (2019), Table 3: 3973.01 (three sequences) and 3972.50 (one) — see `tests/test_composite.py`. The imperial units are needed for this: Ahmid et al.'s SI table prints b = 25.4 mm (inconsistent with the stated a/b = 2), and the SI conversion alone gives 3976.64.

**Reference optimum.** `composite/exhaustive_search.py` enumerates all 3¹⁶ = 43,046,721 designs (`reference_optimum.json`):
- without contiguity: λ_cb = 3973.0137, reached by 7 designs. This confirms that the published 3973.01 is the global optimum.
- with contiguity (≤ 4 contiguous identical plies, counted across the mid-plane): λ_cb = 3953.2830, reached by 1 design. No published value for this exact constraint definition was found.

**Encoding for continuous optimizers.** x ∈ [0, 3]¹⁶ is mapped to stack ids by floor (x = 3 → 2). The contiguity variant adds 1000 × (number of violating stacks) to −λ_cb (`PENALTY` in `laminate_buckling.py`).

## 2. UAV 3-D path planning

**Problem.** A Python port of Phung & Ha (2021), from the authors' public MATLAB code (github.com/duongpm/SPSO: `CreateModel.m`, `MyCost.m`, `SphericalToCart.m`, `DistP2S.m`, `SPSO_MAIN.m`). The terrain is a real elevation map of Christmas Island (`uav/data/ChrismasTerrain.tif`, 1045 × 879). There are six cylindrical threats; start (200, 100, 150), end (800, 800, 150), 10 path nodes.

- Decision vector: 30 values (r, ψ, φ for each node): r ∈ [0, 2‖start − end‖/10], ψ ∈ [−π/4, π/4], φ ∈ [φ₀ − π/4, φ₀ + π/4].
- Cost: F = 5·J1 (length) + J2 (threat proximity) + 10·J3 (altitude) + J4 (turning/climb smoothness), with F = ∞ on collision.

**Porting details kept from the original:**
- MATLAB `round` rounds halves away from zero, and indexing is 1-based.
- Every node is clipped to the map/altitude limits.
- φ₀ = atan2(dy, dx) is measured from the x-axis, while the conversion uses x = r cos ψ sin φ (angle from the y-axis). This inconsistency exists in the original code and is kept as is.

**Validation.** Component-level unit tests (`tests/test_uav.py`: rounding, point–segment distance, clipping, zero-length path length, collision → ∞). An exact numerical comparison with the paper is **not possible**: the paper reports eight scenarios, the repository contains only this one, and which scenario it corresponds to is not documented. Mean costs in the paper are about 4.6k–8.1k for n = 10 nodes (Table 1); costs obtained with this port are of the same order of magnitude.

**Difficulty.** Only 67 of 20,000 uniformly random paths are collision-free, so the initial population is usually all-infinite.

## Non-finite values

- **Non-finite candidates and infinite costs:** the runner evaluates any non-finite candidate as infeasible (f = +∞) for **all** algorithms and reports the count per algorithm (`mean_invalid_candidates`). RTI-DE, DE and L-SHADE treat a NaN objective value as +∞ (`tests/test_rtide.py`).

## Sources

- R. Le Riche, R. T. Haftka (1993). Optimization of laminate stacking sequence for buckling load maximization by genetic algorithm. *AIAA Journal*, 31(5), 951–956. doi:10.2514/3.11710
- A. Ahmid, T.-M. Dao, V. N. Le (2019). Comparison study of discrete optimization problem using meta-heuristic approaches: a case study. *Int. J. Industrial Engineering and Operations Management*, 1(2), 97–109. doi:10.46254/j.ieom.20190201
- M. D. Phung, Q. P. Ha (2021). Safety-enhanced UAV path planning with spherical vector-based particle swarm optimization. *Applied Soft Computing*, 107, 107376. doi:10.1016/j.asoc.2021.107376 (code and terrain: BSD-style license, `uav/data/LICENSE_SPSO.txt`)
