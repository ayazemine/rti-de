# RTI-DE: Rejected-Trial Interpolation Differential Evolution

RTI-DE is a differential evolution (DE) variant for the minimisation of a black-box function over a
box `[lb, ub]` within a fixed budget of function evaluations. Unlike classic DE, it does not discard
the evaluation of a rejected trial point. Late in the run (by default after 70 % of the budget), an
agent whose trial point was rejected evaluates the half step along the same displacement. If that
is rejected as well, it evaluates the minimiser of the parabola through the three values known on
that line (the *rejected-trial chain*). The other components come from established DE variants:

- DE/current-to-pbest/1 mutation (JADE) without the external archive, with midpoint bound repair;
- binomial crossover with a fixed rate CR = 0.9;
- step factor F ~ N(mu_F, 0.15) clipped to [0.05, 1], where mu_F is adapted from the successful
  trial points (weighted Lehmer mean);
- linear population size reduction from 18 D to 4 agents (L-SHADE).

The complete rules are in the docstring of `python/RTIDE/optimizer.py` and in Section 3 of the paper.

This repository contains:

```
python/RTIDE/        the Python package (import name RTIDE, distribution name rtide-optimizer)
matlab/              MATLAB / GNU Octave port (rti_de.m), example and verification against Python
tests/               unit tests of the algorithm (Python)
experiments/         code of the experiments of the paper (benchmarks, engineering applications,
                     experiment runner, table and figure scripts, their tests)
```

## Publishing to PyPI

Releases are built and published from GitHub tags using PyPI Trusted Publishing. To enable the
first release, sign in to PyPI as `emineayaz` and add a pending publisher at
<https://pypi.org/manage/account/publishing/> with these values:

- PyPI project name: `rtide-optimizer`
- GitHub owner: `ayazemine`
- GitHub repository: `rti-de`
- Workflow: `publish.yml`
- GitHub environment: `pypi`

Create the `pypi` environment in the GitHub repository settings and require a reviewer before
publishing. The pending publisher does not reserve the project name until its first successful
upload.

For each release, update `version` in `pyproject.toml`, create a matching `v<version>` tag, and
push it to GitHub. For example, the current `1.0.0` version can be published with:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow builds and checks the distributions before publishing them to PyPI.

## Installation (Python)

The only dependency is NumPy; the code was tested with Python 3.11.9 and NumPy 1.26.0.

```bash
pip install rtide-optimizer   # from PyPI
pip install .                 # from a clone of this repository
```

## Usage (Python)

```python
import numpy as np
from RTIDE import rti_de

def rastrigin(X):                      # vectorised: (k, D) array -> k values
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)

D = 10
lb, ub = np.full(D, -5.12), np.full(D, 5.12)
x_best, f_best, fes = rti_de(rastrigin, lb, ub, max_fes=50_000, seed=1)
```

Optional arguments (defaults): `n_init=None` (18 D), `n_min=4`, `p=0.11`, `chain=2`,
`chain_from=0.7`, `CR=0.9` (`None`: no crossover), `lpsr=True`, `adapt=True`, and `trace=None`
(a list that receives one record per generation). A NaN objective value is treated as +inf. The
function returns `(x_best, f_best, evaluations_used)`.

## Usage (MATLAB / GNU Octave)

```matlab
addpath('matlab')
rastrigin = @(X) 10 * size(X, 2) + sum(X .^ 2 - 10 * cos(2 * pi * X), 2);   % one point per row
D = 10;
[x_best, f_best, fes] = rti_de(rastrigin, -5.12 * ones(1, D), 5.12 * ones(1, D), 50000, 1);
[x_best, f_best, fes, trace] = rti_de(rastrigin, -5.12 * ones(1, D), 5.12 * ones(1, D), 50000, 1, ...
                                      'chain_from', 0.5, 'CR', 0.9);
```

`rti_de.m` takes the same options as name-value pairs (or a struct) and also needs `rtide_rules.m`.
See `matlab/example_rti_de.m`. MATLAB R2016b or later is required (implicit expansion). Because the
two languages use different random number generators, a MATLAB run with a given seed differs from
the Python run with that seed. With the same random numbers, the two implementations give
bit-identical runs. `matlab/VERIFICATION.md` reports this check (GNU Octave 11.3.0).

## Tests

```bash
python -m unittest discover -s tests -v                    # algorithm; also imports DE and L-SHADE from experiments/
cd experiments && python -W ignore -m unittest discover -s tests -v
```

Install `experiments/requirements-experiments.txt` first. The CEC tests compile the official C
code with `g++` on first use (into `experiments/benchmarks/official_cec/_build/`). The reference-run
test in `tests/test_rtide.py` compares a stored CEC2017 result bit-exactly on Windows (MinGW build)
and with a relative tolerance of 1e-3 elsewhere.

MATLAB/Octave verification (from `matlab/verification/`):

```bash
python verify_rtide.py prepare
octave-cli --no-gui --quiet --eval "run_octave_checks"
python verify_rtide.py compare          # writes output/summary.json and ../VERIFICATION.md
```

## Reproducing the experiments of the paper

All commands are run from `experiments/` in a Python 3.11 environment:

```bash
cd experiments
pip install -r requirements-experiments.txt
python benchmarks/cec_official_functions.py        # builds and checks the official CEC libraries (g++)

# CEC2022 (D = 20) and CEC2017 (D = 100): 7 algorithms, 30 runs, 1,000,000 evaluations
python -W ignore run_experiments.py --suites cec2022 cec2017 --budget fe --max-fes 1000000 --runs 30 --workers 31

# engineering problems (composite laminate, with and without contiguity constraint; UAV path planning)
python -W ignore run_experiments.py --suites composite composite_contiguity uav --runs 30 --workers 31

# ablation (6 variants) and parameter sensitivity (8 variants)
python -W ignore run_experiments.py --suites cec2022 cec2017 --budget fe --max-fes 1000000 --runs 30 --workers 31 --algorithms RTI-DE-noChain RTI-DE-halfStep RTI-DE-chainAlways RTI-DE-noLPSR RTI-DE-fixedF RTI-DE-noCR --out results_cache/paper_ablation
python -W ignore run_experiments.py --suites cec2022 cec2017 --budget fe --max-fes 1000000 --runs 30 --workers 31 --algorithms RTI-DE-cf50 RTI-DE-cf90 RTI-DE-p05 RTI-DE-p20 RTI-DE-n10D RTI-DE-n25D RTI-DE-cr70 RTI-DE-cr95 --out results_cache/paper_sensitivity

# tables and figures (written to experiments/outputs/)
python -W ignore scripts/make_tables.py
python -W ignore scripts/make_figures.py
```

- `--dry-run` lists the work and a rough duration without running anything.
- An interrupted command can be restarted; finished runs are skipped.
- `--workers` should be at most the number of cores minus one.
- The runner refuses to continue a result directory if `python/RTIDE/optimizer.py` or the
  reference implementations in `experiments/benchmarks/` have changed (SHA-256 recorded in
  `config.json`).
- `experiments/applications/composite/exhaustive_search.py` recomputes the exact reference optima of
  the laminate problems (`reference_optimum.json`).
- `experiments/scripts/supplementary_stats.py` computes supplementary statistics (seed-paired Wilcoxon
  tests, rank-sum tests with a tie tolerance, Holm adjustment over a suite) from the generated
  tables and from the `run_log.txt` and `summary.csv` files of `run_experiments.py`, copied to
  `experiments/results_summary/<set>/`.

The values of all tables of the paper, at full precision, are in `experiments/results_tables/`
(one CSV file per table; the README there maps the files to the table numbers). The raw results of
the individual runs are not included in this repository. The problems are described in
`experiments/applications/README.md` and the CEC code in
`experiments/benchmarks/official_cec/README.md`.

## License and citation

The RTI-DE code is released under the MIT License (`LICENSE`). If you use RTI-DE in research, please
cite the paper:

> E. Ayaz, "Rejected-Trial Interpolation Differential Evolution with Applications to Composite
> Laminate Design and UAV Path Planning."

Citation metadata: `CITATION.cff`.

The official CEC 2017/2022 test-function code in `experiments/benchmarks/official_cec/` and the SPSO
terrain data in `experiments/applications/uav/data/` are third-party material under their own
terms; see `NOTICE.md`.
