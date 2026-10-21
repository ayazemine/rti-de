# Third-party material

The RTI-DE code (`python/`, `matlab/`, `tests/` and the experiment code written for the paper) is
under the MIT License (`LICENSE`). The following files in `experiments/` come from third parties
and keep their own terms. They are not covered by the MIT License of this repository.

| Path | Origin | Terms |
|---|---|---|
| `experiments/benchmarks/official_cec/cec2017/` (`cec17_test_func.cpp`, `input_data/`) | Official CEC 2017 bound-constrained test-function code of the competition organisers, `CEC17_fast_pow-C++.zip` from https://github.com/P-N-Suganthan/CEC2017-BoundContrained (details and SHA-256 of the archive in `experiments/benchmarks/official_cec/README.md`) | The archive carries no licence file. |
| `experiments/benchmarks/official_cec/cec2022/` (`cec22_test_func.cpp`, `input_data/`) | Official CEC 2022 single-objective bound-constrained code, `CEC2022.zip` from https://github.com/P-N-Suganthan/2022-SO-BO | Same as above. |
| `experiments/applications/uav/data/ChrismasTerrain.tif` | Terrain map of the SPSO UAV path-planning code by Manh Duong Phung (https://github.com/duongpm/SPSO) | BSD-style licence, reproduced in `experiments/applications/uav/data/LICENSE_SPSO.txt` |

`experiments/benchmarks/official_cec/wrapper.cpp` and `stub/WINDOWS.H` were written for this
repository. The CEC sources themselves are unmodified.

`experiments/applications/uav/path_planning.py` is a Python port of the MATLAB code of the SPSO
repository named above; its retained licence is `LICENSE_SPSO.txt`. The packages listed in
`experiments/requirements-experiments.txt` (NumPy, SciPy, pandas, Matplotlib, Pillow, mealpy) are
installed separately and are under their own licences.
