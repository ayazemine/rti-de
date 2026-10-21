# Official CEC test-function code

Unmodified C sources and input data of the competition organisers, used by
`benchmarks/cec_official_functions.py` (ctypes).

| Directory | Source archive | SHA-256 of the archive |
|-----------|----------------|------------------------|
| `cec2017/` | `CEC17_fast_pow-C++.zip` from https://github.com/P-N-Suganthan/CEC2017-BoundContrained (file `cec17_test_func.cpp`, the original version, and `input_data/`) | `51e7f55dc5cddf263ec128771dee1db6a2d0e2816b3d650160eb51960f269a1e` |
| `cec2022/` | `CEC2022.zip` from https://github.com/P-N-Suganthan/2022-SO-BO (`C-Code/cec22_test_func.cpp` and `C-Code/input_data/`) | `dd46b9efcfe79253c1dc0cb0f09c7f0ed0e24d5eb235ee14ca09be29f7def626` |

- `wrapper.cpp` defines the globals the official files expect and a C entry point `cec_eval`.
- `stub/WINDOWS.H` is an empty header: `cec22_test_func.cpp` includes `<WINDOWS.H>` and uses nothing from it, so the source file itself stays unchanged.
- `_build/` holds the compiled libraries (`g++ -O2 -shared -fPIC`); it is created on first use and can be deleted.

The archives carry no licence file.

Numbering follows the C code: CEC2017 `func_num` 1, 3–30 (2 is the deleted function), F* = 100 · `func_num`.
The updated technical report lists the same 29 functions renumbered 1–29 with F* = 100–2900; this repository
uses the numbering of the code.
