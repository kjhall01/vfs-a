# PR: Add installed-wheel CI and contributor setup

The repository now has a GitHub Actions workflow that builds and installs a
wheel with the small `test` extra (pytest only), then runs the suite, both
baseline comparison tools and the MoistAtmosphere example outside the checkout.
It asserts the installed import path, records dependency versions, preserves
failure status, and runs each check even when another check fails. Successful
comparison reports and pytest XML are uploaded as workflow artifacts; failure
tracebacks remain in the Actions logs.

CONTRIBUTING.md documents local setup, wheel validation and focused PRs. The
handoff now identifies GitHub as the development location. Numerical package
sources, fixtures, tolerances, operators, defaults, state conventions and
integrators are unchanged. The bounded species amount ledger,
`release_as_heat=True`, tensor boundary and interface/edge staggering remain
intact. Vapor must be strictly positive, with no floor or claim of exact-zero
water equivalence to the dry core. No legacy modules were removed, xarray
features added, license selected or package published.

## PyTorch availability

On 2026-10-29, the [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
listed `torch-2.15.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`. A direct pip
download of that wheel completed successfully (209.5 MB), using:

```bash
python -m pip download --no-deps --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --python-version 312 \
  --implementation cp --abi cp312 --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.15.0+cpu' -d /tmp/vfs-ci-linux-wheel
```

Thus the recorded version is publicly obtainable; no replacement version was
invented. CI targets Ubuntu 24.04, Python 3.12 and this CPU build. Downloading
the Linux wheel on macOS is not a Linux installation/execution test, and a
matching version does not promise cross-platform bitwise agreement.

## Local validation

macOS 14.6.1 ARM64, standalone CPython 3.12.13, PyTorch 2.15.0 macOS wheel,
CPU float64, one thread. Other resolved versions: numpy 2.5.4, scipy 1.18.1,
pytest 9.2.0, netCDF4 1.7.4, xarray 2026.9.0. Installed in a fresh temporary
virtual environment; tests/tools/examples copied to `/tmp/vfs-ci-check`.
The asserted VFS import was in the environment's `site-packages`, outside the
checkout, with Python isolated mode (`-I`).

- Wheel build and dependency installation passed; `pip check` passed.
- All 138 Python files in the wheel match the package source bytes; no extra
  Python files were packaged. Wheel metadata contains the pytest-only extra.
- Full installed-wheel suite: **76 passed, 2 failed** in 20.82 s. The failures
  are `test_expanded_dry_baseline` and
  `test_checkpoint54_exact_equivalence_without_runtime_autograd`.
- Both standalone baseline tools ran and failed at their first exact mismatch:

| Comparison / first field | Unequal elements | Maximum absolute difference | Maximum relative difference |
| --- | --- | --- | --- |
| checkpoint-54: `smooth/initial/u_e` | 204 / 1116 | 4.440892098500626e-15 | 7.036663749195453e-16 |
| dry helpers: `canonical/primitives/dp_e` | 102 / 360 | 1.0913936421275139e-11 | 3.905571514745662e-16 |

The tools stop at the first mismatch; these are not whole-fixture error bounds.
Differences occur in initial wind and dry pressure primitives, before the
bounded time update. Different platform/build and dependency environment are
possible causes; this run does not isolate the cause or establish equivalence.
No fixtures were regenerated and no assertions were relaxed.

- `examples/moist_atmosphere.py` passed: 10 steps, 300 simulated seconds.
  Relative mass change: -8.289568870754183e-16; species changes: [-24, -1, 0] kg;
  energy change: -1.5360724581863678e-5 J/m²; minimum vapor/liquid/ice:
  [0.009999999995771525, 1.4423007889908123e-50, 0]. These are diagnostics,
  not scientific acceptance thresholds.
- Workflow YAML parsing, embedded Bash syntax and `git diff --check` passed.

Initial local setup issues were resolved without changing package requirements:
pip selected a netCDF4 source release requiring absent system headers;
`--prefer-binary` selected an available wheel. The existing Conda Python 3.12.4
segfaulted on a standalone `import readline`, also preventing pytest startup.
A separate downloaded CPython 3.12.13 ran the full suite successfully to
completion. Both interpreters produced the same first fixture mismatches and
smoke diagnostics.

## GitHub Actions status and limits

**Not run in GitHub Actions yet.** No hosted check is claimed as passing, and
the Linux job's exact comparisons remain unverified. Review its logs and
artifacts after pushing/opening the PR, and investigate any mismatches without
regenerating fixtures or weakening tolerances. The existing pass-4 results
remain historical same-environment evidence, not results of this workflow.

This PR adds CPU forward checks and retains the existing diagnostic-gradient
tests. It does not qualify GPU execution, long runs or full backward rollouts.
