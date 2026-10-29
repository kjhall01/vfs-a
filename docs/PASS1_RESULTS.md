# First implementation pass — 29 October 2026

## Scope

A separate working package extracted from checkpoint 54. Original archive SHA256:
`2e62765ffafe88be3250f695c4f600d1680995810c72f7ed1359846aed7027de`.
The original archive and source trees were not edited.

Existing numerical source is byte-identical. The only pre-existing Python file
changed is atmosphere/__init__.py, which exports BoundedGibbs. Four new modules:

- bounded_transport.py: CP carrier graph, all-neighbor species amount limiter,
  conservative divergence and limited traces.
- gibbs_means.py: analytic face means and composition entropy derivatives.
- bounded_moist.py: matched Gibbs correction, limiter work allocation and native
  bounded Euler/SSPRK3. Replaces imports from the historical experiment chain
  with explicit package-local imports; retains only the needed shared methods.
- moist_budgets.py: differentiable tensor inventory calculations.

The historical NativeMatchedUpwind ancestor is replaced by a small private
carrier-transpose helper retaining the exact pullback and CFL expressions.
The Gibbs correction and bounded update remain separate readable stages. No
numerical branches, formulas, floors or heat-release defaults were changed.
Newly extracted files were formatted and commented; old files were not reformatted.

## Evidence

CPU, float64, one thread; Python 3.12, PyTorch 2.15.0+cpu. Fixed 64-cell Fibonacci
mesh with no Lloyd iterations, six vertical levels, dt30. Original baseline
fixtures were generated in a separate process importing the old checkpoint.
The wheel was installed into a separate directory with --no-deps and tested
using the existing scientific environment; this is NOT a fresh dependency-solver
or all-platform installation test.

The installed wheel matched **142 tensors exactly**: dry JW-B control, smooth
mixture, sharp cloud and reversed cloud wind; initial states, tendencies,
Euler endpoints, three SSPRK3 steps, limiter amounts/fluxes, analytic energy
gradients and initial/final budgets. Active face counts: 0, 105, 119 for the
three moist cases. Scalar diagnostics and stage limiter counts also matched.
No research module was imported in the installed comparison process.

The focused suite completed **30 tests passed** (dry conservation, mixture
thermodynamics, moist reconstruction/dynamics, exact migration, inventory
autograd and all-neighbor flux bounds).

See equivalence.json, tests.log and tests/fixtures/provenance.json. The fixture
contains tensors and primitive containers and is loaded with weights_only=True.
Sources are hashed in tests/fixtures/baseline_sources.json. Exact equality is a
same-environment migration result, not a cross-platform promise.

The installed example completed 10 dt30 steps (300 simulated seconds), with
relative mass change -6.22e-16 and energy change -1.54e-5 J/m². That small forward
example is not a new climate or finite-step energy-conservation qualification.

One test initially demanded exact equality between an autograd inventory gradient
and independently computed area/gravity. Reciprocal multiplication versus division
differed by ~1.5e-16 relative; this test now permits two float64 epsilons. The
142 baseline tensor comparisons retain zero absolute and relative tolerance.

## Reproduce

From the extraction root:

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps --target /tmp/vfs-installed dist/vfs-1.0.0+extraction1-py3-none-any.whl
cd /tmp
PYTHONPATH=/tmp/vfs-installed OPENBLAS_NUM_THREADS=1 python /path/to/VFS_package_pass1/tools/check_baseline.py
PYTHONPATH=/tmp/vfs-installed OPENBLAS_NUM_THREADS=1 python -m pytest /path/to/VFS_package_pass1/tests -q
PYTHONPATH=/tmp/vfs-installed OPENBLAS_NUM_THREADS=1 python /path/to/VFS_package_pass1/examples/bounded_moist.py
```

To regenerate fixtures from the preserved archive, explicitly run
`tools/check_baseline.py --reference /path/to/extracted/checkpoint54`.
Do not replace fixtures merely to make an unexplained regression disappear.

## Limits and next pass

The full older package remains inside this single source tree, including broad
root imports, inherited dry infrastructure, extra components and implicit
solvers. This pass establishes installation and actual moist configuration
ownership before removing dependencies. Tests and docs outside the scoped
regression set remain in checkpoint 54; no historical code/results were deleted.

No GPU, compile, long-run or end-to-end backward qualification was added. The
inventory gradient test checks that the new tensor diagnostic retains autograd;
it is not a limiter-gradient or 6-hour backward test. No new physics, xarray
initializer, regrid scheme, time-averaged writer or implicit algorithm was added.
No license or public package publication was chosen. Proposed final class names
and API examples are not yet implemented.
