# Readability consolidation: changes and evidence

Baseline is merged PR #8, `849ccb829c7555ab8ecd1c11f0ae45993538f8ed`.
[The pre-edit map](SIMPLIFICATION_MAP.md) records the dependency trace and the
three-PR delivery. This report covers the dependent implementation slice; it
is regression evidence, not a new scientific verification campaign.

## Resulting implementation and deliberate removals

- `DryAtmosphere` owns geometry and the conservative dynamics once, with plain
  `DrySSPRK3()` or `DryIMEX(solver_config=...)` collaborators. It replaces the
  old `DryHydrostaticModel`, `HeldSuarezModel` and registry-driven `VFSModel`;
  there are no aliases or recipe migration dispatchers. Dry state is unbatched
  and tracer-free for these dynamics; standalone tensor State still supports
  tracer storage/conversion and mesh restriction.
- Conservative equations are ordinary methods in `conservative.py`; the
  `DryEnergy` base groups thermal-energy derivatives and budget diagnostics.
  The externally bound dry function table, canonical/strategy alternatives,
  process/SCM and registry-based forcing recipes, split explicit and shadow tracer paths are removed.
  Existing atmospheric initial states, humidity/I/O, spherical geometry,
  topography, regridding and mesh transfers remain. Nonzero analytic-climate
  nonlinear-balance tuning tied to the removed canonical core now rejects;
  the default analytic initialization remains.
- `ScalarTransportState` borrows wind, pressure, scalar content and geometry.
  Dry theta content (Pa K) and moist entropy content (Pa J/(kg K)) share the
  transport calculation without pretending entropy is `mtheta_i`. The view has
  no dry temperature API, conversions, copies or cached diagnostics. Mixture
  validation and gradient connections remain. The two CP mass implementations
  are not merged: half-then-add versus add-then-half order is preserved.
- `BoundedGibbs` retains the direct Euler species amount ledger; its low-level
  constructor now requires explicit `release_as_heat=True`. False is rejected
  by both it and `BoundedSSPRK3`. The model's existing bounded/heat default is
  unchanged. The old centered explicit step entry is removed; the centered RHS
  remains essential to bounded explicit and ARK4.
- ARK4 stage assembly lives in `dry_ark.py` / `moist_ark.py`; residual-checked
  GMRES, Schur and multilevel preconditioning live in `solvers.py`. Analytic
  conservative/mixture waves remain. Coupled-JVP, compiled, fixed-work and
  provider-composition implementations are removed. Low-level dry construction
  requires an explicit analytic-wave config; `DryIMEX()` selects that wave and
  Schur. Rejected historical flags do not select replacement algorithms.

The supported ARK4 evidence is narrow: dry full, Schur and multilevel Schur;
moist full and Schur; eager analytic waves on the existing representative cases.
Centered moist ARK4 does **not** provide bounded species transport. No bounded
IMEX algorithm, automatic fallback or IMEX backward qualification is introduced.

## Same-environment preservation

Before edits, `tools/capture_readability.py --original --save ...` captured the
installed merged-PR-8 wheel. After each slice, `--compare ...` uses exact tensor
comparison and exact scalar diagnostic comparison. Captures live outside frozen
fixtures. All **298 tensors and complete solver diagnostics are exactly equal**:
states, tendencies, component energies, diagnostic energy gradients/RHS VJPs,
bounded Euler and SSPRK3 ledgers, and all ARK4 nonlinear stages. At dt=600 s,
rtol=1e-11, max_iterations=100, total iterations remain dry 29/13/23 (full/Schur/
multilevel) and moist 30/20 (full/Schur). See the capture script for case seeds,
shapes and measurement definitions.

Local environment: macOS arm64, Apple M3 CPU, Python 3.12.13, PyTorch 2.15.0,
float64, Accelerate BLAS, one Torch/OMP thread. No new accelerator test or IMEX
backward measurement was performed.

## Installed-wheel checks outside the checkout

Implementation slice: **519 passed, 1 CUDA skip**. Both standalone portable
comparison tools and both examples (`moist_atmosphere.py`, `bounded_moist.py`)
pass. Existing conservation, positivity, limiter, exact ledger, finite-difference
and active-branch derivative tests remain. The donor subcycle count still emits
the documented scalar-conversion warning under gradients; it was not hidden.

The unchanged checkpoint fixture still checks 142 tensors. The unchanged dry
fixture checks the 93 conservative-diagonal tensors; its 100 canonical-core
paths are explicitly retired with that algorithm. `retained_reference` accepts
only the known original archive layout and the portable path map still rejects
unknown actual paths. Fixture bytes, all retained quantity mappings, tolerances
and independent physical/discrete checks are unchanged. This deliberate API
removal explains part of the test-count reduction; it is not a tolerance change.

Frozen-fixture **exact mode fails locally**, separately from exact pre/post and
portable success. First dry mismatch: `diagonal/primitives/dp_e`, max absolute
1.0913936421e-11 Pa; first moist mismatch: `smooth/initial/u_e`, max absolute
4.4408920985e-15 m/s. These are cross-environment baseline failures, not converted
to passes. Existing CI records exact failures informationally and gates portable.

GitHub evidence: PR #9 head `465849d` completed both Linux CPU workflow runs
[41089424874](https://github.com/kjhall01/vfs-a/actions/runs/41089424874) and
[41089353027](https://github.com/kjhall01/vfs-a/actions/runs/41089353027)
successfully. Implementation-slice hosted results are recorded in its PR after
completion; local results above are not a claim of GitHub completion.

## Held–Suarez retention before merge

The standalone [Held–Suarez source](HELD_SUAREZ.md) preserves PR #8 thermal
relaxation and Rayleigh drag without restoring the old physics module/framework.
All original methods have identical executable ASTs; 132 archived/new tensors
(rates, RHS, forced stages/state, energy and gradients) agree exactly on the
local CPU environment. Separate source tendencies expose distinguishable budget
work. The new runnable example is included in installed-wheel CI.
