# Final bounded cleanup before verification

Baseline: merged stack at `0a69072acf6cb58e988dcb142dee5b26ebf0d1f6`.
This PR removes dead configuration and tensor wrappers, synchronizes examples and
current guides, and adds a reproducible archive tool. No benchmark physics,
architectural framework, performance work or Conda packaging is included.

## Deliberate API cleanup

- `IMEXSolverConfig` no longer accepts `implicit_operator`, `compile_kernels`,
  `compile_backend` or `compile_mode`. `DryPELinearOperator` likewise has no
  operator selector. Analytic ARK4 is the retained implementation, and the config
  defaults are valid for both dry and moist solvers. Removed keywords raise
  TypeError rather than selecting a fallback.
- Public effective settings are unchanged: DryIMEX uses Schur, dtype-dependent
  rtol (1e-8 for float64), atol=0, 40 iterations, restart=20; CenteredIMEX uses
  Schur, rtol=1e-9, atol=0, 80 iterations, restart=30. Explicit config overrides
  still select the supported preconditioners and residual controls.
- `release_as_heat` is removed from MoistAtmosphere, BoundedSSPRK3 and BoundedGibbs.
  Negative limiter work always becomes heat, with the existing endpoint allocation
  and mechanical correction. The inactive no-heat expressions are gone. All
  remaining arithmetic, lower-bound policy and direct Euler species ledgers remain.
- The `v(x)` helper and its tensor-only numerical call sites are removed. The
  audited state/column/transport inputs are tensors, and mesh operators return
  tensors for tensor inputs. Standalone Field-aware mesh/regridding dispatch and
  explicit legacy Field serialization in developer tools remain; no Field API is
  removed or changed.
- `tools/check_held_suarez_archive.py` reads pinned commit
  `849ccb829c7555ab8ecd1c11f0ae45993538f8ed` from local Git history. Only the four
  relevant definitions execute in memory. No historical package is installed or
  included in runtime wheels; the tool reports the source hash and exact results.

## Local validation

Apple M3, macOS 14.6.1 arm64, Python 3.12.13, PyTorch 2.15.0/Accelerate,
CPU float64, one Torch/OMP thread; wheel installed into site-packages and tested
outside the checkout. No accelerator or IMEX-backward qualification is added.

- **536 passed, 1 CUDA skip** in the installed-wheel portable suite. Existing
  conservation, positivity, ledger and derivative checks remain. The documented
  bounded donor-count scalar-conversion warning is unchanged.
- Both portable fixture tools pass: 142 checkpoint tensors and 93 retained dry
  tensors. No fixture or quantity-mapping/tolerance changes.
- All **298** existing pre/post captures and complete solver diagnostics are
  exactly equal to the pre-edit wheel. This includes states, tendencies, energy
  terms, representative gradients, bounded ledgers, and ARK4 stages for dry
  full/Schur/multilevel and moist full/Schur configurations.
- **15** additional tensors agree exactly: public default dry/moist ARK4 outputs,
  bounded SSPRK3 output and energy gradients through that complete bounded step.
  Retained default config fields and full solver records also compare exactly.
- The committed Held–Suarez tool verifies identical original method ASTs and
  **132** exact tensor comparisons, including forced stages and energy gradients.

All six example commands pass: dry SSPRK3/ARK4, bounded moist SSPRK3, centered
moist ARK4, low-level bounded ledger, and Held–Suárez forcing. The centered moist
example uses smooth positive composition: reusing the bounded sharp-cloud edge
was rejected by the unchanged stage validator, so it is not presented as a
supported centered scenario. No floor, clipping or tolerance adjustment was used.
README and ownership Python snippets and relative document targets were checked.
Six modules changed only by tensor unwrapping have matching executable ASTs after
removing the old wrapper calls.

Frozen-fixture exact mode fails locally: both fixture tests and standalone tools
report their exact failures separately from portable passes. First mismatches are
smooth initial wind (4.4408920985e-15 m/s) and diagonal dp_e (1.0913936421e-11 Pa).
Exact same-environment preservation is distinct from frozen-fixture exact mode.
Hosted checks are recorded in the PR after execution;
local checks must not be presented as completed GitHub runs. No further style
restructure is recommended: freeze the implementation for the
[verification handoff](VERIFICATION_HANDOFF.md).
