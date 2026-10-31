# Ocean extraction results — 31 October 2026

Base: `941b794eb221c3e31d0244b936c1c5c79ec14e1b` (PR #22).
No shared runtime or atmospheric source changed. Source archive unchanged.

- Installed wheel outside checkout, CPU float64, Python 3.12.14,
  PyTorch 2.15.0+cpu: **566 passed, one CUDA skip**, 59.74 s.
- Ocean focused exact suite: **11 passed**, including the capture comparison,
  tensor identity, shape/dtype/geometry rejection, restart, output, and a
  finite-difference check through one SSPRK3 step.
- Archive/source to extracted-package capture: **1,416/1,416 arrays bitwise
  equal**, including all recorded stages, terms, gradients and process outputs.
- Parent atmospheric source capture to installed extracted wheel: **298 tensors
  and complete scalar/solver records exactly equal**. Main-to-installed output
  capture: **74 tensors and metadata exactly equal**.
- Both existing portable fixture tools pass: **142 + 93 tensors**, maximum
  reported budget fraction zero in every category. Fixtures/tolerances intact.
- Source audit: **194 retained definitions unchanged** after excluding
  docstrings. The two changed common definitions are state construction
  (Field unwrapping) and kinetic_energy (direct call to the identical mesh
  method). Removed methods are output/schema/registry/mesh-forwarding shells.
- Ten-step new example: volume relative change 1.672991116787672e-16,
  minimum h=999.9668268066263 m. This is a smoke result, not a science target.

The first installed test attempt used PYTHONPATH; six isolated subprocess tests
could not see those dependencies. A private virtual environment with the same
dependencies resolved that setup issue; all tests then passed. No test source,
fixture or tolerance was altered to address it. The inherited atmospheric
limiter scalar-conversion warning remains.

No new energy closure, balance, positivity, convergence, GPU, compiled, implicit
or long-rollout/backward claim is made by this extraction. The source audit's
known numerical defects remain reproducible. The new facade supports unbatched
states and explicit geometry/dtype ownership; it does not restore recipes,
old coupled models, ETOPO initialization, or implicit ocean stepping.
