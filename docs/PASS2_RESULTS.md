# Pass 2: imports and bounded-model inheritance

29 October 2026. Checkpoint 54 and pass 1 are preserved separately.

## Changes

- Reduced default exports/imports in vfs, core, atmosphere, geometry and mesh.
  Default atmospheric construction loads 29 VFS modules versus 63 in pass 1
  (fresh processes, same environment). This is an import-footprint measurement,
  not evidence of faster timesteps or lower rollout memory.
- Root exports now include BoundedGibbs, MoistState, MoistConservativeDynamics
  and DryHydrostaticConservativeDynamics. Existing explicit module imports work.
- Optional old names resolve on explicit attribute/named import through small
  module-level compatibility maps. No new model registry or configuration layer.
  `from vfs import *` now deliberately exposes only the small core list. Old
  global registries are no longer automatically populated by importing every
  domain: callers relying on those side effects must import their owner modules.
- Learned PGF dependencies load only inside the explicitly selected constructor
  branch. Its equations and options are unchanged; construction is tested.
- BoundedGibbs is now one class. The carrier transpose, CFL and Gibbs face
  correction are explicit methods. Removed two private ancestor classes and
  replaced super calls with direct calls/assignment. No arithmetic changed.

## Verification

Installed wheel outside the source checkout, CPU float64 / PyTorch 2.15.0+cpu:

- All 142 checkpoint-54 tensor comparisons still match exactly, with identical
  limiter counts and scalar diagnostics. See equivalence_pass2.json.
- 33 focused tests passed (tests_pass2.log), including new fresh-process tests
  for default import isolation, named optional imports and learned PGF selection.
- The cloudy 300-second example completes with the same pass-1 output.
- Imports of dry/moist ARK classes remain available; this pass does not add
  numerical tests or performance qualification for those implicit solvers.

Reproduce the baseline and test commands in README. Wheel version is
1.0.0+extraction2. No dependency changes or fresh-environment dependency solver
validation in this pass; installation uses the existing scientific environment.

The footprint script initially attempted to import the newly added root export
from pass 1, where it did not exist. It was corrected to use the common
vfs.atmosphere import for both measurements. No model change was needed.

## Still to do

The six-mixin dry inheritance and state/config dependencies remain. They carry
real shared kernel behavior and need a dedicated extraction with broader
coverage; they were not flattened into one enormous file. Legacy code remains
in the archive/install but is no longer pulled into the normal import path.

Next focus: simplify that shared dry/kernel dependency boundary, then introduce
the physical-variable initialization API. Xarray/terrain adapters and regridding
remain planned. No numerical defaults, floors, dt budgets, mesh geometry,
thermodynamic conventions or integration arithmetic were changed.
