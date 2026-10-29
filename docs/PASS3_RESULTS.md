# Pass 3 — focused dry-kernel cleanup

## Change

DryHydrostaticDynamics now directly inherits torch.nn.Module. The six former
mixin files remain cohesive equation modules: transport, momentum,
thermodynamics, tracers, tendency diagnostics, and energy. Their 99 methods
are now ordinary functions explicitly bound in the model class. This retains
subclass overrides and existing method calls without multiple inheritance or
runtime method injection. It avoids merging the equations into one huge file.

Public model methods and numerical defaults are unchanged. Direct imports of
the six internal *Mixin classes are no longer supported. This is an internal
structure migration, not a finished public API or complete package downsizing.
State/configuration infrastructure and optional legacy modules remain.

## Evidence

Tested the installed 1.0.0+extraction3 wheel using PyTorch 2.15.0+cpu, CPU,
float64 and one torch thread.

- 142 checkpoint-54 fixture tensors match exactly.
- 193 additional pass-2 dry fixture tensors match exactly: canonical and
  conservative tendencies, an SSPRK3 step, shared primitives, energy/force/
  pressure/thermodynamic diagnostics, and canonical passive tracer budgets.
- All 99 extracted functions have identical ASTs after normalizing docstring
  indentation only. Function ownership is checked explicitly.
- 66 tests pass, including the inherited small dry/moist IMEX tests and
  analytical derivative/temporal convergence checks.
- The first structural test failed because removing class indentation also
  changed indentation inside docstring literals. The test now normalizes only
  that whitespace on both baselines. No equation or tolerance was changed.
- The additional fixture harness initially omitted the required dt argument
  to cfl_terms; it was corrected to dt=1 before capturing the pass-2 fixture.

Logs and machine-readable comparisons are in this directory. The additional
fixture and its provenance are under tests/fixtures. The original checkpoint
fixture is unchanged.

These results establish migration equivalence on the recorded environment,
not new scientific verification, performance gains, arbitrary-time-step
positivity, long-run IMEX suitability, or GPU/full-backward qualification.
No physical operators, integrators, bounded species ledger, or release-as-heat
defaults were changed.

## Reproduce

Install the wheel in an environment with the scientific dependencies, then run
from outside the checkout:

    python /path/to/package/tools/check_baseline.py
    python /path/to/package/tools/check_dry_helpers.py
    python -m pytest /path/to/package/tests -q

Next is the public model/state and physical-variable initialization boundary.
Xarray adapters, explicit regridding, and averaged output remain later work.
