# VFS — spherical atmospheric dycores

This is an incremental packaging pass from checkpoint 54, not a new scientific
formulation or a finished public API. The original checkpoint is unchanged.
The package now contains the actual bounded moist configuration without imports
from research directories. Existing dry/centered-moist equations are unchanged.

## Install

Use Python 3.10+ in a virtual or Conda environment:

```bash
python -m pip install .
python examples/moist_atmosphere.py
```

For local development: `python -m pip install -e '.[test]'`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for installed-wheel portable checks and exact
comparison mode. Optional data/plotting extras are only needed for those workflows.
`environment.yml` provides a Conda environment; this is not yet a conda-channel
release. No public package name/version availability or license was established
in this pass. The local build version is `1.0.0+extraction4`, derived from the
pre-existing metadata, not a claim of a published release.

## Physical-field workflow

Start with [the physical-fields guide](docs/PHYSICAL_FIELDS_API.md) and
examples/moist_atmosphere.py. MoistAtmosphere owns geometry, creates the existing
MoistState from interface temperature, pressure, edge wind and water fractions,
and exposes bounded stepping and differentiable conservation diagnostics.
Vapor must be strictly positive; no floor is added.

## Underlying API

```python
from vfs.atmosphere import MoistConservativeDynamics, BoundedGibbs

base = MoistConservativeDynamics()
model = BoundedGibbs(base, release_as_heat=True)
# state is a MoistState on a VoronoiMesh and CharneyPhillipsColumn
rhs = model.tendency(state, dt=30.0)
next_state = model.step(state, dt=30.0)
energy = base.boundary_total_energy(next_state)
```

Use `release_as_heat=True` to match the frozen bounded moist baseline. The original
wrapper's default remains False. `MoistConservativeDynamics.step` alone uses
centered transport and is NOT the bounded configuration. Do not substitute a
generic `state + dt * rhs`: the bounded Euler stage updates species using its
cancellation-safe amount ledger. This dt is a finite transport budget, not a
claim of positivity for arbitrary RK/IMEX combinations.

`vfs.atmosphere.moist_budgets.inventories(base, state)` returns tensor-valued
mass (kg), vapor/liquid/ice masses (kg), entropy inventory (J/K), and total energy
(J, including the top-boundary reservoir). Existing energy components and
analytic gradients remain on the base model. Reporting may detach; these
calculations do not. Full backward/GPU qualification is not claimed.

## Verify the extraction

```bash
python tools/check_baseline.py --mode portable
python tools/check_dry_helpers.py --mode portable
python -m pytest tests --fixture-mode=portable -q
```

The first command compares against a checked-in fixture captured from the
unchanged checkpoint-54 research configuration. It covers dry, smooth moist,
cloudy and reverse-wind cloudy cases, tendencies, direct Euler update, three
SSPRK3 steps, flux amounts, energy gradients and budgets. It also rejects
research-module imports. Exact comparisons are a migration check on the recorded
CPU/float64 environment, not a cross-platform bitwise reproducibility promise.

## Package scope

Retained: spherical geometry and mesh generation; dry and moist atmospheric
dycores; explicit and existing experimental implicit integrators; conservation
and differentiable diagnostics; atmospheric initialization, regridding and I/O.
Implicit methods remain experimental. The bounded moist ledger and positive-vapor
restriction are unchanged.

Ocean, land, ice, surface-process and atmosphere–ocean coupling packages, coupled
component-graph orchestration, learned PGF/damping/transport models and ML mesh
kernels have been deliberately removed. These imports and learned recipes are
no longer supported. Mesh hierarchy and non-ML tracer helpers remain. No second
legacy package is provided; removed code remains in Git history.

See [scope reduction and migration notes](docs/SCOPE_REDUCTION.md), including
retained dependencies, validation and the next readability work. Read
[the handoff](handoff/index.md) before further changes. Historical extraction
reports remain under docs/PASS1_RESULTS.md through docs/PASS4_RESULTS.md.
The small root API and explicit named atmospheric compatibility imports remain.
No numerical kernel reorganization or scientific verification campaign is part
of this reduction.
