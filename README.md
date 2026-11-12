# VFS — atmosphere and ocean dynamics in PyTorch

VFS models hydrostatic atmospheric and ocean flow on spherical Voronoi C-grids.
It provides dry and moist atmospheric cores, an active stratified ocean with
JMD95 density, conservative transport, and energy-derived dynamical operators.
Models use explicit tensor states and inspectable component objects.

## Install

Python 3.10 or newer is required; verification uses Python 3.12 and CPU float64.
From a checkout:

```bash
python -m pip install .
python examples/moist_atmosphere.py
python examples/ocean_stratified.py
```

For an editable installation, use `python -m pip install -e '.[test]'`.
Optional extras include `plotting`, `data`, `diagnostics` and `validation`.
See [contributing](CONTRIBUTING.md) for the complete test environment.

## Models

| Model | Main features | Example |
| --- | --- | --- |
| `vfs.DryAtmosphere` | Conservative dry dynamics; SSPRK3 or ARK4 IMEX | [Dry atmosphere](examples/dry_atmosphere.py) |
| `vfs.MoistAtmosphere` | Entropy and water contents; bounded SSPRK3 with matched energy work | [Moist atmosphere](examples/moist_atmosphere.py) |
| `vfs.ocean.stratified.StratifiedOcean` | Active temperature/salinity, JMD95 density, bounded transport, optional energy-closing ALE | [Stratified ocean](examples/ocean_stratified.py) |
| `vfs.ocean.Ocean` | Configurable layered equations and source objects | [Layered reference](examples/ocean_reference.py), [constant density](examples/ocean_conservative.py), [centered JMD95](examples/ocean_jmd95.py) |

The atmosphere uses a hybrid pressure Charney–Phillips column. The ocean uses
layer thicknesses and layer-centered thermodynamic contents. Both store horizontal
normal velocity on mesh edges. Ocean components do not import atmospheric code.

## A moist atmospheric step

```python
import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients

mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
a, b = make_hybrid_coefficients(3, p_top=5000.)
column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
state = model.state_from_fields(temperature=280., surface_pressure=100000.,
                                normal_wind=0., vapor=.01)
state = model(state, dt=1.)
budgets = model.diagnostics(state)
```

The [ocean example](examples/ocean_stratified.py) constructs a three-layer state,
advances it, explicitly remaps its vertical grid, and reports energy and salt
budgets. Examples demonstrate API use; they are not balanced initial conditions
or timestep recommendations for other resolutions.

## Numerical scope

- Moist SSPRK3 requires positive vapor and uses bounded species-amount ledgers.
  Moist `CenteredIMEX` uses centered transport and does not provide those bounds.
- `StratifiedOcean` is unforced and unbatched. Its ALE correction checks column
  energy and entropy feasibility and rejects unsuccessful remaps.
- Semidiscrete energy closure does not imply exact finite-step conservation.
  The atmospheric energy budget includes the fixed-pressure lid reservoir.
- Tensor operations retain autograd connections, but limiter switches, iterative
  solves and long backward trajectories require separate qualification.
- CPU float64 is the verified baseline. GPU, reduced precision and long climate
  runs are not generally qualified. No coupled atmosphere–ocean driver or
  horizontal EKE closure is provided. An optional prescribed-length
  [ocean TKE closure](docs/OCEAN_TKE.md) supports
  [wind stress and flat-bottom drag](docs/OCEAN_BOUNDARY_STRESS.md).

## Documentation

- [Documentation index](docs/README.md)
- [State shapes and ownership](docs/STATE_OWNERSHIP.md)
- [Timestepping](docs/CURRENT_PATHS.md) and [package structure](docs/PACKAGE_OWNERSHIP.md)
- [Atmospheric initialization](docs/PHYSICAL_FIELDS_API.md) and [moist API](docs/MOIST_MODULE_API.md)
- [Stratified ocean API](docs/OCEAN_INTEGRATION.md)
- [Verification](verification/README.md) and [ocean measurements](verification/ocean/README.md)

- [Compatible atmosphere/ocean rotation](docs/COMPATIBLE_ROTATION.md)
- [Equiangular grid diagnostics and operators](docs/EQUIANGULAR_GRID.md)
