# VFS — spherical atmospheric dynamics

VFS provides conservative dry and energy-closing, transport-bounded moist
hydrostatic dynamics on a spherical Voronoi C-grid, with PyTorch tensor states,
explicit or ARK4 stepping, differentiable diagnostics, and atmospheric I/O.
Models own fixed geometry; states hold prognostic tensors. Positive vapor is
required by the moist core; no input moisture floor is added.

## Install and try

Use Python 3.10+ (CI uses 3.12). From this repository:

```bash
python -m pip install .
python examples/moist_atmosphere.py
```

For development use `python -m pip install -e '.[test]'`; see
[CONTRIBUTING.md](CONTRIBUTING.md) for isolated installed-wheel checks.
Conda packaging is deferred. No package publication or license selection is
part of this cleanup.

```python
import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients

mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
a, b = make_hybrid_coefficients(3, p_top=5000.)
column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
state = model.state_from_fields(temperature=280., surface_pressure=100000.,
                                normal_wind=0., vapor=.01)
state = model(state, dt=1.)  # bounded SSPRK3 with intrinsic heat release
budgets = model.diagnostics(state)  # tensor mass, species, entropy and energy
```

## Supported configurations

| Configuration | Runnable command | Important distinction |
|---|---|---|
| Dry `DrySSPRK3()` (default) | `python examples/dry_atmosphere.py` | Conservative, unbatched, tracer-free core |
| Moist `BoundedSSPRK3()` (default) | `python examples/moist_atmosphere.py` | Authoritative Euler species ledger; intrinsic heat allocation |
| Dry `DryIMEX()` | `python examples/dry_atmosphere.py --imex` | ARK4 analytic wave split; residual-checked solves |
| Moist `CenteredIMEX()` | `python examples/moist_atmosphere.py --imex` | Smooth-composition demo; **not bounded species transport** |
| Standalone Held–Suarez forcing | `python examples/held_suarez.py` | Explicit sum at each dry RK stage; separate thermal/drag budgets |

These short runs demonstrate the API, not balanced benchmarks or safe timesteps
at arbitrary resolutions. Instantaneous energy closure includes the fixed-lid
reservoir; it does not imply exact finite-step energy conservation. Tested
one-step derivatives do not qualify IMEX backward, arbitrary limiter switches,
GPU execution or long-rollout gradients.

## Read further

- [Follow one timestep](docs/CURRENT_PATHS.md) and [bounded moist details](docs/BOUNDED_MOIST_READING_GUIDE.md).
- [State/model ownership](docs/STATE_OWNERSHIP.md), [physical initialization](docs/PHYSICAL_FIELDS_API.md), [Held–Suarez sources](docs/HELD_SUAREZ.md).
- [Cleanup evidence](docs/FINAL_CLEANUP_RESULTS.md) and [verification handoff](docs/VERIFICATION_HANDOFF.md).
- [Scientific verification tracker, frozen baseline and reproducible measurements](verification/README.md).
- [Portable regression policy](docs/PORTABLE_REGRESSION.md), [historical reports](docs/HISTORY.md), and [development handoff](handoff/index.md).

Freeze the retained implementation for the verification phase. Numerical fixes
or missing benchmark capabilities should be proposed separately from cleanup.
