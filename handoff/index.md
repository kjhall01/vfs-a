# Current handoff — GitHub development and CI

Development is now on https://github.com/kjhall01/vfs-a. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the small test extra, local setup,
installed-wheel checks and focused PR preparation. The new GitHub Actions
workflow targets Ubuntu CPU/Python 3.12/PyTorch 2.15.0+cpu and runs the suite,
both unchanged exact baseline tools and examples/moist_atmosphere.py outside
the checkout. The public Linux CPU wheel was successfully downloaded.

See [CI setup results](../docs/CI_SETUP_RESULTS.md) for local validation and
remaining failures; GitHub Actions execution has not yet been verified.
No numerical operators, defaults, state conventions or integrators changed.
MoistAtmosphere retains release_as_heat=True and its bounded species amount
ledger. Vapor remains strictly positive with no floor; exact-zero-water dry
equivalence is not established. GPU, long-run and full backward behavior remain
untested. Package publication and license selection still require direction.

Earlier sections below are historical snapshots.

---

# Historical handoff — extraction pass 4

Read docs/PASS4_RESULTS.md and docs/PHYSICAL_FIELDS_API.md first.
Earlier sections below are history.

Added MoistAtmosphere (composition, no new state type) and a physical-field
initialization/diagnostic/step workflow. Geometry and pressure ownership,
units, staggering, broadcasting and tensor boundaries are explicit.
Numerical kernels are unchanged. Run examples/moist_atmosphere.py.

Important finding: the inherited moist adapter rejects zero vapor. The facade
now requires positive vapor up front, with no floor. Zero condensates work.
This is NOT a demonstrated exact-zero-water reduction to the dry core.

Next: move this snapshot to GitHub and iterate on API/docs there. Before any
public distribution, select a license and review legacy package metadata.
Xarray initialization/regridding and a friendly dry facade remain later work.
Keep the separate scientific verification campaign and kernel changes explicit.

---

# Current handoff — extraction pass 3

Read [pass-3 results](../docs/PASS3_RESULTS.md), then [README](../README.md).
Earlier sections below are historical snapshots, not current next-step instructions.

Removed the six dry mixins. DryHydrostaticDynamics directly inherits nn.Module;
99 ordinary equation functions are explicitly bound in its class body.
Shared dry/conservative/moist numerical expressions and defaults are unchanged.
335 fixture tensors match exactly; 66 tests pass, including small dry/moist IMEX
checks. These checks do not establish production implicit performance or stability.

Next: define the small public model/state boundary and recognizable-variable
tensor initialization, preserving the current kernel and bounded amount ledger.
Then add explicit xarray adapters, including independent surface pressure and
topography DataArrays. Do not silently regrid or detach tensors. Legacy optional
modules and Field/state/configuration infrastructure still ship.
Checkpoint 54, pass 1, and pass 2 remain unchanged.

---

# Current handoff — extraction pass 2

Read [pass-2 results](../docs/PASS2_RESULTS.md), then [README](../README.md).
Pass 1 history follows below. Original checkpoint 54 and pass 1 are unchanged.

Current: default atmospheric imports reduced from 63 to 29 modules; optional
named exports resolve lazily; wildcard exports intentionally narrowed; learned
PGF imports only on selection. BoundedGibbs is now one explicit class, retaining
all update expressions. 142 tensors match exactly; 33 focused tests pass.

Do not describe this as a completed package downsizing: optional code still
ships and six dry mixins/state configuration remain. Next dedicated pass should
extract those dependencies with focused fixtures before public initialization
API work. No numerical defaults or implicit algorithms changed.

---

# Current handoff — extraction pass 1

Read [results](../docs/PASS1_RESULTS.md), then [README](../README.md).

Implemented: bounded moist extraction into four package modules; preserved
existing package numerical files byte-for-byte except adding an explicit export
in atmosphere/__init__.py. A single package tree is installed; no research tree
is shipped or imported. The old broad package contents remain for later cleanup.

Public entry: `from vfs.atmosphere import BoundedGibbs`; instantiate with
`MoistConservativeDynamics()` and `release_as_heat=True` for the campaign baseline.
Use its own `step(state, dt)`. Generic explicit integration of its dt-aware RHS
is not a substitute for the cancellation-safe species Euler update.

Next pass: reduce inherited/root import scope and simplify the dry/kernel
structure incrementally. Preserve old baseline fixtures and add coverage when
changing another path. The friendly MoistAtmosphere/physical-field constructor,
xarray/DataArray adapters (including pressure/terrain), and regridding remain
planned work. TrajectoryWriter/averaged output is deferred. Existing implicit
methods remain experimental and unchanged. User will set up GitHub after a few
passes; no GitHub operation or public release occurred here.

User priorities: teachable, hackable, inspectable, object-oriented, research QOL;
geometry/meshes, dry and moist atmosphere, explicit/implicit methods, tensor
conservation diagnostics and explicit I/O. No column-physics integration now.
