# Current handoff — moist module ownership and explicit steppers

PR #5 is merged. The ownership branch is `api/moist-module-ownership`, published as
[PR #6](https://github.com/kjhall01/vfs-a/pull/6).
Read the [module API](../docs/MOIST_MODULE_API.md) and
[validation results](../docs/MOIST_MODULE_RESULTS.md).
MoistAtmosphere registers mesh, column and dynamics once; dynamics owns kernel.
BoundedGibbs and the explicit stepper instances are plain non-owning collaborators.
Components cannot be replaced/deleted; borrowed references are checked before
advancement. IMEX binds its solver once and retains per-step wave construction.

Default bounded SSPRK3 still delegates to the exact existing Euler amount ledger
with release_as_heat=True. CenteredSSPRK3 and CenteredIMEX (existing ARK2/3/4
objects) are explicit opt-ins. There is no bounded IMEX option, generic RHS
substitution or duplicated RK formula. Centered IMEX rejects inadmissible stages;
it does not provide bounded species transport. Positive vapor remains required,
with no new floor. Private entropy adaptation and dry APIs remain unchanged.

Move the module before initialization. state_to_model explicitly converts only
prognostic tensors on identical shared geometry, retaining gradients. state_dict
is not a simulation restart: geometry layout and numerical/stepper configuration
must be reconstructed separately. See the API for supported ownership operations.

The [pre-existing gradient limitation](../docs/BOUNDED_GRADIENT_LIMITATION.md)
blocks gradient-enabled bounded timesteps even though diagnostics and conversion
retain gradients. Its allclose assertion remains unchanged; any fix must be a
separate narrow change. Local CPU float64 forward regression, short diagnostic
and centered one-step gradient probes, and one MPS float32 smoke are distinct
results, not full-rollout/accelerator numerical qualification. CUDA is unavailable.

Next: review this ownership PR and, separately, address the allclose scalar
validation failure without changing tolerance semantics or hiding gradients.
No new physics, integration scheme or scientific verification campaign belongs
in this change. Development remains on GitHub.

---

# Historical handoff — centered moist readability

`moist_conservative.py` now expands packed statements, uses consistent formatting
and names energy/transport intermediates explicitly. Follow the
[updated reading guide](../docs/BOUNDED_MOIST_READING_GUIDE.md#inside-the-centered-moist-base)
from tendency through the private adapter, mixture energy gradients, matched
momentum assembly and total-energy diagnostics. Function boundaries specify
units and staggering; familiar mathematical symbols remain defined locally.

The private State.mtheta_i buffer still aliases MoistState.entropy_content.
Inherited theta accessors mean specific entropy only on this adapter; it must
not escape to dry thermodynamic diagnostics or I/O. No tensor copies/detaches,
state redesign, new helper layer or cache were introduced. Arithmetic, evaluation
order, defaults, gradient connections and fixtures are unchanged. The bounded
wrapper's species amount ledger, release_as_heat=True and strictly positive
vapor requirement remain intact; no new floor or verification campaign is added.

The guide records local and hosted validation separately. Next: document the dry
RHS call order and wider State/Field ownership. Any adapter redesign or reuse of
intermediates needs a separate numerical/gradient review. CPU float64 checks do
not qualify GPU, long-run or full backward behavior.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — bounded moist readability

Read [the bounded-step guide](../docs/BOUNDED_MOIST_READING_GUIDE.md) for the
execution order from MoistAtmosphere.step through SSPRK3, Euler transport,
Gibbs traces, species ledger limiting and heat allocation. Source comments now
state intermediate shapes, units, staggering and conservation roles; descriptive
locals distinguish Gibbs correction, limiter correction and RK stage states.

This is comments/docstrings/local renaming only. No operator, expression,
operation order, default, state convention, fixture or exact/portable policy
changed. Euler water still comes directly from the exact species amount ledger.
MoistAtmosphere retains release_as_heat=True and strictly positive input vapor;
no state floor is added. Existing transfer lower bounds remain unchanged.

The guide records validation and the boundary for any later structural work.
No scientific verification campaign, GPU/long-run/full-backward qualification,
new abstraction layer or caching/reorganization is included. Next readability
work remains the dry RHS call order and State/Field ownership guide; any
structural simplification needs its own numerical/gradient review.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — atmospheric package scope

Development remains on GitHub. This scope-reduction PR removes ocean, land, ice,
surface-process, coupling, coupled graph orchestration and ML-specific code from
the distributed package. See [scope/migration/results](../docs/SCOPE_REDUCTION.md)
and [dependency trace](../docs/SCOPE_DEPENDENCIES.json). Removed code remains in
Git history; there is no second legacy package.

Retained: spherical geometry/mesh generation, dry/moist dycores, explicit and
existing experimental implicit integrators, conservation/differentiable
diagnostics, atmospheric initialization and I/O. Shared State/Field/configuration,
model/process support, ETOPO terrain ingestion and IMEX mesh hierarchy remain
because retained paths use them. The obsolete surface dependency extra is gone.

Numerical expressions/defaults, fixtures and integrators are unchanged. The
bounded amount ledger, release_as_heat=True MoistAtmosphere configuration and
strictly positive vapor requirement remain; no floor is added. Portable is the
required installed-wheel gate; exact is separate and retains failure status.
CPU float64 results do not qualify GPU, long runs or full backward rollouts.

Next: explain the dry RHS method index/call order, document State/Field ownership,
and annotate the bounded Euler/Gibbs/heat ledger sequence. Keep those focused
readability changes separate from kernel changes or scientific verification.
Earlier handoffs below are historical snapshots.

---

# Historical handoff — GitHub development and CI

Development is now on https://github.com/kjhall01/vfs-a. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the small test extra, local setup,
installed-wheel checks and focused PR preparation. The new GitHub Actions
workflow targets Ubuntu CPU/Python 3.12/PyTorch 2.15.0+cpu and runs the suite,
both baseline tools and examples/moist_atmosphere.py outside the checkout.
Portable mode is required in the full pytest suite and both tools. Exact mode
remains the default locally and runs separately in CI with failures reported
explicitly as informational outcomes. The public Linux CPU wheel was successfully downloaded.

See [CI setup results](../docs/CI_SETUP_RESULTS.md) for the initial local checks.
GitHub commit 911aada subsequently had one passing and one failing run with
identical recorded versions/source trees. The
[CI investigation](../docs/CI_911AADA_INVESTIGATION.md) includes all 335 tensor
differences from four Linux CPUs and macOS, controlled dispatch experiments,
and the reviewed quantity-specific portable tolerances. The
[portable implementation](../docs/PORTABLE_REGRESSION.md) now uses an exhaustive
path manifest, separate species scales and independent physical/discrete/ledger
checks. See [validation results](../docs/PORTABLE_REGRESSION_RESULTS.md) for
local and hosted outcomes separately. Exact comparison expressions remain
unchanged. Original CPU models were not logged.
The separate audit/ci-911aada branch collects diagnostic evidence only.
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
