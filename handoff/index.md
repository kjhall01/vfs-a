# Current handoff — approved SSPRK3 extraction (#16)

PRs #13 and #14 are merged. Numerical baseline remains
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`. The sole runtime change is the
approved stage-sequence extraction at
`ae132773e621671f70bfa6a698eebb1a82cffa94` in `bounded_moist.py`.
Read [extraction evidence](../verification/SSPRK3_EXTRACTION_RESULTS.md).
The atmospheric wrapper still validates and returns the same state and ledgers.
The manifest verifier accepts only the original blob or this exact approved
blob, rejecting all other runtime modifications. Keep #16 unmerged for review.

Next: dependent C5 prescribed horizontal transport harness using actual timed
production Euler maps, not copied RK formulas or invented candidate fluxes.
Replace prescribed wind at all RHS times and final output; evolve carrier,
entropy and species. Record discrete divergence and reference quadrature error.
Keep terrain-rest construction queued afterward. No coupled energy, GPU,
IMEX backward or long-rollout qualification follows from this work.

---

# Historical handoff — verification increments 1a and 1b

Baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648` (merged PR #12).
PR #13 is increment 1a: geometry/provenance and the full C1–C12 tracker.
Its updated head e657729 passed both required hosted runs and remains unmerged
for review. No numerical package source, fixtures or tolerances changed.

Increment 1b is developed separately, dependent on #13. Read the
[tracker](../verification/README.md), [1b methods](../verification/INCREMENT_1B.md)
and [1b results](../verification/INCREMENT_1B_RESULTS.md).
It adds scalar-energy AD/analytic/FD sweeps (C2), component work matrices including
fixed-lid reservoir and bounded heat (C3), and observation of the three actual
production Euler stages with independent incidence/inventory audits (C4).
The shared manifest approach archives drivers, geometry, prognostic inputs,
directions and actual stage diagnostics. No RK formulas are duplicated.

Small 16/3 and 32/4 cell/layer CPU float64 cases run first. Local installed-wheel
suite: 548 passed, one CUDA skip. Broad campaign qualification remains pending.
No IMEX backward, rollout or GPU support is inferred from these diagnostics.

Next queued work is prescribed horizontal/vertical/combined transport (C5), then
physically consistent terrain-rest construction (C6). Do not treat the current
nonuniform-terrain diagnostic states as balanced rest. C10 integrated budgets,
C11 Held–Suarez statistics and C12 repeat/restart equivalence remain explicitly
pending. Numerical defects require separate reproductions/proposed fixes;
keep the runtime frozen, and do not silently expand benchmark physics.

---

# Previous handoff — implementation freeze for verification

Final bounded cleanup follows merged PRs #9–#11. Read the concise
[README](../README.md), [cleanup results](../docs/FINAL_CLEANUP_RESULTS.md), and
[verification handoff](../docs/VERIFICATION_HANDOFF.md). Conda packaging is deferred.
Recommend freezing the implementation after review, not another stylistic refactor.

IMEX has only the analytic ARK4 operator: obsolete operator/compile fields are
removed, default configs work, and public steppers preserve their effective
solver settings. Bounded moist heat release is intrinsic; release_as_heat is
removed from constructors, with its inactive branches deleted. Tensor-only
numerical paths no longer use v(x); standalone Field boundaries remain intact.
The pinned Held–Suarez archive comparison is now a committed developer tool.
No equations, stage formulas, species ledgers, boundary accounting or fixtures
were changed. Class-instance ownership and gradient connections remain.

Installed-wheel local CPU float64: 536 passed, 1 CUDA skip; portable tools pass
235 retained tensors. Existing 298 pre/post captures and full solver records,
15 additional default-step/bounded-gradient tensors, and 132 Held–Suarez archive
comparisons are exact. All explicit/ARK4/forcing example commands run; the moist
ARK4 example uses smooth positive composition, not the bounded sharp-cloud case.
Frozen exact/hosted outcomes are reported separately in results and the PR.

The next phase needs reproducible benchmark drivers, references and convergence
metrics. JW initialization and HS forcing exist; topographic rest requires a
consistent pressure setup, standalone dry tracer transport is absent, and a
hydrostatic mountain-wave case needs specification/initialization and attention
to the fixed lid and absent absorbing/open boundaries. No new benchmark physics
was added. IMEX backward, long-rollout/GPU and general limiter-switch derivatives
remain unqualified. Report missing capability work separately from verification.

---

# Historical handoff — readable retained atmospheric paths

The three dependent readability PRs start from merged PR #8: #9 removes obsolete
public paths; #10 consolidates retained equations and ownership; the final slice
reconciles current guidance. Read [current paths and timestep traces](../docs/CURRENT_PATHS.md),
[state/model examples](../docs/STATE_OWNERSHIP.md), and
[changes and evidence](../docs/READABILITY_RESULTS.md). Earlier reports are indexed
under [history](../docs/HISTORY.md), not current API instructions.

DryAtmosphere owns conservative dynamics with DrySSPRK3 or DryIMEX instances.
Moist default is bounded SSPRK3 with heat; centered ARK4 is an explicit choice
without species bounds. Required centered/donor/limiter ingredients and wave,
solver, preconditioner and transfer infrastructure remain. External dry method
binding/recipes are gone. ScalarTransportState names shared entropy/theta contents
without dry thermodynamic disguise, conversion or hidden state ownership.

Local installed wheel: 519 passed, 1 CUDA skip; both portable tools (235 retained
quantities) and examples pass. All 298 pre/post tensors and complete ARK4 solver
diagnostics exactly equal merged PR #8. Fixture files and retained tolerances
are unchanged; 100 removed canonical-core paths are explicitly retired. Frozen
exact mode fails separately on this macOS CPU. GitHub outcomes are recorded in
the results and PR descriptions, not inferred from local checks. No IMEX backward,
full-rollout or accelerator qualification follows from this refactor.

Held–Suarez thermal relaxation and Rayleigh drag remain as a standalone
[source](../docs/HELD_SUAREZ.md), with a runnable forced-step example and separate
thermal/drag budget contributions. Explicitly sum it with conservative dynamics
at each SSPRK3 stage; model(state, dt) alone remains unforced. No registry,
process framework, legacy dycore or forced IMEX support is restored. Local updated
wheel: 527 passed, 1 skip; both portable tools and all three examples pass.
Original forcing method ASTs and 132 tensor comparisons are exact, including
forced stages, energy and representative gradients. Hosted results are in PRs.

Next specific readability work, if requested: inspect the remaining dry State
uncached diagnostic/I/O conveniences and clarify their model/external boundaries;
review long analytic pressure-derivative blocks against their adjoint expressions.
Do not combine that work with numerical changes, performance optimization or a
scientific verification campaign. Development remains on GitHub.

---

# Historical handoff — scientific readability, supported-path scope slice

Start from merged PR #8 (849ccb8). Read [the pre-edit map](../docs/SIMPLIFICATION_MAP.md)
and [current paths](../docs/CURRENT_PATHS.md). First branch simplify/supported-paths
removes pressure-prototype solvers, ARK2/3 selectors and CenteredSSPRK3 model
selection; shared ARK4 machinery moves to dry_ark.py/tableau.py without changes
to the retained formulas. Bounded explicit remains default, release_as_heat=True;
centered ARK4 is explicit opt-in and has no bounded species guarantee.

Capture tools/capture_readability.py records retained outputs/stages/ledgers and
gradients outside fixtures. Same-environment pre/post: 298 tensors and complete
solver diagnostics exactly equal. Local installed wheel: 601 passed, 1 CUDA skip;
335 portable fixture tensors and both examples pass. IMEX backward is untested.

Next dependent implementation work: simplify dry equation ownership and remove
superseded dry recipes after extracting required helpers; replace entropy in
mtheta_i with a named shared scalar-transport input. Preserve the baseline capture
and constants/arithmetic/gradients. Final PR reconciles current and historical
guidance. No numerical or performance campaign.

---

# Historical handoff — tensor-only State ownership

PR #7 is merged. The focused branch `cleanup/state-tensor-ownership` makes dry
State/PrognosticFields expose plain tensors, matching the existing MoistState
boundary. Compatible Field inputs unwrap by identity; standalone Field remains
for placement-aware mesh/regridding. Dry IMEX accesses were adjusted mechanically.
No operators, arithmetic order, fixtures, tolerances, bounded ledger, moist
steppers or private entropy-adapter method bodies were changed.

Read [ownership rules, dry/moist examples and validation](../docs/STATE_OWNERSHIP.md).
Models own geometry (including column topography), dynamics and stepping setup;
states borrow geometry and share prognostic tensors. Dry model binding now rejects
unrelated geometry. Explicit state_to_model preserves gradients; mismatched
stepping dtype/device and implicit tracer casts are rejected. Derived dry fields
are available through model.physical_fields(state), with legacy State diagnostic
methods retained as uncached compatibility helpers. Xarray stays external.

Local installed wheel after PR review: 603 passed, 1 CUDA skip; both portable tools and examples
pass. All 335 capture tensors plus 47 additional forward/gradient tensors are
exactly equal pre/post on the same CPU environment. Frozen-fixture exact mode
still fails separately on this macOS host. No new accelerator/full-rollout claim.
GitHub check status is recorded separately in the PR. Development remains on GitHub.

PR #8 review follow-up fixes coarse Schur tracer restriction to use axis -3 for
[..., cell, full, tracer]. Three regression cases exercise actual transfers
with zero, one and two batch axes and check exact weighted values/shape without
input mutation. This covers coarse construction, not a complete batched solver.
Retained consumers were audited; other Field metadata accesses serve explicit
Field operations or schema metadata, not state tensors. Operators are unchanged.

Remaining legacy boundaries: private entropy in mtheta_i, dry State convenience
methods/attrs/restart shell, and the older dry recipe configuration. Keep future
cleanup of those interfaces separate from numerical changes and verification.

---

# Historical handoff — bounded tensor tolerance and one-step gradients

PR #6 was merged after confirming both required CPU checks at its final head.
The separate branch `fix/bounded-tensor-tolerance`,
[PR #7](https://github.com/kjhall01/vfs-a/pull/7), changes only bounded_transfer's
flux-sum validation to the equivalent tensor-valued inequality, preserving the
128*eps scale, finite-error rejection, operators, defaults and species ledger.
There is no detach, disabled-autograd workaround or limiter redesign.

Read [the fixed gradient failure, measurements and remaining limits](../docs/BOUNDED_GRADIENT_LIMITATION.md).
Backward now completes through one bounded SSPRK3 step for wind, ps, entropy
and water. Smooth CPU float64 directional checks use three finite-difference
sizes with inactive limiter margins. A sharp-cloud case also agrees along the
tested support-preserving directions with stable active masks and subcycle counts.
A controlled switch has different left/right derivatives: autograd follows the
selected branch, not a globally smooth limiter. The donor-count scalar-conversion
warning remains. No further backward exception appeared in these probes.

Installed-wheel results and any hosted CI evidence are recorded in the linked
document, separately from frozen-fixture exact failures. Same-environment
pre/post forward captures are exact; fixtures and portable tolerances are unchanged.
No full-rollout, arbitrary-switch, higher-derivative, IMEX-backward or accelerator
backward qualification is claimed. Positive vapor remains required; no floor,
new physics, integration scheme or scientific verification campaign was added.

Next work must distinguish fixed-branch derivatives from switch behavior; do not
turn this narrow fix into a limiter redesign or broad backward-support claim.
Development remains on GitHub.

---

# Historical handoff — moist module ownership and explicit steppers

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
