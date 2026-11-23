# Pass 4 — physical-field workflow

## Full sea-ice coverage in the day-11 campaign

At second 1028400 (day 11, 21:40), the reported 4,096-cell run rejected the post-exchange ice phase update. When freezing fills the open water, multiplying both category footprints by the same scaling factor can round their sum up to 1.0000000000000002 in float64. The original error displayed that value as 1.

The revised two-category geometry retains the smaller fraction and assigns the remaining footprint to the larger category if the ordinary calculation overshoots. The phase-energy root, its derivative and the accepted state all use this geometry. The strict coverage and store checks remain active. A future rejection prints the offending cell, its two areas and volumes, and the excess coverage at full precision.

The original failed checkpoint was unavailable for this verification. On the earlier day-10 checkpoint, a controlled 0.150 K cooling of the top ocean at cell 22 reproduced the same zero-flux thermal phase failure. At the corrected endpoint, one of 8,192 category areas differed from the rejected endpoint by one binary64 rounding increment (1.11e−16). The cell coverage was exactly one. The tested ice volumes and velocities, ocean layer thicknesses, temperature and salt inventories, edge velocities, TKE and eddy stores matched bit for bit. The maximum liquid-plus-ice water residual was 9.09e−13 kg/m².

The installed-wheel ice, snow, ridging and thermal workflow checks passed 82 tests on CPU, and the dry and corrected-Gibbs fixtures passed 93 and 142 portable tensor comparisons. The small native coupled restart test passed from the prior day-11 limiter source and again from the revised source. Three historical PyAM21 endpoint source-pin tests fail on both this experiment and its parent because they expect an older source inventory.

From the `fix/day11-ice-phase-coverage` checkout in the original coupled environment, run `python -u -I tools/resume_gm_ice.py`. It searches for the unique matching failed run under the Desktop and VFS run folders, reads its original configuration and manifest, and targets model day 30 in a new sibling directory ending in `-ice-fix`. If the run is elsewhere, use `--resume /path/to/last_good_restart.pt`. Supply `--source` or `--radiation-data` if the original PyAM21 checkout or Sea ESF tables are elsewhere than `~/Desktop/PyAM21` or `~/Desktop/sea_esf`.

## Day-11 donor-parcel GM with local limiting

On branch `experiment/gm-parcel-limiter-day11`, run this from the checkout in
the original `vfs-joint` environment:

```sh
python -u -I tools/run_gm_day11.py
```

The launcher reads `~/Desktop/vfs-a/gm-parcel-day30/last_good_restart.pt` and
the adjacent `manifest.json`, verifies step 780 at second 936000, and runs 12
coupled intervals to second 950400. It uses the original configuration and saved
mesh, `~/Desktop/PyAM21`, and `~/Desktop/sea_esf`. Use `--help` for path
overrides. Results go to `~/Desktop/vfs-a/gm-parcel-limiter-day11`.

GM attempts the donor-parcel flux at its original 60-second substep. If local
eddy energy cannot fund the work, the limiter reduces the streamfunction on
interfaces adjacent to the depleted cell-layer and recomputes the fluxes.
`gm_transport_audit.jsonl` records each accepted substep, the location of any
failed trials, and the proposed and accepted transport differences.
`gm_limiter_events.jsonl` records limiter activations. The native campaign
writes diagnostics, restarts and `compatible_resume_audit.json`. The
experimental source is included in the new restart fingerprint.

On Linux CPU float64, both ocean–ice legs from the original checkpoint
reproduced the earlier replay byte for byte across all 20 substep audit rows,
four limiter events and both leg reports. The strict operator failed at
cell 3744, zero-based layer 58. A small coupled radiation-off preflight
exercised the native restart bridge. The full day-11 run with Sea ESF
radiation is the experiment to run on the original machine.

## Experimental day-9 GM replay

Run `python -u -I tools/run_gm_day9.py` from the parcel-energy experiment
checkout in the existing coupled environment. Use `--help` for the input and
output overrides. The launcher starts from second 692400 and targets second
777600. It loads the checked-in GM operator, verifies the exact parent and
candidate GM file hashes, and lets the native campaign validate the mesh,
radiation, configuration, runtime, source inventory and checkpoint identity.
The experiment writes restarts with the candidate fingerprint.

The previous standalone launcher compared a resolved Python module path to a
temporary directory path that could contain a symlink. A macOS-style temporary
directory alias reproduced the guard failure on Linux even when the correct
file was imported. The repository launcher verifies the physical files and can
retry into an empty output directory. It preserves directories containing
earlier results. A completed limiter run supplies the reference metrics for
`day9_comparison.json`. If its default location is absent, the candidate still
writes normal campaign outputs.

The donor flux passed the 20 GM/Redi substeps in the original failing ocean
leg at 60 s on Linux CPU float64. The 49 targeted GM/Redi, feedback and
reservoir cases passed outside the checkout using an installed wheel.
The corrected-Gibbs and dry-helper portable fixture comparisons passed (142
and 93 tensors). The dry helper also matched exactly. On this runtime the
first exact atmospheric mismatch was an initial velocity difference of
4.44e-15. Full coupled day-9 behavior with the production radiation tables
remains to be tested on the original runtime.

## Cloud-fraction cancellation in the coupled campaign

A 4,096-cell, 32-atmosphere-level, 60-ocean-level CPU float64 campaign stopped
at second 231,600 in the atmospheric endpoint check. The saved checkpoint
contains 193 committed intervals. The radiation driver requested new shortwave
and longwave calculations for the next interval.

A Linux replay used the saved atmosphere, ocean, ice, and snow with the
previous interval’s cached optical calculation. At cell 2289, level 29
(zero-based), the input cloud fraction was about 0.174. Vertical diffusion
reduced it to about 0.125. RAS added about 160.416, and stratiform cloud physics
brought the fraction back to 1. The RAS intermediate is an extreme cloud-area
excursion. Its cloud-top detrainment term produces that increment before the
stratiform update. The adapter continues to use the upstream physical endpoint.

With the saved optical cache and an input cloud-fraction perturbation of
64 float64 epsilons in one column, the published net tendency reconstructed
0.9999999999999705. The explicit endpoint was 1. The original check rejected
their absolute difference of 2.9532e-14 against a limit of 2.8422e-14.
The fixed adapter verified the actual diffusion, RAS, and stratiform arithmetic.
The exchange completed and the new state packed and restored at 232,800 s
after post-exchange phase adjustment and ALE on all 4,096 columns. The
unperturbed eager and compiled controls using the cached optics also completed
the exchange. The original Mac interval with its freshly calculated SeaEsf
radiation and the earlier L4 failure require separate on-device verification.

The restart bridge compares the old checkpoint with the original manifest and
records the source transition. Use the original mesh, configuration, radiation
tables, and execution environment when continuing a real campaign.


## Endpoint/tendency failure diagnostic (2026-11-18)

User logs report two-day real-table float64 completion on T4 eager and L4 with
regional Inductor. The latter continued from its day-two restart and rejected
the next interval after second 249600 at the explicit PyAM21 endpoint/tendency
consistency check. This does not establish whether the cause is roundoff,
upstream physics or device-specific behavior. The actual failed state has not
been reproduced locally.

The new standalone `tools/diagnose_pyam21_endpoint.py` preserves the complete
configuration, restart identity and numerical checks, capturing per-field
discrepancies and selected columns' moist-process inputs only on this failure.
It leaves the original rejection active and its replay status failed. The tool,
tests and Markdown documentation do not alter campaign-hashed Python source;
existing checkpoints can be diagnosed on their original runtime. No numerical
fix, tolerance relaxation or production restart migration is introduced.

Validation outside the checkout against installed VFS: 41 endpoint/workflow
tests pass, including unchanged successful physics and a deliberately inconsistent
endpoint that remains rejected and produces a CPU-loadable moist replay bundle.
Frozen portable comparisons pass for 142 corrected-Gibbs and 93 dry-helper
tensors. All fingerprinted Python files match the user's `33e888c` source
byte-for-byte. Actual L4 failure capture remains pending.

## Coupled dtype/MPS preparation (2026-11-17)

The single-GPU draft now accepts CPU/CUDA/MPS and float32/float64, including
CLI/YAML, physics state/cache ownership, restart storage, timing and comparison
drivers. MPS requires explicit float32; half precision and silent CPU fallback
are rejected. Setup restores the caller's default dtype. CPU float32 eager and
native compiled all-component tests run, while GPU cases remain unexecuted.
Two impossible sub-ulp FP32 checks were given explicit dtype-aware defaults:
lagged mixing budget `max(1e-9,32*eps)` and midpoint drag `max(1e-12,32*eps)`.
No physical process is disabled. Float64 keeps the original thresholds and
matches 240 tensors/127 metadata leaves against the merged base bitwise.

Float32 is experimental: after four default-cadence intervals at 32 cells,
maximum specific eddy-energy difference is 0.0753 J/kg and native numerical
energy residual is 2781.84 J/m², versus -0.1284 J/m² for float64. Native budgets
include diagnostic rounding. See [measurements and tests](../verification/atmosphere/pyam21/campaign/precision/README.md).

## Coupled single-device CUDA preparation (2026-11-17)

After PR #90 merged, the coupled workflow gained explicit CPU/CUDA float64
placement, device/host-thread CLI switches, CPU-stored restart tensors restored
onto the model device, deterministic CUDA replay policy, hardware/runtime
provenance and synchronized comparison timing. Physical settings and packaged
numerical kernels are unchanged. CUDA is a single-device path; no component
parallelism is claimed. The upstream PyAM21 revision is unchanged.

Installed-wheel focused validation: **110 passed, 2 CUDA skips** with native
CPU Inductor enabled; frozen portable **142/93** comparisons pass. Against the
merged CPU base, a four-interval 32-cell/default-level/default-timestep analytic
radiation campaign agrees bitwise across **240 tensors and 127 scalar leaves**.
CUDA tests and a full real-table CPU/CUDA comparison command are provided but
cannot be executed on this CPU-only host. See
[evidence](../verification/atmosphere/pyam21/campaign/cuda/README.md) and
[GPU setup](../workflows/coupled/GPU.md).

> Current numerical-fix evidence: [paired Gibbs revision](../verification/GIBBS_FIX_RESULTS.md).
> Historical fixtures below remain unchanged; the proposed revision fails their moist gates.
>
> Historical report. For current APIs and supported configurations, read
> [retained timestep paths](CURRENT_PATHS.md) and [readability results](READABILITY_RESULTS.md).

Added a small MoistAtmosphere facade using composition around the existing
MoistConservativeDynamics and BoundedGibbs. It holds mesh/column, constructs the
existing MoistState from physical fields, projects edge-sampled east/north wind,
recovers interface fields, advances bounded SSPRK3 and returns tensor budgets.

No numerical kernel file was changed. Only root/atmosphere exports and local
package version changed in existing package sources. New implementation:
vfs/atmosphere/model.py. Guide: PHYSICAL_FIELDS_API.md.
Runnable example: examples/moist_atmosphere.py.

## Validation

Installed wheel tested outside checkout, PyTorch 2.15.0+cpu, CPU float64.
78 tests passed in 23.47 s (see tests_pass4.log).
142 checkpoint-54 tensors match exactly (equivalence_pass4.json).
The suite also compares 193 dry helper fixture tensors exactly and checks all
99 extracted dry helper ASTs. Thus all 335 existing fixture tensors still match.

New tests cover physical-field round trips, exact agreement with manual state
construction and the underlying bounded step, wind projection, invalid shape/
dtype/nonfinite/physical inputs, explicit rejection of external numpy arrays,
geometry mismatch, zero condensates, zero-vapor rejection, and finite-difference
agreement of energy gradients with respect to temperature, pressure and vapor.

The 64-cell, 6-layer example completed 10 steps of 30 s:
- Relative total mass change: -6.217176653065638e-16.
- Species inventory changes: [-24, -1, 0] kg, on planetary total inventories.
- Energy change: -1.542650926958044e-5 J/m².
- Minimum vapor/liquid/ice: [0.009999999995771525, 1.4474766504931322e-50, 0].
These are illustrative diagnostics, not new timestep guidance or a balanced
benchmark; no scientific acceptance thresholds are asserted.

## Existing limitation exposed

The first new dry-limit test failed because the unchanged moist adapter
requires vapor strictly greater than zero. Initialization now requires an
explicit vapor input and rejects zeros early with an explanatory error.
No floor, numerical change or automatic dry-core substitution was introduced.
Exact-zero-vapor dry equivalence has NOT been established. Liquid/ice zeros
are supported. This remains a separate numerical question for later work.

## Scope

The model is a thin Python facade, not an nn.Module with automatic device moves.
Autograd is retained in tensor initialization and diagnostics; this pass does
not qualify full backward rollouts, GPU behavior or every limiter branch.
Temperature/composition must already be at CP interfaces; winds at layer-center
edges. No interpolation, regridding or phase equilibration is hidden.

Xarray/DataArray adapters, output/restarts and the friendly dry facade remain
future work. Legacy optional modules still ship. This snapshot is suitable for
moving development to GitHub; public release metadata/license remain undecided.
Original checkpoint 54 and passes 1–3 are unchanged.

## Subsequent CI regression policy

The [portable policy](PORTABLE_REGRESSION.md) adds a separate installed-wheel
regression mode without changing pass-4 numerical code or fixtures. The
[CI investigation](CI_911AADA_INVESTIGATION.md) retains the CPU/backend evidence;
[validation results](PORTABLE_REGRESSION_RESULTS.md) distinguish portable and
exact outcomes. Historical exact results above describe their original environment,
not a cross-platform bitwise guarantee.

The later [scope-reduction PR](SCOPE_REDUCTION.md) deliberately removes non-atmospheric
and ML APIs while preserving this physical-field workflow and its numerical baseline.

## Subsequent module-boundary work

The [module API](MOIST_MODULE_API.md) owns mesh, column and dynamics once,
with explicit non-owning steppers delegating to the existing bounded SSPRK3,
centered SSPRK3 or experimental centered IMEX algorithms. See the separate
[validation results](MOIST_MODULE_RESULTS.md) and [bounded gradient limitation](BOUNDED_GRADIENT_LIMITATION.md).
These do not turn historical CPU forward results into accelerator or rollout
qualification. Centered IMEX does not provide bounded species transport.

The subsequent [tensor-tolerance fix](BOUNDED_GRADIENT_LIMITATION.md) preserves
forward results and adds scoped CPU float64 one-step directional derivative
checks. It fixes the allclose argument error without qualifying full rollouts
or differentiability at limiter switches.

## Subsequent State/Field ownership cleanup

See [the ownership guide and validation](STATE_OWNERSHIP.md). Dry prognostic
attributes now return tensors; standalone Field retains its mesh/regridding role.
Dry geometry mismatches and implicit tracer casts are rejected, explicit
conversion preserves gradients, and model methods expose uncached physical fields.
Local installed-wheel portable tests: 600 passed, 1 CUDA skip. Both comparison
tools and examples pass; 382 same-environment capture tensors agree exactly.
Frozen-fixture exact failures remain reported separately. No numerical operator,
fixture, moist stepper or private entropy-adapter body changed.

PR #8 review follow-up: coarse Schur tracer restriction now uses tensor axis -3
instead of Field metadata. Three new cases verify restricted values and shape
with real transfers and up to two leading batch axes. Local installed-wheel
portable suite: 603 passed, 1 CUDA skip; both standalone comparisons and examples
pass. No operators, transfer weights, fixtures or tolerances changed. See the
ownership guide for the audit and the limited scope of batched coverage.

## Subsequent supported-path scope slice

Merged PR #8 is the baseline for [the scientific-readability plan](SIMPLIFICATION_MAP.md).
ARK4-only public IMEX selection preserves 298 captured tensors and all stage solver
diagnostics exactly. Installed-wheel portable suite: 601 passed, 1 CUDA skip;
335 frozen fixture tensors and both examples pass. See CURRENT_PATHS.md for
explicit/implicit guarantees and the precise representative-case evidence.

## Subsequent retained-equation consolidation

See [readability changes and validation](READABILITY_RESULTS.md). The dry function
binding/recipe stack is replaced by ordinary conservative/energy methods and
DryAtmosphere with explicit steppers. A borrowed scalar transport view replaces
entropy-as-mtheta; bounded ledgers and formulas remain. ARK4 wave/stage/solver
owners are separated without new methods. Installed-wheel portable: 519 passed,
1 CUDA skip; both tools/examples pass, 298 same-environment pre/post tensors and
solver diagnostics exactly equal. Frozen exact failures remain separate. The
100 removed canonical-core fixture paths are retired explicitly, with fixture
bytes and retained tolerances unchanged. No IMEX backward qualification.

## Current guide reconciliation

Start with [retained paths](CURRENT_PATHS.md), [ownership examples](STATE_OWNERSHIP.md)
and [current regression evidence](READABILITY_RESULTS.md). Earlier reports and
readability/ownership measurements are preserved under [history](HISTORY.md).
The final guide slice changes no executable expressions; both usage snippets ran
against the installed wheel. PR #10 final-head hosted CPU checks passed; its push
artifact records 519 portable passes, 1 skip and separate exact failures.

## Standalone retained dry forcing

Before merging the simplification stack, preserve existing Held–Suarez thermal
relaxation and Rayleigh drag in [a standalone source](HELD_SUAREZ.md). No legacy
framework is restored. Original method ASTs and 132 same-environment comparisons
are exact, including forced-stage states and representative energy gradients.

## Final bounded cleanup and verification handoff

The [cleanup report](FINAL_CLEANUP_RESULTS.md) records removal of obsolete IMEX
configuration, intrinsic bounded heat release and redundant tensor wrappers,
with exact same-environment comparisons and unchanged fixtures. The committed
Held–Suarez archive tool and five supported scenario types are runnable. Freeze
implementation for [verification](VERIFICATION_HANDOFF.md); Conda is deferred.

## Subsequent structural ownership migration

See [package owners](PACKAGE_OWNERSHIP.md) and [migration checks](STRUCTURAL_RESULTS.md).
The structural baseline is merged main; #18/#19 numerical evidence is separate.
Historical fixtures and tolerances remain unchanged.

## Atmospheric integration/reference policy

See [the separate integration record](../verification/ATMOSPHERIC_INTEGRATION.md)
for #18/#19 integration onto reconciled main. Historical fixture files, hashes,
tolerances and original evidence remain unchanged.

The separately authorized `corrected-gibbs-v1` regression reference and integration
results are recorded in [the atmospheric stack report](../verification/ATMOSPHERIC_INTEGRATION.md).
Checkpoint-54 files, tolerances and historical reproduction commands are unchanged.

Ocean-only rotating-balance diagnosis after merged #26/#27: see
[the report](../verification/ocean/ROTATING_BALANCE.md). Atmospheric runtime,
fixtures and verification policy remain unchanged.

Ocean-owned bounded horizontal transport on merged JMD95 #31: see
[the derivation and measurement report](../verification/ocean/BOUNDED_TRANSPORT.md).
Installed-wheel portable suite: 659 passed, one CUDA skip. Existing runtime and
fixtures remain unchanged. An experimental active composition preserves scalar
bounds but reports unclosed limiter work; no bounded energy-closing active
stepper is promoted. ALE #32 remains a separate draft.

Dependent thermal-work research: see [the caloric derivation and entropy audit](../verification/ocean/THERMAL_WORK.md).
Installed-wheel portable: 680 passed, one CUDA skip. A declared caloric completion
closes instantaneous energy work continuously at zero velocity, but its simple
thermal allocation fails an entropy counterexample on tilted layers. A separate
thermodynamic exchange reference satisfies spatial energy/salt/entropy laws;
finite-step entropy is not inferred. All runtime files remain unchanged.

Dependent seawater-reference research: see [surface calorics and enthalpy coordinates](../verification/ocean/SEAWATER_CALORICS.md).
Installed-wheel portable: **697 passed, one CUDA skip**, including 17 new tests;
164 legacy ocean tensors exactly equal; atmospheric portable comparisons pass.
One surface Gibbs fit reproduces GSW caloric references with explicit scale
conversions, and an enthalpy-content research state retains JMD95 density with
analytic matched energy derivatives/forces. The first-law work identity and
separate entropy-producing exchange are independently checked. The tilted-layer
advective entropy failure remains; no bounded active runtime promotion follows.
All 100 runtime Python files, defaults, fixtures and tolerances are unchanged.

Dependent Gibbs/entropy candidate: see [the ocean transport derivation and evidence](../verification/ocean/ENTROPY_TRANSPORT.md).
Installed-wheel portable: **717 passed, one CUDA skip**, including 20 new tests;
164 legacy ocean tensors exactly equal; atmospheric portable checks pass. The
research entropy state, ocean Gibbs means, composition-attached entropy flux
and mechanical/nonnegative-heat work split remove #35's tilted entropy failure.
Per-incidence budgets avoid a force jump when one edge crosses zero while
neighbors remain limited. Active salinity must be strictly positive; scalar
admissibility is not a local-extremum theorem. Pressure and short internal-wave
refinement are measured; long runs/GPU/performance remain unqualified. All
installed runtime files, defaults, historical fixtures and tolerances are intact.


Frozen #36 qualification campaign: see [the methods and measurements](../verification/ocean/QUALIFICATION_CAMPAIGN.md).
Installed-wheel portable suite: **722 passed, one CUDA skip**, including five
new instrumentation/reference tests; all 164 legacy ocean tensors exactly
equal and atmospheric portable comparisons pass. Full nonlinear waves and
balanced flows are distinguished from isolated prescribed-carrier transport.
The campaign reports phase/envelope behavior, jet drift and sharp-front
oscillations without modifying the candidate or setting empirical accuracy
gates. Smooth transport is approximately second order; sharp transport does
not preserve local extrema; rotation/kinetic consistency remains unresolved.
All runtime code, defaults, fixtures and tolerances are unchanged.


## Sharp-front research pass after draft #37

See [SHARP_FRONT.md](../verification/ocean/SHARP_FRONT.md) and its four compact JSON
reports. Local FCT targets plus a kinetic-work guard remove measured sharp-front
SP/passive overshoots while retaining the parent Hamiltonian work identity and
single-edge zero-velocity continuity. Active local bounds are conditional on the
work guard; temperature can still exceed initial extrema. Smooth-profile convergence
slows to observed orders 1.6–1.7, and numerical entropy production is explicitly
accounted. Rotation/kinetic operators and all installed runtime files remain unchanged.

The two-hour active-front comparison at dt=30 s gives SP 34–36 and passive maximum
1 to roundoff, versus parent SP 33.743–36.171 and passive maximum 1.084. Halving dt
reduces energy drift from −43.05 to −5.52 J/m². These are short trajectory measurements,
not physical accuracy gates or long-run qualification.

## Validation

Installed-wheel tests run outside the checkout: **736 passed, one CUDA skip**.
All 164 legacy ocean integration tensors are exactly unchanged. Atmosphere portable
comparisons pass for 142 moist and 93 dry tensors. Fourteen new focused tests exercise
independent Hamiltonian derivatives/work, entropy accounting, local passive bounds,
global salinity admissibility and zero crossings with other faces still limited.
Historical exact archive differences remain separate; no fixtures or tolerances changed.


## Entropy ALE integration and vertical refinement

See [ENTROPY_ALE.md](../verification/ocean/ENTROPY_ALE.md). The research wrapper
composes #38's checked entropy/salt timestep with #32's bounded overlap remap.
Native entropy, salt, passive contents and edge momentum are conserved; nonlinear
energy and temperature-content changes are measured explicitly. No energy fix,
state clipping, runtime default change or atmospheric import is introduced.
The original full-period wave improves through 64 layers but retains an RMS
penalty (0.137 versus 0.0627 without remap); its energy defect is non-monotonic.
Short integrated entropy runs expose a persistent reference-thermal energy defect.
Installed-wheel suite: 763 passed, one CUDA skip; 164 legacy ocean tensors exactly
unchanged; both atmosphere portable comparisons pass. See the report for scope,
reproduction and actual-remap diagnostics. This is not long-run qualification.


## Potential-enthalpy ALE experiment after #39

[ENTHALPY_ALE.md](../verification/ocean/ENTHALPY_ALE.md) documents explicit
potential-enthalpy/salt remapping followed by reconstruction of the entropy state.
The ~63 kJ/m² thermal loss in the short wave experiment is removed to numerical
precision; total remap residuals remain +88/−132/−167 J/m² at 8/32/64 layers, mainly
dynamic enthalpy. Wave accuracy is measured separately using column salinity, which
both remap schemes conserve. This is a thermal-energy improvement, not full energy
closure or a claim of improved wave accuracy. Entropy changes are recorded explicitly.
Installed-wheel portable suite: 783 passed, one CUDA skip; all 164 legacy ocean
integration tensors exactly unchanged. No installed runtime/default/fixture/tolerance
changes; rotation and atmosphere remain frozen. All uploads below 128 KiB.


## Conditional local Hamiltonian ALE closure after #40

[HAMILTONIAN_ALE.md](../verification/ocean/HAMILTONIAN_ALE.md) derives local energy
ownership and a nonlinear thermal projection using the original Hamiltonian.
Short waves at 8/32/64 layers close remap energy to roundoff, with accumulated
residuals near 1e-4 J/m². Salt, momentum and passives remain conservative; finite-step
energy drift is not compensated. Wave errors are essentially unchanged.
Three weak stable stratification cases fail the column entropy condition and are
explicitly rejected. This is a conditional research method, not a universal ALE fix.
Installed-wheel suite: 799 passed, one CUDA skip; 164 legacy ocean tensors unchanged;
142 moist and 93 dry portable comparisons pass. No runtime/default/fixture changes.


## Adaptive thermal ALE allocation

The research candidate now permits different temperature corrections by layer,
with mass-weighted linearized minimum-change projections and full nonlinear
Hamiltonian/entropy checks. It rescues all three previous weak-stable rejection
cases with unchanged tolerances; all eight feasibility cases and 16 remap audits
complete. Matched 8/32/64-layer .05-period waves retain essentially unchanged
errors and roundoff energy closure. This does not establish universal feasibility
or nonlinear optimality. Installed-wheel tests: 816 passed, one CUDA skip; legacy
ocean 164 exact; moist 142/dry 93 portable comparisons pass. No runtime changes.
See [the derivation and measurements](../verification/ocean/ADAPTIVE_THERMAL_ALE.md).


## Selective stratified-ocean integration

The installed `vfs.ocean.stratified` API selects the entropy/Gibbs work pairing,
sharp-front transport and explicit adaptive Hamiltonian ALE correction. Legacy
Ocean/defaults remain unchanged. Superseded candidates are retained as evidence.
See [stratified API and supported scope](OCEAN_INTEGRATION.md). The standalone
example imports only the installed package and exercises both timestepping and
actual remapping. Exact migration comparisons cover states, stage ledgers and
remap reports across four profiles with/without coast masks. Seven numerical
module expression trees match their frozen research source apart from imports;
source hashes are retained. This is API integration, not new physics qualification.

## Current documentation

The README and current guides describe both components, their public APIs and
verification limits. Historical numerical reports remain separate and unchanged.
This documentation update changes no runtime code, fixtures or tolerances.

## Rotation/PV diagnostic audit

The read-only `tools/audit_rotation_pv.py` assembles independent circulation,
divergence and true primal/dual area transfers. Seven generated meshes and the
recovered 16,384-cell atmosphere mesh show a rotation compatibility defect. The
fine-mesh operator norm defect is 0.09803; a smooth synthetic uniform-PV test has
about 0.027 relative vorticity-tendency mismatch. Index/intersection ablations
reduce the geometric defect to floating-point geometric residuals, while a
separate variable-thickness absolute-vorticity pairing discrepancy remains.
A compatible constant-PV mass-flux comparison removes that residual without
losing zero work. These are diagnostic alternatives, not runtime replacements.

See [definitions, results and limits](../verification/ocean/ROTATION_PV_AUDIT.md).
Production sources, fixtures and tolerances are unchanged. The frozen mesh is
not uploaded because it exceeds 128 KiB; its input hash is preserved. New tests
check independent geometry partitions, public/sparse operator agreement, path
reconstruction, corruption sensitivity, production pairing and buffer restoration.

Installed-wheel validation outside the checkout: 831 passed, one CUDA skip;
corrected-Gibbs and dry-helper portable comparisons pass. Four focused audit
tests pass. Relevant installed source hashes match the base tree. CPU float64
only; no new time integration, GPU, gradient or physical-flow qualification.

## JW-A t=0 rotation comparison

On the recovered 16,384-cell mesh with 32 levels, consistent corner/index
geometry lowers the current unperturbed JW-A mass-weighted momentum RMS from
4.40312e-5 to 3.36493e-5 m/s² (23.6%). Maximum residual falls 19.6%. The same
fixed-vorticity metric-skew pairing retains rotation-work cancellation and
closed atmospheric energy plus lid work at roundoff. Initial fields and all
non-rotation terms are identical. No production geometry is replaced and no
timestep is taken. See [the report](../verification/ocean/JWA_ROTATION_BALANCE.md)
for configuration, alternate norms, source/state hashes and remaining limits.
Six focused audit tests pass against the installed wheel, including independent
dense pairing, production equality and unchanged geometry buffers. No full-suite
rerun is claimed for this diagnostic-only follow-up; the preceding audit suite
result remains separately recorded. Oversized mesh input is omitted.

## Compatible runtime rotation

The explicit mesh scheme `compatible` implements true kite intersections,
exit-vertex-consistent path weights, dual carrier transfer and PV-times-flux
rotation in atmosphere/ocean consumers. Historical default outputs and all
fixtures/tolerances remain unchanged. On the recovered 16,384-cell mesh, initial
JW-A RMS acceleration improves 23.6%; ten-minute wind-change RMS improves 20.2%.
The variable-thickness uniform-PV defect falls to 2.93e-11 relative; energy-work
closure remains at roundoff. Installed-wheel portable suite 840 passed/one CUDA
skip; expanded correction tests 10 passed; both atmospheric fixture comparisons
pass. See [complete scope and evidence](../verification/ocean/ROTATION_CORRECTION.md).
## Equiangular diagnostic operators

Added `vfs.geometry.EquiangularGrid`: exact-area integrals/means, face gradients,
conservative divergence, matched scalar Laplacian, face inner products and an
explicit collocated-wind adapter. Existing LatLonGrid behavior and dynamical-core
operators are unchanged. The grid is global and cell-centered, without masks.

Installed-wheel validation outside the checkout: 834 passed, one CUDA skip;
corrected-Gibbs and dry-helper portable comparisons pass. Seven new tests cover
geometry, cancellation, adjointness, dissipation, harmonic refinement, input
validation and field-value autograd (including gradcheck). The example and guide
snippet run; local guide links and diff whitespace checks pass. CPU float64 is
tested; dtype movement is smoke-tested in float32. No GPU or model-rollout claim.

The degree-one harmonic's dimensionless RMS Laplacian error decreases from
0.00652093 at 16×32 to 0.000127905 at 128×256. Polar-row maximum errors converge
approximately first order; see [the guide](EQUIANGULAR_GRID.md). A smooth
northward-wind adapter check (v=cos(latitude), unit radius) has divergence RMS
errors 0.0111116, 0.00278122 and 0.000695514 at 16×32, 32×64 and 64×128.

## Equiangular curl and Voronoi identities

The equiangular grid provides native dual-cell curl, polar-cap circulation,
conservative dual-to-cell averaging and `curl_from_collocated` for wind/stress.
Each polar row repeats one cap average with fractional area weights. The native
curl-gradient and global Stokes identities are checked separately from accuracy
of the collocated-data interpolation. Tilted solid-body rotation's cell-result
RMS error falls from 0.0211032 to 0.00602573 to 0.00166804 on unit-radius grids
16×32, 32×64 and 64×128. Axis-aligned native circulation matches exact dual-area
averages, including caps, within roundoff.

Direct Voronoi regression checks now cover curl-gradient, global circulation and
curl adjoint identities on (cells, Lloyd iterations) = (16,0), (64,0), (64,5),
(256,2), with batched three-level random fields. For dimensionless scalar input,
maximum closed-loop residuals are 8.88e-16, 8.88e-16, 8.88e-16 and 1.33e-15.
Existing source documented curl-gradient cancellation, but no direct test for
that identity was found in the preceding suite. No production Voronoi/rotation
operator changes are made. The separate balance investigation's limitations on
PV and rotation consistency still apply; zero energy work does not prove those
stronger properties.

Validation: installed-wheel portable suite 842 passed, one CUDA skip; both
corrected-Gibbs and dry-helper portable comparisons pass. The equiangular example
and guide snippets execute. New curl tests include field-value gradcheck. CPU
float64 only; no GPU or model-rollout qualification is inferred.


## Equiangular dataset diagnostics

`EquiangularGrid.from_coordinates` retains either latitude order and cyclic
longitude offsets/seams, validates midpoint bounds, and supports centered and
pole-sampled global layouts. Polar scalars use a single cap per pole; vector
components retain their local longitude-dependent basis. An explicit xarray
example provides divergence and wind/stress curl without spatial regridding.
See [qualification](EQUIANGULAR_VALIDATION.md) for analytic vector refinement,
polar error limits and CPU float64 structural/field-derivative checks. Dycore
sources, fixtures and historical tolerances are unchanged.

## Equiangular real-data diagnostics

The optional [WB2 campaign](../validation/equiangular_wb2/README.md) measures
vorticity/divergence agreement, regional circulation/flux closure, geographic
ordering invariance, scalar identities and coarse-scale sensitivity. It keeps
WB2 archive comparisons separate from full spherical finite differences because
the inspected WB2 formulas omit spherical curvature terms. Package dependencies,
dycore code and fixture tolerances are unchanged. Ordinary CI stays offline.

## Equiangular polar and mapping diagnostics

[Analytic and mapping evidence](../validation/polar_energy/README.md) covers both
layouts through 720 latitude intervals and one ERA5 pole-input sensitivity case.
An analogous mechanical Hamiltonian is evaluated on the target grid; inventory
and energy defects of existing nonconservative KNN are recorded separately.
Production operators, fixtures and tolerances are unchanged.

## Optional ocean TKE

[TKEOcean](../docs/OCEAN_TKE.md) composes the existing stratified ocean with
a prescribed-length vertical closure and an explicit h*k state. Native shear
loss, column buoyancy work and dissipative heat have separate energy ledgers.
Original dynamics/defaults/fixtures remain unchanged. The column buoyancy
allocation, timestep rejection, first-order splitting and absence of boundary
forcing are explicit limits; see [qualification](../verification/ocean/TKE.md).

## Local ocean buoyancy work

The TKE scalar exchange now uses adjacent-layer pair solves with a per-layer
energy ledger. Each interface charges only adjacent TKE reservoirs; geometric
half-layer weights replace the column-wide allocation. The interface energy
transport is reported separately from mixing work. Original #51 evidence is
preserved; see [local qualification](../verification/ocean/LOCAL_TKE_BUOYANCY.md).

## Ocean boundary stress

Optional prescribed surface stress and flat stationary bottom drag now compose
with per-layer TKE. Native impulse/work and drag-to-TKE ledgers close independently
of the large thermal background. Independent constant-viscosity Ekman references
show spatial convergence and first-order split startup convergence; a short
spherical evolving-TKE run checks integration. Separate surface TKE input and
sloping-bed stresses remain deferred. See [qualification](../verification/ocean/BOUNDARY_STRESS.md).

Boundary-stress installed-wheel regression: 941 passed, one CUDA skip; corrected
Gibbs and dry-helper portable comparisons pass. Fifteen boundary tests and both
new runnable diagnostics pass on CPU float64.

## Bottom drag over varying depth

Removed the constant-depth restriction from horizontal bottom stress. Equations,
flat-bed numerical expressions and KE-to-TKE allocation are unchanged. Wet-cell
column/domain depth consistency remains required. Nine additional boundary
regressions, a varying-depth example and 32–2048-cell reconstruction refinement
separate boundary work from existing rest-pressure errors. See
[qualification](../verification/ocean/BATHYMETRIC_DRAG.md). Explicit slope-normal
traction, surface TKE injection and topographic form drag remain absent.

Varying-depth drag regression: 950 passed, one CUDA skip on the installed wheel
outside checkout; corrected-Gibbs and dry-helper portable comparisons pass.

## Prescribed ocean radiation

Added signed surface heat and conservative two-band sunlight absorption with
explicit bottom export/absorption. Layer entropy changes match the existing
Hamiltonian energy increment at fixed mass/salt/velocity. The shared heating
extraction preserves the previous TKE path exactly across 28 tensor comparisons.
Analytic 1/4/16/64-layer absorption, signed cooling, smooth derivatives and a
short radiation/TKE example are recorded in
[qualification](../verification/ocean/RADIATION.md). No atmospheric radiation or
freshwater exchange is added.

Radiation installed-wheel regression: 967 passed, one CUDA skip; corrected-Gibbs
and dry-helper portable comparisons pass. All tests run outside the checkout.

## Prescribed surface exchange

Added an ocean-only operator for supplied freshwater, sensible/latent heat and
water-carried energy/momentum fluxes. Native water/momentum updates retain salt
and passive contents; a top-layer thermal adjustment closes the existing full
Hamiltonian energy budget. All carrier reference conventions are explicit.
Fifteen focused tests, manufactured-target recovery, constant-flux subdivision
and combined radiation/stress/TKE evidence are recorded in
[qualification](../verification/ocean/SURFACE_EXCHANGE.md). Atmospheric flux
calculation, wetting/drying and advective TKE exchange remain absent.

Prescribed-exchange installed-wheel regression: 982 passed, one CUDA skip;
corrected-Gibbs and dry-helper portable comparisons pass outside checkout.

## Forced-column qualification

See [campaign results](../verification/ocean/COLUMN_CAMPAIGN.md): 32 cases,
26 active-mixing rejections, six 48-hour momentum controls. Installed-wheel
regression: 985 passed, one CUDA skip. Production code remains unchanged.

## Energy-limited mixing candidate

See [local energy-limited mixing](../verification/ocean/ENERGY_LIMITED_MIXING.md)
for the numerical change and separate evidence. Historical fixtures and guards
are unchanged; this is a physics update, not packaging equivalence.

## Sustained TKE qualification

The [sustained campaign](../verification/ocean/SUSTAINED_MIXING.md) completes all
23 cases for 48 hours. Maximum energy residual: 1.383e-9 J/kg. TKE profiles
remain timestep/vertical-resolution sensitive; local trial-factor sensitivity
is explicitly reported.
Default mixing behavior is unchanged; limiter diagnostics and the existing trial
factor are explicit. Installed-wheel portable suite: 990 passed, one CUDA skip;
final checkpoint/column harness: five passed.

## Ocean mixing cost experiment

[Sweep-invariant calculation reuse and analytic cost prediction](../verification/ocean/MIXING_COST.md)
retain actual Hamiltonian/local-TKE/entropy guards. The cached default matches
parent state/report outputs and a two-hour trajectory exactly. A research-only
predictor saves roughly 12% in the 128-column fixed-state CPU benchmark, with no
material improvement in timestep error. Large-step mixing and weak-TKE-layer
sensitivity remain unresolved. No defaults, fixtures or tolerances change.

## Coupled ocean column prototype

[Research-only end-TKE coupling](../verification/ocean/COUPLED_COLUMN.md) retains
actual-work residual checks and native kinetic-energy accounting. Resistance-based
momentum/scalar conductance and nonlinear TKE transport are explicit numerical
choices. Eight short cases complete; the accelerated one-hour matrix rejects.
Cautious iteration completes three one-hour wind runs but is expensive and has no
demonstrated matched-accuracy speed advantage. Supported runtime code is unchanged.
Installed-wheel regression: 1005 passed, one CUDA skip; seven focused tests pass.

## Linear-EOS mixing control

[Matched linear-density/caloric controls](../verification/ocean/LINEAR_EOS_CONTROL.md)
show that constant cp greatly reduces evaluation cost but does not by itself cure
accelerated convergence failures. An exact affine work formula completes five of
six one-hour controls; dt600 and the 48-hour attempt remain failed. Same-EOS bulk
profiles closely match the cautious finite-work solve, implicating cost-evaluation
sensitivity. The adapter, hook and controls are research-only; installed runtime,
EOS defaults, physical sources, fixtures and tolerances are unchanged.

## JMD95 conservative-path work research

The research `PathWorkClosure` replaces subtraction of nearly equal parcel
energies with analytic derivative integration along conservative h0/SP paths,
retaining the coupled solver's local work allocation and full-H/entropy guards.
Eight one-hour cases complete, including 32 layers and cooling/rain forcing.
The matched accelerated finite-work control rejects at 40 minutes; the cautious
finite-work endpoint agrees closely. TKE timestep sensitivity remains material.
See [full evidence](../verification/ocean/PATH_WORK.md). No installed package
source, fixture, default or tolerance changed.
The 48-hour attempt rejects after 6 hours on the unchanged layer-budget gate;
last accepted full-H residual 2.07e-12 J/kg. Installed-wheel portable regression:
1017 passed, one CUDA skip; corrected-Gibbs and dry-helper comparisons pass.

## Local TKE budget diagnostic

The rejected six-hour path-work input passes at iteration 92 without changed
physics or tolerances. Other accepted variants reveal that frozen-budget closure
plus the existing TKE-change check does not ensure refreshed end-coefficient
consistency. A local positive splice also fails that fresh check. See
[ledger and replay evidence](../verification/ocean/LAYER_BUDGET.md). Next target:
refreshed residual acceptance and a constrained nonlinear solve. Supported runtime
is unchanged. Installed suite: 1020 passed, one CUDA skip; final focused tests:
10 passed. No GPU or backward qualification.

## Refreshed end-TKE acceptance

The opt-in research closure now checks and commits physics recomputed from the
candidate end TKE, retaining positivity, entropy and full-H guards. All ten
one-hour controls pass, and the pinned six-hour rejection passes at iteration 92
with an independent refreshed residual below its unchanged limit. Budget-weighted
acceleration reduces one-hour trial counts, but fixed-input cost rises versus
legacy and TKE remains timestep-sensitive. Supported runtime/defaults/fixtures
are unchanged. Installed suite: 1023 passed, one CUDA skip; portable comparisons
pass. See [qualification and limitations](../verification/ocean/REFRESHED_SOLVE.md).
The 12-hour dt300/16-layer wind run completes with full-energy residual
1.60e-11 J/kg. It requires 7637 iterations and 8381 trials, including costly
transition periods. Longer-run and temporal-accuracy qualification remain open.

## Semi-implicit vertical mixing research

A bounded predicted-coefficient update splits TKE transport, actual JMD95
buoyancy work and dissipation, retaining native shared-edge energy allocation.
Multiple scalar right-hand sides share a factorization; momentum and TKE factors
are reused through funding retries. The twelve-hour/four-substep control completes
with 1727 trials and energy residual -1.20e-10 J/kg. A 162-cell mesh comparison
finds a lower-cost update closer in all reported fields to its fine numerical
reference. Coarse-step/limiter/global-scaling limitations remain explicit in
[the report](../verification/ocean/SEMIIMPLICIT_MIXING.md). Supported runtime and
fixtures are unchanged; installed suite: 1030 passed, one CUDA skip, portable
fixture comparisons passed. No GPU/backward or full global-dycore qualification.

## Higher-resolution mixing campaign

The [32–64-layer and larger-mesh campaign](../verification/ocean/RESOLUTION_CAMPAIGN.md)
qualifies the research semi-implicit candidate with longer forcing, fixed
physical initial profiles and native spherical batches. Conservation, numerical
reference errors, limiter activity and cost are reported separately. Supported
runtime and defaults remain unchanged. Installed-wheel regression: 1,031 passed,
one CUDA skip; portable Gibbs and dry comparisons passed.

## Mixing profile and cost budget

The [mixing profile](../verification/ocean/MIXING_PROFILE.md) attributes 96.9%
of instrumented 1,024-by-64 mixing time to thermodynamic kernels. Unprofiled
mixing costs 123.63 CPU s versus 22.25 s for one dynamics step or 109.64 s for
five dt60 steps. A proposed 20% two-block budget allows 27.41 s for mixing at
the latter cadence; that combined schedule remains unqualified. The report
proposes cached thermodynamic work, residual-controlled warm starts and
derivative-only heating before frozen-conjugate trial approximations. No
physical formulation, supported runtime, defaults or tolerances are changed.

## Optimized mixing thermodynamics

The opt-in research `fast_thermodynamics=True` path retains JMD95, the same
Hamiltonian work and guards, and reduces the 1,024-by-64 four-substep mixing
median from 106.15 to 29.94 CPU s (3.55×). Long 64-layer forced columns agree
with the original to about 4.1e-13 TKE relative L2 or better. Public defaults
remain unchanged; an internal validation hook reuses residual-checked physical
fields. Installed suite: 1,052 passed, one CUDA skip, plus a separately passing
composed-step regression; both portable fixture checks pass. See
[implementation and evidence](../verification/ocean/FAST_THERMODYNAMICS.md).

## Local CPU throughput harness

The [performance campaign](../validation/performance/README.md) records warm-up,
repeated evolving-step wall timings, process CPU time, memory, source fingerprints,
endpoint inventories and limiter/forcing/remap ledgers. Failed/timed-out cases
receive no SYPD. Supported and research closures are explicitly distinguished.
No package kernels, fixtures, tolerances or defaults change.

Harness validation: 12 installed-wheel smoke cases and six 128×16 evolving-step
cases pass; throughput accounting tests and both portable fixture comparisons
pass. See `verification/performance/README.md`. These are execution checks,
not M4 Pro measurements or global timestep qualification.

## Ocean stage work reuse

Transient RHS work avoids repeated physical-state inversions, Gibbs face
calculation, JMD95 gradients and donor transport between the base and sharp-front
transport calculations. Original expressions, guards, fixtures and tolerances
remain. At 1024×64, single-thread CPU dynamics cost falls 27.22 → 14.20 s per
step with exact state/report equality; full-ocean gains are smaller. See
[measurements and reproduction](../verification/ocean/STAGE_REUSE.md).
Installed-wheel portable suite: 1061 passed, one CUDA skip, one existing warning;
corrected-Gibbs and dry portable comparisons pass. Focused reverse-mode and
invalid-content checks pass; no full-model backward qualification is asserted.

## Validated ocean field reuse

Checked dynamics/forcing fields are reused only for unchanged contents within a
call; transport connectivity is prepared with mutation guards, and repeated local
gathers are shared. Optional `warm_start=True` keeps the same entropy residual
and domain checks with Newton polishing and fallback. The default solver remains
unchanged. See [measurements and limitations](../verification/ocean/FIELD_REUSE.md).
1024×64 dynamics exploratory medians improve 14.65 → 9.56 s/step (1.53×), or
6.42 s (2.28×) with optional warm inversion. Full-model mixing cost remains.
No atmospheric kernels, Hamiltonian/EOS formulas, fixtures or tolerances change.
Final installed-wheel suite: 1074 passed, one CUDA skip, one existing warning;
corrected-Gibbs and dry portable comparisons pass. Default captures match exactly
(122/232/280 small-case tensors; 64 large dynamics tensors). Warm endpoint relative
L2 differences reach 3.16e-15 in these short tests; limiter reports can differ.
Thirteen new focused checks and three warm CLI smoke cases pass. Inference
geometry without mutation counters retains ordinary uncached indexing.

## Cheaper ocean energy evaluation

The opt-in quadrature campaign exposes orders 2 and 3 in `StratifiedOcean`, with
order 4 unchanged by default. The EOS is unchanged; lower order defines a different
discrete Hamiltonian whose derivatives and mixing/remap operations remain matched.
Against order 12 on 100–6000 m states, maximum pressure-force relative L2 differences
are 6.58e-6 (order 2), 1.35e-9 (3), and 1.11e-12 (4). Order 8 checks convergence.
See [source, timings and limits](../verification/ocean/ENERGY_QUADRATURE.md).
A pure surface-Gibbs kernel enables isolated compilation experiments beneath
unchanged public guards; no compiled backend is enabled by default.

Final per-model compiled full-step samples (1024×64, dt60, four mixing substeps,
one CPU thread) improve eager q4 61.25 s to 18.70 s with q4 or 14.83 s with q2.
These are short samples beyond #72, not Mac or long-run qualification. Final
installed-wheel suite: 1084 passed, one CUDA skip; both portable fixtures and
three compiled CLI smoke cases pass. Defaults remain eager q4. Compact evidence
and source/wheel hashes are retained; oversized tensor captures are omitted.

## Temperature-native thermal physics prototype

The temperature-coordinate research path now supports prescribed layer-energy
increments and lagged vertical TKE mixing. The same finite-H heating solve,
funding retries, local/column energy and entropy checks are retained. Direct
h*T storage removes entropy reconstruction; reference-enthalpy inversions remain.
A 1024×64 compiled short heating+mixing sample shows no meaningful speedup
(10.074 vs 10.034 s), despite removing 15 entropy and four baseline enthalpy
inversions. Both energy-minus-input residuals are -5.93e-12 J/kg. This is not a
supported full-ocean timestep: surface exchange, stress/radiation adapters and
ALE remain outside the temperature-native composition. See
[methods and evidence](../verification/ocean/temperature-mixing/README.md).
Final outside-checkout portable regression: 1102 passed, one CUDA skip, one
existing warning. Old entropy research helper outputs matched 24 state/report
tensors bitwise in each of ordinary and fast mode on the recorded small case.

## Temperature-native complete research composition

Optional research adapters now cover prescribed surface exchange, radiation,
stress, mixing and adaptive ALE while retaining the same Hamiltonian and
acceptance budgets. T/S-path work removes intermediate enthalpy inversions;
the complete 1024×64 step including ALE uses nine enthalpy and zero entropy
inversions. Small-case timing improves 2.55×; a noisy large-case single sample
improves only 1.09×. No supported default changes or global speedup claim.
See [qualification and raw results](../verification/ocean/throughput-composition/README.md).

Installed-wheel, outside-checkout portable regression: **1115 passed, one CUDA
skip**, one existing warning. Thirteen new composition checks pass.

An alternating 1024×64 follow-up (three samples each, same compiled kernels)
measures 21.728 → 9.709 s including ALE, **2.24×**. The earlier noisy
single-sample comparison remains recorded; no Mac/global extrapolation.

## Explicit linear-density ocean configuration

The separate research `LinearOcean` configuration uses one affine-density,
constant-cp Hamiltonian across dynamics, mixing, heating and ALE. It retains
active T/S, TKE, conservative inventories and existing acceptance guards.
See [equations and runnable example](../validation/temperature_coordinate/LINEAR.md)
and [measured qualification](../verification/ocean/linear-configuration/README.md).
Supported JMD95 defaults and atmospheric expressions remain unchanged. The only
package extension is optional reference-owned analytic face/inverse dispatch;
research-only mixing/remap reconstruction hooks retain their prior defaults.

The subsequent [matched-EOS qualification](../verification/ocean/matched-eos/README.md)
adds reproducible wave, rotating-flow and isolated mixing comparisons plus EOS
response diagnostics. Runtime code/defaults are unchanged. Close matched warm
cases do not establish accurate cold-water formation or global climate behavior.

The [Roquet freezing research configuration](../validation/temperature_coordinate/ROQUET.md)
adds CT/SA polynomial thermodynamics without changing supported JMD95 defaults.
Independent derivative/work checks and the installed-wheel portable suite are
recorded in [its qualification report](../verification/ocean/roquet/README.md).
Density coefficients are published; the accompanying constant-cp entropy model
is explicitly idealized. This does not constitute TEOS-10 or climate qualification.

The [Roquet timestep campaign](../validation/timestep/README.md) separates
finite-duration completion, conservation and reference-relative temporal error.
It supports saved meshes, explicit resolution/layer/depth choices, fixed maximum
mixing substeps, isolated timeouts and resumable completed trials. It does not
assign a universal or climate-safe timestep. Compact implementation checks and
target-resolution attempt results are retained under `verification/ocean/timestep`.

Optional Roquet kernel compilation and live timestep progress are available in
the timestep campaign. Selected pure energy/caloric kernels compile; numerical
guards, physics control and timestep loop remain eager. Recorded wall time
includes first-use compilation; it is not a warmed performance measurement.

The opt-in [GM–Redi research operator](../validation/temperature_coordinate/GMREDI.md)
adds prescribed-coefficient eddy transport to the Roquet CT/SA path. A discrete
closed circulation supplies GM donor transport; an adjoint small-slope Redi
operator conserves inventories and dissipates tracer variance. Surface-referenced
potential density approximates neutral geometry. Exact finite Hamiltonian work
is diagnosed but not compensated with heat or eddy energy. The implementation
rejects invalid/negative tracer updates; rotated Redi has no maximum principle.
See [front relaxation and refinement evidence](../verification/ocean/gmredi/README.md).
This research addition changes no existing runtime expressions or defaults.

GM–Redi validation on CPU float64: 13 focused tests, 1171 passed / 2 skipped
in the outside-checkout portable suite, 142 corrected-Gibbs fixture tensors and
93 dry-helper tensors passed. Initial user-site isolation failures were resolved
by installing into a virtual environment, with no test expectation changes.
Front relaxation and first-order timestep refinement are recorded; GPU,
backward rollout, spatial convergence and climate performance are unqualified.

The [mesoscale reservoir extension](../verification/ocean/gmredi/ENERGY_RESERVOIR.md)
replaces the unfunded GM–Redi diagnostic with an explicit `EddyState.eddy_h`.
`RoquetHamiltonian` composes the original resolved energy, turbulent TKE and
mesoscale reservoir and supplies exact finite tracer work. Each GM/Redi stage
pairs conservative energy transport with a local reservoir exchange, rejecting
unaffordable withdrawals. The reservoir survives resolved transport, forcing,
vertical mixing and ALE. Redi slopes now refresh after GM; cold and warm tests
verify both signs of its energy exchange using the published default EOS.
Existing resolved RK errors are measured, not absorbed into the reservoir.

Reservoir qualification: 22 focused tests; installed-wheel outside-checkout
portable suite 1180 passed / 2 skipped; 142 corrected-Gibbs and 93 dry-helper
fixture tensors passed. Thirty-six isolated front runs retain nonnegative
funding; endpoint finite-H residuals are at most 3.73e-16 J/kg. Local stage
first-law residuals are at most 6.66e-17 J/kg. Both signs of Redi energy work
are covered. Direct large-H subtraction remains limited by roundoff, with a
maximum 8.15e-12 J/kg difference. One backward sensitivity is checked; general
backward rollouts and climate/target-resolution qualification remain open.

The [GM–Redi optimization pass](../verification/ocean/gmredi/optimization/README.md)
reuses immutable geometry within each closure call, builds only the applicable
GM/Redi operators, reduces tensor temporaries, and avoids revalidating unchanged
physical fields after reservoir-only updates. All physical guard classes remain
active; Redi still recomputes slopes after GM. Against commit 774e2f3, 18 cases
advanced five steps retain bitwise-identical resolved fields and turbulent TKE;
maximum eddy-content difference is 1.21e-13 m³/s². One backward sensitivity differs
by 6.39e-18. Installed-wheel portable regression: 1186 passed, 2 skipped;
142 corrected-Gibbs and 93 dry-helper fixture tensors passed. Arithmetic regrouping
changes reduction order, so long-trajectory bitwise identity is not claimed.

Sequential 16384×40 eager one-thread CPU timing: funded closure median
3.910 → 3.356 s (14.2% less), whole-process peak RSS 3.526 → 3.267 GB
(7.4% less). Estimated full step remains about 20 s (19.835 → 20.040 s);
unchanged-control and reservoir-dynamics timing variation precludes a claim of
end-to-end speedup. Three samples per operation are retained in the report.

The [factored Redi pass](../verification/ocean/gmredi/factored/README.md) computes
column vertical derivatives once and averages them to edges, then reverses those
operations for the transpose. Tracer and Hamiltonian energy flux use the same
transpose; the old absolute eight-entry bound is algebraically retained. Four
new explicit-stencil/adjoint/variance tests pass, alongside the 28 existing
GM–Redi/reservoir checks with unchanged tolerances. In 18 five-step cases,
velocity, h, CT content, SA content and TKE remain bitwise identical; maximum
passive and reservoir content differences are 1.42e-14 and 8.53e-14 respectively.
One backward sensitivity differs by 1.00e-18. Extensive energy-report differences
from cancellation are retained in comparison.json; these are not a claim of
bitwise equivalence. CPU float64 only; no compiled/GPU/long-rollout claim.

Installed-wheel tests outside checkout: 1189 passed, 2 skipped and one known
exact atmospheric-fixture mismatch (max 4.44e-15 initial velocity). The affected
extraction file passes all 3 tests under the existing portable policy; combined
coverage is 1190 distinct passing tests. Standalone portable comparisons pass
142 corrected-Gibbs and 93 dry-helper tensors. Both test logs are retained.

At 16384×40, uncompiled single-thread CPU float64, closure medians decrease
3.184 → 1.649 s in the full benchmark (48.2%) and 3.614 → 2.587 s in a separate
closure-only process pair (28.4%). Closure-only peak process RSS decreases
2.396 → 1.765 GB (26.3%); full-process peak remains about 3.3 GB. Summed component
medians decrease 20.803 → 18.493 s, with unchanged dynamics also timing 4.4–7.0%
faster; no precise end-to-end speedup is attributed from those sums. Raw timing
samples, source hashes and reproduction commands accompany the report.

## Optional eddy energy feedback and decay to heat

The [feedback report](../verification/ocean/gmredi/feedback/README.md) adds
`kappa_GM = c L sqrt(2 f E_edge)` with explicit mixing length, efficiency and
kinetic fraction. Prescribed GM, Redi and installed production defaults remain
unchanged: all 487 reference outputs match bitwise. A separate exponential
reservoir decay uses the existing exact Roquet thermal quadratic solve and
reports actual stored reservoir debit, thermal gain and local full-H residual.
No direct identification with turbulent TKE or hidden energy seed is made.

45 focused tests pass (13 new), including coast/zero-energy cases, decay from
dt/tau=1e-6 to 1000, independent polynomial energy accounting, entropy increase,
invalid inputs and a backward finite-difference check. Fifteen one-hour-front
and ten-minute-zonal-band experiments have maximum local closure residual
3.64e-12 J/kg and cumulative residual 1.67e-12 J/kg, at stored-CT rounding scale.
Resolved-step drift is measured separately. The masked spherical channel is a
short coupled budget check, not a balanced Eady/eddy-saturation benchmark.
Long channel calibration, backscatter, GPU/MPS, float32, compiled timing and
long backward rollouts remain open.

Installed-wheel portable regression outside checkout: **1203 passed, 2 skipped**
in 194.72 s. Corrected-Gibbs and dry-helper comparisons pass 142 and 93 tensors.
Production VFS source/defaults and installed wheel remain unchanged.

Same-process interleaved timing at 16384×40: prescribed closure median 2.420 s,
energy-dependent GM–Redi plus decay 2.497 s; dissipation alone 0.172 s.
Estimated full step (same 16.517 s dynamics median) is 18.937 → 19.014 s,
about 0.4% difference within timing variation. Whole-process peak RSS 3.234 GB.
This supports no large eager CPU overhead, not a universal performance guarantee.

## Ocean dynamics tensor lifetimes

The [dynamics memory report](../verification/ocean/dynamics-memory/README.md)
profiles the 16384×40 resolved step and identifies overlap of donor/original/
target/final ledgers with large temporary edge arrays. The optimization drops
consumed edge gathers and limiter budgets, superseded ledgers, endpoint reports
before validation, and dead RK state references. It preserves numerical
expressions, all guards, full diagnostic contents and autograd graphs.

Six front/coast cases capture tendencies and reports, three successive resolved
steps with reservoir transport, donor/FCT ledgers and one-step derivatives.
All **486 tensors match bitwise** across separate before/after installed-wheel
executions. No tolerance changes. Production ocean transport modules are edited;
atmospheric source and all frozen fixtures are unchanged.

Rebuilt installed-wheel portable suite: **1203 passed, 2 skipped**, 202.79 s,
from outside the checkout. All 142 corrected-Gibbs and 93 dry-helper comparisons
pass. Wheel hashes, strict equivalence result and full logs are retained.

At 16384×40, one-thread eager CPU float64, the unprofiled dynamics benchmark
peak RSS decreases **3.298 → 2.669 GB (19.1%, 629 MB)**. Median step timing is
18.442 → 17.966 s; overlapping samples do not establish a reliable speedup.
This measures resolved dynamics plus reservoir transport, not a combined long
GM–Redi/decay integration. All 486 reference tensors remain bitwise identical.

## Skip discarded donor-reference limiter work

The [donor-work report](../verification/ocean/donor-work/README.md) moves inner
FCT budget/ratio/limiter evaluation into the FCT branch. Donor mode creates its
identical zero limiter directly; all amount expressions, returned bounds and
fluxes, admissibility checks and outer energy-consistent corrections remain.
All 486 same-environment comparison tensors, including backward gradients and
both donor/FCT reports, remain bitwise identical. No default or tolerance changes.

Installed-wheel portable suite: **1203 passed, 2 skipped**, 199.34 s, outside
the checkout. All 142 corrected-Gibbs and 93 dry-helper comparisons pass.

At 16384×40, one-thread eager CPU float64, median step time falls
18.433 → 17.233 s (6.5%) and peak RSS falls 2.654 → 2.600 GB
(2.0%, 54 MB). Three timed samples follow one warm-up per fresh process.

## Reuse transport work and omit overwritten target ledgers

The [transport reuse report](../verification/ocean/transport-reuse/README.md)
separates target limiter/trace construction from target amount/flux materialization.
The energy-constrained path skips the latter, which it replaced immediately;
public prescribed-carrier transport still returns its complete ledger. Repeated
correction products and local antidiffusive sign splits are reused. Carrier
clamps remain separate to preserve backward accumulation exactly.

All 486 captured state/tendency/ledger/backward tensors match `214da28` bitwise.
Installed-wheel portable tests outside the checkout: **1203 passed, 2 skipped**,
196.90 s. All 142 corrected-Gibbs and 93 dry-helper comparisons pass. No checks,
numerical defaults, fixtures or tolerances change.

Median step time falls **17.359 → 15.063 s (13.2%)**. Peak process RSS falls **2.658 → 2.574 GB (3.2%, 85 MB)**.
Measured at 16384×40, one-thread eager CPU float64, three timed steps.

## Historical finite-step backscatter prototype (superseded)

The opt-in [backscatter prototype](../validation/temperature_coordinate/BACKSCATTER.md)
uses rotational negative viscosity, the existing adjoint curl pair and the full
Roquet Hamiltonian's mechanical metric. Exact finite kinetic work debits the
mesoscale reservoir, with an explicit fraction/exponential-timescale allowance.
The conservative funding limiter uses one amplitude per layer; a depleted cell
can restrict remote cells in that layer. This is not yet a calibrated local,
resolution-aware closure and no new grid-scale dissipative operator is supplied.

Four 20-step channel cases, including coupled dynamics and strong funding caps,
retain local backscatter residuals below 7.11e-18 J/kg. All 486 default-path
comparison tensors remain bitwise identical. See the [verification report](../verification/ocean/backscatter/README.md)
for tests, measurements, usage and scientific limitations.

Installed-wheel portable regression: **1218 passed, 2 skipped**, 201.10 s,
including 15 new backscatter cases. Both frozen-fixture comparisons pass.

Isolated backscatter median: 0.4755 s at 16384×40, one-thread eager CPU
float64; about 3.2% of the preceding dynamics median (estimated incremental
cost, not a combined-step timing). Isolated-process peak RSS is 0.802 GB.

## Instantaneous local backscatter

The [current formulation](../validation/temperature_coordinate/BACKSCATTER.md)
replaces the split kick with dt-independent Hamiltonian-paired momentum and
reservoir tendencies. A local vertex power limiter replaces the layer minimum.
`RoquetOcean(..., backscatter=...)` evaluates both sources within every SSPRK3
stage; source/transport positivity failures reject rather than clip. No finite
quadratic correction or hidden thermal repair remains. Existing defaults are
unchanged. The old `apply` API is removed; use `tendency` or model integration.

Independent full-H gradient contractions and a distant-forcing locality test
pass. Channel power residuals stay below 3.933e-23 W/kg; finite-step energy
errors are reported separately and approach third-order convergence in the
smooth isolated test. See [current evidence](../verification/ocean/backscatter/instantaneous/README.md).

Current installed-wheel suite: **1217 passed, 2 skipped**, 200.33 s, including
14 instantaneous backscatter cases. Both frozen comparisons pass; all 486
default-path state/ledger/backward tensors match `34cc640` bitwise.

Isolated tendency median at 16384×40: 0.3811 s, eager one-thread CPU float64.
Three RK evaluations imply about 1.14 s estimated incremental step cost (7.6%
of the prior dynamics benchmark), not a measured combined-step speed claim.

## Integrated ocean evaluation (2026-11-14)

Seventeen low-resolution integrations completed (50,400 steps; 58 aggregate
model days), plus two expected rejection checks. Energy and inventories remain
well controlled, but backscatter strongly concentrates KE on short mesh edges;
physical spatial quality is not qualified. Stratified bathymetric rest has small
spurious currents that decrease with vertical refinement. No model methods were
changed. See [evaluation evidence](../verification/ocean/integrated/README.md)
and `validation/ocean_evaluation/README.md` for the reusable local runner.

## Budgeted biharmonic sink (2026-11-14)

`EddyBackscatter(..., biharmonic_viscosity=nu4)` adds opt-in rotational
biharmonic damping with nonnegative dissipation deposited into the mesoscale
reservoir. The full Hamiltonian supplies the metric. Local viscous energy
redistribution is explicit in the report; global pairing closes to roundoff.
No finite-step repair, floor, or altered default is introduced.

Installed-wheel CPU float64 suite: 1227 passed, 2 skipped; both frozen
comparisons pass. Six two-day coupled comparisons complete. At 128 cells the
peak-speed spike falls from 1.86 to .216 m/s and single-edge KE concentration
from 22.1% to 5.24%; at 256 cells the concentration falls from 38.3% to 7.58%.
This is a short-run improvement, not mesoscale calibration or climate skill.
See [implementation/evidence](../verification/ocean/backscatter/biharmonic/README.md).

## Global wave/trade-wind evaluation (2026-11-14)

Seven global CPU float64 integrations completed (11,520 steps, 23 model days):
a two-day rotating surface bump and five-day prescribed trade-wind aquaplanet,
with closure ablations and a finer-step wave control. The paired closure changes
wave elevation by ~0.025% relative to the finer-step baseline. Trade winds yield
~7.4 cm/s mean westward tropical surface flow; full-H-minus-wind-work residual is
~5.5e-9 J/kg. No model numerical methods changed. A new RMS broadcasting error
was corrected in analysis from snapshots; raw records and the identical-endpoint
repeat are retained. See [report](../verification/ocean/waves-and-trades/README.md)
for limits, figures and local higher-resolution commands.

## Hamiltonian stationary ocean–ice columns (2026-11-14)

New opt-in research components use the existing full ocean Hamiltonian plus
latent and gravitational ice energy for freezing/melting and external conductive
heat exchange. Fifteen focused tests include independent derivative and actual
energy-change checks, coastal/material conservation and a smooth-branch backward
check. Two year-long prescribed-air column runs (daily/half-daily) retain maximum
accumulated energy residuals 3.85e-5/8.09e-5 J/m² and water residuals below
8.74e-11 kg/m². Peak ice thickness is 1.098/1.094 m. No packaged kernels changed.
Moving ocean and ice dynamics remain explicitly unsupported pending common-H
loading, momentum and stress-work terms. See [scope/evidence](../verification/ocean/sea-ice/README.md).

External installed-wheel regression: 1239 passed, 2 skipped (including the first
12 ice tests). The final 15-test ice suite then passed independently in that
same harness, including the final near-salinity-limit bracket change. Both
frozen fixture comparisons pass; no packaged source was modified.

## Compression-only ice ridging (2026-11-14)

Added an opt-in, two-category instantaneous mechanics operator, with common-H
ice KE, hydrostatic subcell ridge PE and matched compressive stress. Thin-to-thick
redistribution preserves cell ice volume; dissipative work warms the ocean via
the full Roquet caloric conjugate. It does not alter TKE/eddy reservoirs or any
packaged kernel/default. All 28 ice tests pass, including 13 new mechanics tests.
A 1000-second isolated 32-cell experiment converts ~0.218 J/m² ice KE into equal
ridge PE and heat for the chosen uncalibrated friction parameter. Step errors
at dt500/250/125 s are 3.94e-7/2.19e-8/1.33e-9 J/m²; no energy repair occurs.
Coastal and 128-cell experiments also complete. These are isolated mechanics
experiments, not global ocean–ice integrations. See [scope and evidence](../verification/ocean/sea-ice/ridging/README.md).

Final installed-wheel portable regression outside the checkout: 1255 passed,
2 skipped. Both frozen comparisons pass (142 corrected-Gibbs tensors, 93 dry
helpers). Research modules were explicitly copied into the external harness;
no packaged source changed. CPU float64 only.

## Coupled moving ocean–ice research model (2026-11-15)

Added category transport with energy-adjoint forces, ocean loading, bounded
zero-work ice rotation, prescribed wind, reciprocal implicit drag to ocean TKE,
all-category compression and category-aware freezing/melting with donor momentum
transfer. The full Roquet/category Hamiltonian is shared throughout. The wrapper
preserves existing vertical mixing and mesoscale backscatter composition. No
packaged numerical kernels or defaults changed.

Nine global/coastal experiments complete (2010 steps, 57 aggregate model hours),
including one full day, moving ice margins and complete melting. Peak energy
residuals range from 5.26e-7 to 1.82e-3 J/m²; water/salt relative errors remain
below 1.19e-15. Melt-case timestep halving decreases the peak energy residual
1.87e-4 → 2.32e-5 J/m². These are coarse numerical coupling checks, not climate
qualification. Final focused coverage is 66 passing ice tests (38 new coupling
tests); independent derivatives/work, phase momentum and smooth backward checks
are included. Portable frozen comparisons pass. See [full evidence and external
suite logs](../verification/ocean/sea-ice/coupled/README.md).

Explicit drag and inverse-mass PV rotation failed near disappearing ice during
development. Implicit midpoint drag and bounded harmonic-mass zero-work rotation
resolve the demonstrated failure. The rotation extension claims energy closure,
not discrete PV conservation. Compression-only ice remains without shear/elastic
rheology, snow or brine physics. CPU float64 only; selected local backward checks
are not a long-horizon differentiability qualification.

External installed-wheel portable suite: 1292 passed, 2 skipped. The final
partial-margin drag correction and one additional regression test then passed
in the final 66-test ice suite (1293 distinct passing tests across both runs).
Frozen tolerances are unchanged; the platform exact-mode mismatch is retained
in the evidence rather than suppressed.

## Zero-layer snow on moving category ice

Added opt-in snow accumulation, carrier-consistent advection/ridging, snow-first
melt, insulation and smoothly depth-dependent albedo. Snow is a labelled subset
of total frozen ice-equivalent volume, with an explicit snowfall mass/energy
boundary ledger. Snow-only melting preserves underlying ice volume and area.
The shared Hamiltonian counts snow once; porous depth affects surface exchange
while mechanical geometry remains ice-equivalent. No packaged default changed.

Fifteen new tests cover mass/energy/momentum, analytic phase derivatives, snow
transport and redistribution, coastal/open-water behavior, optical/thermal
responses and a smooth backward check. The bare coupled step is bitwise equal
to its predecessor. Five moving coupled experiments complete (1980 steps,
54 aggregate model hours), including coastal ice and complete melting. Peak
energy residuals are below 1.8e-5 J/m²; ordinary-cycle dt halving decreases the
residual from 1.27e-5 to 1.02e-6 J/m². [Evidence and runnable commands](../verification/ocean/sea-ice/snow/README.md).

This is zero-layer snow with fixed density and illustrative albedos, not porous
snow gravitational geometry, snow enthalpy, separate basal/surface melt fronts,
snow aging, brine or complete precipitation forcing. Incoming snow is co-moving
and only the ice-covered fraction is intercepted. CPU float64 only.

Final external installed-wheel portable regression: 1308 passed, 2 skipped.
Both frozen comparisons pass (142 corrected-Gibbs tensors, 93 dry helpers).

## Coupled freshwater and prescribed runoff

Added an opt-in freshwater boundary to the moving ocean/ice/snow wrapper: rain
drains to liquid, snowfall is partitioned by coverage, evaporation removes open
water, sublimation consumes snow before ice, and prescribed rivers enter explicit
wet receivers. The discharge adapter conserves mass and incoming thermal energy.
Liquid thickness changes retain salt/passive-tracer contents and extensive
TKE/mesoscale energy. Explicit boundary momentum, thermal enthalpy, latent cooling
and hydrostatic incorporation work (including overlying ice) pair with a local
analytic Roquet full-H thermal solve. No packaged kernel/default changed.

Twenty-two new tests cover component and combined budgets, wet/dry routing,
under-ice loading, snow-first removal and complete cover exhaustion, conservative
river mapping, forcing rejection, no-flux equivalence and one smooth backward
derivative. Final installed-wheel portable regression outside the checkout:
1330 passed, 2 skipped. Both frozen comparisons pass (142 corrected-Gibbs tensors,
93 dry helpers). Research modules were explicitly copied into the harness.

Seven six-hour experiments complete (1980 steps, 42 aggregate model hours).
Peak external-energy-adjusted residuals are 6.52e-7 to 1.75e-6 J/m²; relative
combined-water and salt errors are at most 1.06e-15 and 5.91e-16 respectively.
Energy residuals do not decrease monotonically with timestep here; background-H
subtraction is near its floating-point resolution. Selected physical diagnostics
change under refinement, so exact boundary closure is not a trajectory-order
claim. See [commands, time series and scope](../verification/ocean/sea-ice/freshwater/README.md).

This is prescribed forcing with constant latent heats and a declared cp*CT liquid
thermal reference. It is not land hydrology, automatic river-mouth mapping, bulk
atmospheric exchange, brine or resolved river plumes. Boundary and liquidus stages
remain separate. CPU float64 forward checks and a local backward check do not
qualify GPU, long integrations or long-horizon differentiation.

## External PyAM21 atmospheric workflow

Added `workflows/pyam21/`, a separately installed-dependency integration with
Evan Wellmeyer's PyAM21 at `e1bdfcc3b23c05adea80b35af880702bb85718eb`.
No VFS package source, default, installed dependency or upstream physics source
changed. CI installs the pinned optional dependency only to test the workflow.

The adapter calls the four actual upstream stages and exposes post-coupling
heat/moisture/stress, precipitation and optional radiation fluxes. Full-level
readout preserves native integrated water and mixture enthalpy. Finite physics
increments return to the original CP state, with donor-aware species withdrawals,
surface-pressure changes retaining dry air, and analytic work-adjoint edge forces.
Cloud fraction is a workflow-owned passive field using the unchanged atmospheric
bounded Euler maps and SSP combinations. Physics/radiation memory, clocks and
restart payloads are explicit and failed calls do not mutate the input state.

Twenty-one focused tests cover amounts, work, no-flux identity, actual staged
upstream equivalence, rejection, restart/cache behavior, bitwise preservation of
the atmospheric dynamics, passive-cloud transport, and one smooth SST derivative.
Radiation tests use the real PyAM21 scheduler with a synthetic package; full
SeaEsfRad and its GFDL data bundle have not been exercised in this workflow.

This first pass deliberately reports the difference between the native full-H
change, AM2's fixed-pressure thermal/latent proxy, thermodynamic conventions,
and finite mapping/carrier changes. Those differences are not corrected and
must be reconciled before claiming shared-H atmospheric physics or interactive
ocean/ice coupling. The example host is explicitly configured with gravity 9.8
before initialization, matching PyAM21's pressure-to-mass conversion. The drivers
use analytic columns, prescribed SST and no radiation. CPU float64 only.
See [commands and numerical evidence](../verification/atmosphere/pyam21/README.md).

Installed-wheel validation passed 1,351 tests with 2 skips; the corrected Gibbs
and dry portable comparisons passed 142 and 93 tensor comparisons. Four actual
physics runs completed (1,500 intervals, 19 aggregate model hours). Peak water
ledger residual was 4.77e-12 kg/m² and peak relative dry-air change 2.13e-13.
The six-hour mean VFS-minus-AM2-heat-ledger difference was +0.594 W/m² for
columns, +0.424 W/m² with dynamics and +0.443 W/m² at half timestep. These
remain unresolved energy-accounting differences, not a closure claim.

## Conservative PyAM21 energy transfer

The workflow now defaults to a finite conservative transfer of AM2 layer energy
into native VFS full-H; `energy_mode="legacy"` retains the prior temperature
mapping. No package source, defaults, dependencies or upstream physics changed.
The independent surface ledger includes sensible heat, vapor/rain/snow enthalpy
in the host mixture datum, transported-water kinetic/potential energy and net
top-minus-bottom radiation. Water momentum enters through the same work-adjoint
map at the bottom layer. The imposed ocean is stationary, hence zero stress work.
Default precipitation temperature/velocity are the old lowest air level, with
explicit overrides available; those are boundary assumptions, not AM2 outputs.

Each native layer receives AM2's actual thermal/latent plus collocated kinetic
increment and the surface-reference/mechanical difference. A cell-local CP
entropy inversion uses an analytic two-entry-per-layer Jacobian and minimum-norm
log-temperature Newton increments. It retains **any upstream energy imbalance**;
an injected heat defect and actual disabled dissipative heating demonstrate this
independently. Nonconvergence and invalid states reject without committing state.
The allocation is a documented numerical choice, not an entropy-stability proof.

Conservative mode uses actual upstream temperature/tracer endpoints, checking
their agreement with the returned tendencies at roundoff. Remaining-amount donor
mapping avoids subtraction errors when condensate is exhausted. Nonnegative
interpolation weights transfer the supplied endpoint, including signed upstream
roundoff; no moisture floor or state clipping is used. Focused regressions cover
both exhaustion issues, explicit boundary momentum and topography, layer and
global energy, phase/drag/precipitation signs, and a zero-source derivative.

See [derivation and assumptions](../workflows/pyam21/ENERGY.md) and
[current evidence](../verification/atmosphere/pyam21/energy/README.md).
Interactive ocean/ice still requires a common enthalpy datum and shared flux
record, reciprocal moving-surface stress and dissipation allocation. Full
SeaEsfRad, production meshes, long climate runs and entropy stability remain
unqualified; the examples continue to omit radiation.

Final installed-wheel portable suite: 1,366 passed, 2 skipped, including 36
workflow tests. Both frozen portable comparisons pass (142/93 tensors). Four
conservative integrations complete 1,500 physics intervals over 19 model hours.
The six-hour column full-H-minus-boundary residual is 1.03e-6 J/m²; aquaplanet
residuals are 1.218e-3 J/m² at dt60/15 and 6.312e-4 J/m² at dt30/7.5, dominated
by native dynamics drift. Accumulated transfer and upstream residuals stay near
roundoff. The legacy six-hour run exactly reproduces all 7,920 archived scalar
values. Final precipitation remains timestep-sensitive; energy closure is not
cloud/climate accuracy qualification.

## Shared PyAM21–ocean–ice exchange and real-radiation setup

An opt-in workflow outside all package code now produces one interval-integrated
water/energy/momentum/radiation record and applies reciprocal transfers to actual
AM2 physics and the research Roquet ocean/category-ice/snow Hamiltonian. Open-water
and ice tile fluxes share a single implicit atmospheric bottom state; water and
stress use the same work-adjoint native mapping. Recipient midpoint stress work
and a local analytic surface first-law solve preserve the common energy contract.
AM2's upstream defect remains explicit, including in a deliberate injected-heat
test. Cached bottom LW is shared consistently when SST changes between calls.

Installed-wheel portable suite: 1,381 passed, 3 skipped before the final extra
rejection test; final focused suite 52 passed, 1 skipped. Both frozen comparisons
pass (142/93 tensors). Three six-hour integrations cover column exchange and actual
atmosphere/ocean/ice dynamics. Combined numerical energy residual falls from
1.347e-3 to 6.184e-4 J/m² when the coupled/atmosphere timesteps halve. Water and salt
remain near roundoff. See [evidence](../verification/atmosphere/pyam21/coupled/README.md).

The [real SeaEsfRad builder](../workflows/pyam21/RADIATION.md) is wired with explicit
GFDL data, content hashes, reference pressure columns and declared rectangular
climatology sampling. Actual execution is **unverified** because data acquisition
failed; all archived integrations above have radiation off, and synthetic tests
are identified as such. Ice skin temperature remains prescribed in the example;
a solved radiation/conduction skin balance and longer radiating qualification
are still required. No claim of production coupled climate readiness is made.

## Prognostic ice and one-year coupled campaign

The external [campaign](../workflows/coupled/CAMPAIGN.md) adds one prognostic
thermal node per ice/snow category, an extended energy budget and a 365-day
seasonal-radiation configuration with 6-hour snapshots through day 14 and daily
snapshots thereafter. See [verification](../verification/atmosphere/pyam21/campaign/README.md).
Packaged kernels/defaults are unchanged. Actual SeaEsfRad execution and year-long
stability remain unverified; the external data download returns HTTP 503 here.

## Coupled campaign ocean-process follow-up (2026-11-17)

The campaign now calls GM–Redi and eddy decay after each ocean dynamics/mixing
step, includes funded backscatter/budgeted biharmonic dissipation inside RK
stages, and remaps after the shared surface exchange and ice phase solve. ALE
uses the initial stretched layer fractions of current depth, retaining water
mass, salt, passives, TKE, eddy energy and the combined ocean/ice energy budget.
Explicit YAML switches, coefficients, actual call counts, local budget diagnostics
and NetCDF eddy energy make the enabled composition inspectable.

A separate numerical correction adds opt-in positive Redi flux funding. The old
rotated stencil drove near-zero TKE negative after two hours, even with increased
subcycling. The new local row limiter conserves inventories, uses the same flux
in the full-H ledger, and does not clip state or add floors; the default standalone
operator is unchanged. Sharp-store/coastal/variance/energy tests and unlimited
bitwise comparisons pass. Package kernels and upstream AM2 sources are unchanged.

External installed-wheel focused tests: 202 passed, 2 skips (optional GSW and real
radiation data); frozen portable comparisons: 142 corrected-Gibbs and 93 dry.
All runs here are CPU float64 with radiation off. At 256 cells, 24 air levels and
12 ocean layers, the requested ocean/physics 1200 s, atmosphere 240 s, radiation
3600 s and ten mixing/eddy substeps complete six hours, with numerical energy
residual -0.836594 J/m2, water -3.49e-10 kg/m2, salt -2.14e-16 relative.

The one-day attempt fails after 20 h in atmospheric CP energy transfer with
rapidly growing atmospheric winds. The reduced-ocean control also fails after
18 h 20 min; an atmosphere-dt60 trial fails after 20 h 40 min. These failures
remain unresolved and are not attributed solely to the newly enabled processes.
No full-day, real-radiation, 16,384-cell or long-run qualification is claimed.
See [raw statuses, diagnostics and limitations](../verification/atmosphere/pyam21/campaign/ocean-processes/README.md).

## Coupling momentum debugging (2026-11-17)

Stage tracing isolates the former 20-hour rejection to the external wind map.
At 70,800 s, the old mapping converts upstream increments no larger than
1.327/3.581 m/s into a native edge wind of 1107.762 m/s, from 46.015 m/s after
dynamics. Dynamics alone completes 24 hours. The energy solve rejects the next
interval rather than causing the initial acceleration.

The unweighted diagnostic fit gave tiny edges order-one impulse shares, then
divided by their tiny native kinetic mass. The shared coupling now uses native
kinetic-energy ownership weights in a local two-dimensional least-squares fit
and its exact transpose. On this mesh the largest mass-normalized read/scatter
eigenvalue falls from 144.270574 to 1.017534. Atmosphere and ocean/ice surface
exchange share the correction; reciprocal generalized momentum and finite
energy budgets are retained. Packaged dynamics, diagnostic reconstruction,
upstream sources and the legacy atmosphere map are unchanged.

External installed-wheel checks: 173 passed, 1 real-radiation skip; corrected-Gibbs
142 and dry-helper 93 portable comparisons pass. Short-edge finite drag, local
tangent reproduction and the independently differentiated adjoint are covered.
Default full-stack preflight replays bitwise. A fresh radiation-off 256-cell
two-day run completes with unchanged requested timesteps, 1440 GM–Redi substeps
and 144 ALE calls. Endpoint maximum air wind is 23.179 m/s (peak across recorded
stages 31.818 m/s), numerical energy residual -6.39188 J/m2, water -2.56e-9 kg/m2
and relative salt -6.42e-16. The day-one endpoint independently reproduces the
normal campaign driver's completed one-day result.

The separate exactly resting physics-only control exposes an upstream zero-gust
surface singularity after one accepted interval. Early stage checks now identify
the nonfinite surface outputs explicitly; no gust floor is introduced. Restart
schema 4 rejects old-map payloads. Real radiation, the user's mesh, full-year,
GPU and full-coupling backward behavior remain unqualified. See
[diagnosis, controls and reproduction](../verification/atmosphere/pyam21/campaign/momentum-map/README.md).

## Seasonal radiation setup follow-up (2026-11-17)

PR #89 merged after final CI passed. A requested actual-radiation preflight could
not initialize: the GFDL input tables are absent and the official FTP download
fails from this environment. No seasonal-radiation timestep was executed.

An independent configuration check exposed that the inherited BOMEX aerosol
flags are off, whereas the pinned SeaEsfRad supports both SW/LW aerosol effects
on. Radiation-on setup now selects those supported flags before caller overrides;
radiation-off behavior is unchanged. Two regressions call the upstream validator
without loading tables. External installed-wheel shared/campaign tests: 47 passed,
1 real-data skip. The 142/93 frozen portable comparisons also pass. See
[recorded attempt, data inventory and reproduction](../verification/atmosphere/pyam21/campaign/radiation-setup/README.md).

The user's downloaded-data run subsequently reached `AerosolOptics` and rejected
the inherited blank `aerosol_data_set`; BOMEX also omits `aerosol_optical_names`.
Enabling only the two band flags was incomplete. Setup now reads all five optical
settings from the pinned checkout's radiating ASTEX namelist before caller
overrides, retaining BOMEX composition, climatology and other physics options.
Regression tests parse unmodified upstream sections and exercise the actual
aerosol constructor through its optical-file open; they reproduce the user's
exception before the fix and verify model coverage and explicit overrides after
it. A radiation-off test needs no ASTEX file. External installed-wheel focused
checks: **50 passed, 1 real-data skip**; frozen portable comparisons: **142/93 pass**.
This verifies setup, not real radiative fluxes or a run on the user's mesh.

## Conservative radiation-to-host heating (2026-11-17)

The user reported a completed six-hour real-data segment with peak radiation
heating/flux residual 17.362 W/m² and accumulated upstream loss -43947.269 J/m².
SeaEsfRad's native radiation layer masses differ from the VFS host layer masses;
unchanged K/s rates therefore do not conserve their integrated heating.
`HostRadiationDriver` now transfers the native extensive SW/LW heating by
fractional-pressure overlap and divides by current host masses on every physics
call. Native cache/fluxes remain unchanged, cache geometry is restart state, and
the unresolved above-lid cap is explicitly included in the top host layer.
No heat is inferred from flux divergence to conceal an upstream discrepancy.

Installed-wheel focused tests outside the checkout: **105 passed, 1 real-data
skip**. Coverage includes independent overlap integration, 12/24/40 levels,
1200/3600 s physics/radiation cadence, fresh/cached calls, changing pressure,
day/night, solar renormalization, a retained injected +3 W/m² package defect,
bitwise atmosphere/coupled restart and failure rollback. Frozen **142/93**
portable comparisons pass. The analytic comparison reduces the old 3.14–5.92
W/m² residual to at most 1.71e-13 W/m² with zero flux change. The CSV now separates
signed cumulative global radiation residual from remaining upstream residual.
Actual optical tables are still unavailable here: no real-data reproduction or
complete attribution of the user's -2.035 W/m² loss is claimed. See
[evidence and reproduction](../verification/atmosphere/pyam21/campaign/radiation-grid/README.md).

## Stored-temperature ALE acceptance (2026-11-17)

The 16384-cell campaign reported an entropy rejection after second 67200.
The research temperature remapper evaluated iteration entropy at temporary T
but final entropy at reconstructed (h*T)/h. A deterministic 16-column cold
case passes the loop at -0.999716608 entropy tolerances, then fails the final
stored-state check at -1.000013766. Both evaluations now use the actual stored
candidate; this case converges in four iterations with unchanged 256/512-epsilon
energy/entropy tolerances. No clipping, timestep change or acceptance bypass.
Unsuccessful projections now report the worst column, normalized residual and
iteration count.

External installed-wheel focused suite: **99 passed, 1 optional GSW skip**,
including one/four-thread regression, conservative salt/tracer/TKE/EKE contents
and continued rejection of a stalled solve. Frozen **142/93** comparisons pass.
The user checkpoint and hash-matched mesh replay 67200→68400 s on Linux with
both old and corrected code; pre-ALE candidate tensors agree bitwise. Original
cached radiation is sufficient for that interval, with zero fresh optical
calls. Corrected maximum ALE column residual is 1.31845624e-5 J/m². The exact
Mac rejection, corrected fresh-table seasonal integration, GPU behavior and
long-run stability remain unverified. See [evidence](../verification/ocean/ale-cold-columns/README.md).

## Regional coupled compilation (2026-11-17)

The full coupled campaign now has opt-in `--compile` / `--no-compile` and
YAML `compile: false`. Nine full-graph Inductor regions cover shared
atmosphere/ocean/ice mesh operators, atmospheric mixture-energy gradients and
Roquet thermodynamics. State checks, limiting, nonlinear solver control,
PyAM21 column/radiation control and I/O remain eager; physical configuration
is unchanged. Per-instance setup avoids global monkey-patching. Compilation
policy is logged and included in restart identity; no silent eager fallback.
The campaign CLI consistently disables gradient tracking, while its Python
build/advance APIs retain their existing behavior.

Installed-wheel focused checks: **74 passed, 1 optional GSW skip**, followed by
**one passing native Inductor test** covering analytic radiation fresh/cache/fresh,
independent budgets and bitwise compiled restart. Native qualification is now
required in installed-wheel CI. Frozen **142/93** comparisons pass. The new
`validation/coupled_compile/run.py` measures complete intervals, separates setup
and warmup, compares fields/budgets, and checks compiled replay.
Radiation-off benchmarks complete with bitwise compiled replay: 256 cells
(6000 s) and the user's hash-matched 16384 mesh (4800 s). At 16384, steady
median interval time is 82.753 s eager / 65.454 s compiled (1.264× throughput;
two intervals per mode after two warmups on this Linux host). Final numerical
energy residual differs by 6.08e-10 J/m², with zero recorded water/salt drift.
A separate 512-cell/default-Lloyd pilot rejects in eager ocean mixing during
interval five; it is retained as a failed pilot, not a successful benchmark.
Real-table radiation, Mac/GPU performance, long-run stability and full-coupling
backward execution remain unqualified. See [compilation evidence](../verification/atmosphere/pyam21/campaign/compilation/README.md).

## ALE stored-entropy active constraint (2026-11-17)

The second uploaded campaign failure at second 38400 exposed an ALE
active-constraint stall. A stored candidate can violate entropy while the
linearized prediction remains inside tolerance; the resulting sub-ulp step
changes no temperature and repeats for 24 iterations. Once stored entropy
violates its bound, the solver now retains a zero entropy activation threshold
for that column through the remaining solve. Final acceptance tolerances,
iteration cap and physical configuration remain unchanged.

A deterministic 16-column case changes from 24-iteration rejection to acceptance
in four iterations. A nearby state of the actual column 16266 changes from a
24-iteration rejection to acceptance in three, with maximum energy residual
0.00685802 of tolerance. The unmodified uploaded checkpoint passes Linux under
both solvers; the exact Mac rejection is not claimed reproduced. Corrected
eager and native compiled diagnostic replays advance 38400→39600 using original
cached radiation, without fresh optics. A reusable one-interval diagnostic now
captures the pre-ALE state. Installed-wheel checks: **107 passed, 1 optional GSW
skip**; frozen **142/93** comparisons pass. Mac/fresh-table seasonal rerun and
long-run stability remain unverified. See [active-constraint evidence](../verification/ocean/ale-active-set/README.md).
