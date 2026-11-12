# Pass 4 — physical-field workflow

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
