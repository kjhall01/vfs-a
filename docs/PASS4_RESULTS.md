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
