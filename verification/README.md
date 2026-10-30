# Scientific verification: evidence plan and run contract

Baseline: **`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`**, the merge commit of
[PR #12](https://github.com/kjhall01/vfs-a/pull/12), merged
2026-10-30 at 19:24:30 UTC (confirmed through the GitHub pull-request API).
This is a source baseline, not a release or tag. Verification work may advance
independently; record its commit and actual driver hash as well.

The numerical implementation is frozen: operators, thermodynamics, limiter,
energy accounting, RK formulas and supported defaults must not be tuned to
improve results. Drivers, initial conditions, independent references, diagnostics
and plots may develop here. For a numerical defect, retain a minimal reproduction
and propose a separate core fix with explicit baseline consequences. Fixtures
and regression tolerances remain unchanged.

## Configuration labels

* **M-B** (primary): `MoistAtmosphere(..., stepper=BoundedSSPRK3())`;
  intrinsic heat release, authoritative Euler species amounts at each full-dt
  SSPRK3 stage. Positive vapor; no input floor or zero-water dry equivalence.
* **D-E** (control): `DryAtmosphere(..., stepper=DrySSPRK3())`;
  conservative, unbatched, tracer-free dry dynamics.
* **D-I**: `DryAtmosphere(..., stepper=DryIMEX(...))`, retained ARK4.
* **M-I**: `MoistAtmosphere(..., stepper=CenteredIMEX(...))`, retained ARK4;
  centered transport, stage validation, **no bounded-species guarantee**.

Initially CPU/float64, fixed positive lid, `direct_u_gl2_adjoint` column geometry.
Keep D-I and M-I evidence separate from explicit results. Existing dry full,
Schur and multilevel forward cases and moist full/Schur cases are starting
evidence, not qualification of every solver parameter choice. Record all resolved
solver settings, stage residuals, iterations and failures. Do not infer IMEX
backward, GPU, reduced-precision or long-rollout qualification.

## Claim-to-evidence tracker

IDs identify experiment families; a manifest's configuration and artifact hashes
identify a particular run. Existing unit/regression tests are **starting evidence**,
not completed paper experiments. No area below is yet a verified paper claim.
Figure/table names are planned, not generated evidence.

| Area / experiment IDs | Configuration and independent reference | Diagnostics and planned figure/table | Implementation status and limitations |
|---|---|---|---|
| **C1 Geometry and operators**: `C1-GEO-001` identities; `C1-ACC-001` accuracy | Shared geometry for all four paths. Analytic sphere area, grad/div weighted adjoint identity; manufactured spherical harmonics and tangent vector fields. | Area error, constant gradient, integrated divergence, adjoint residual; weighted L1/L2/L∞ gradient/div/curl errors vs cell size and grid rotation. F1 operator convergence, T1 identity residuals. | GEO driver implemented: Fibonacci resolution sequence, area/constant/adjoint/divergence metrics and sin(latitude) edge-gradient error. ACC full vector/curl/mesh-transfer/rotated-grid studies pending. A single harmonic does not establish general order or mesh quality. |
| **C2 Discrete energy derivatives**: `C2-DER-001` | M-B thermodynamic/kinetic/boundary energy, D-E control. Independently differentiate the scalar discrete energy, plus centered directional finite differences over an epsilon sweep; compare analytic gradients, not two calls to the same gradient routine. | Per u, ps, entropy/theta content and separate vapor/liquid/ice directions: absolute/scaled derivative error, Taylor remainder, cancellation plateau. F2 epsilon curves; T2 gradient components and units. | Existing dry/moist derivative and bounded one-step tests. Campaign driver pending. Discrete derivative correctness does not prove continuum accuracy; positivity restricts perturbations. Include topography and fixed-lid reservoir. |
| **C3 Instantaneous closure/work**: `C3-WORK-001` | M-B and D-E; D-I/M-I RHS and split separately. Independent scalar-energy derivative contracted with tendencies; pair physical work contributions. | Signed component powers (W), total residual (W and W/m²), cancellation scale, fixed-lid exchange; finite-step energy drift shown separately. F3 work exchange; T3 closure residuals. | Existing `tendency_terms`, energy gradients, boundary-rate tests. Driver pending. Heat allocation and Gibbs correction must be measured together; closure at one state is not finite-step conservation. Standalone HS thermal/drag sources must be separated if used. |
| **C4 Conservation and bounds**: `C4-LEDGER-001` | M-B smooth/sharp/reversed transport; D-E carrier/content controls. Independent global sums and graph incidence balance; actual bounded Euler amount ledger and SSP convex combinations. M-I is a distinct comparison without bounds. | Stage carrier/species totals (kg), local amount residuals, minima, dry fraction, all-neighbor bounds, limiter beta/activity, donor subcycles, exact ledger checks. T4 stage budgets; F4 species extrema/activity. | Existing fixtures, bounded budgets and deliberate perturbation tests. Campaign recorder pending. Never reconstruct authoritative amounts from RHS; keep all three full outer-dt Euler budgets. Record discrete limiter masks separately from floating tolerances. |
| **C5 Transport accuracy**: `C5-H-001`, `C5-V-001`, `C5-HV-001` | M-B transport ingredients, separately from coupled dynamics. Horizontal analytic rotation/deformation [R2]; vertical/combined manufactured carrier-consistent flows with matching boundary fluxes, specification pending. | Return-time weighted L1/L2/L∞ content and fraction errors, phase/diffusion, mass and extrema, refinement slopes. F5 error curves; T5 conservation/limiter activity. | No campaign driver. No generic dry tracer dycore available. Prescribed-flow harness must preserve existing transport implementation and declared carrier budgets. Horizontal reference alone does not establish 3-D accuracy. |
| **C6 Rest and balanced flow**: `C6-FLAT-001`, `C6-TERRAIN-001`, `C6-JWA-001` | D-E/D-I JW-A [R1]; M-B/M-I separately specified positive-vapor rest. Independent hydrostatic terrain-dependent ps and thermal initialization. | Initial balance residual, max/RMS wind, pressure/temperature drift, budgets vs grid/levels/dt/duration. F6 rest/drift histories; T6 initialization choices. | Flat-rest tests, JW builder available. Drivers and terrain-rest IC pending. Existing uniform-ps isothermal initializer is not terrain balance. Analytic JW balance need not be discrete balance; report ICBuildInfo and optional projection explicitly. |
| **C7 JW-B development**: `C7-JWB-001` | D-E/D-I original dry JW-B [R1], `perturb=True`. M-B/M-I moist extension only after independently documenting a thermodynamically consistent positive-vapor IC; never label it the original dry benchmark. | Surface-pressure and 850-hPa temperature maps, wind/vorticity development, reference error norms with common projection, resolution/timestep dependence and budgets. F7 evolution/reference differences; T7 errors. | Existing dry builder and small fixture control only. Campaign/output/reference acquisition pending. Pin reference dataset version/checksum and interpolation procedure before comparing; no assumed growth envelope. |
| **C8 Hydrostatic mountain wave**: `C8-MW-001` | D-E/D-I first; moist cases separate extensions. Candidate small-amplitude broad ridge, stable background. Independent linear hydrostatic reference **not yet selected**: must match fixed lid, sphere/local approximation and time window. | Perturbation wind/pressure phase and amplitude, vertical structure, momentum/energy flux, linearity and resolution study. F8 reference profiles/phase; T8 flux errors. | Specification/IC/reference/driver blocked on boundary-compatible design. No absorbing top or open lateral boundaries. Assess reflection and recirculation; do not compare to an unbounded radiation solution without quantifying boundary effects or add new physics here. |
| **C9 Time integration/solver sensitivity**: `C9-TIME-001`, `C9-SOLVE-001` | M-B/D-E SSPRK3; separate D-I/M-I ARK4 [R3]. Independent high-accuracy DOP853 reference on the same spatial semi-discretization for smooth cases; tighter dt reference for bounded stage-dependent transport. | State-component weighted errors vs dt at fixed spatial grid, observed slopes, stage residual/iteration counts, work/energy drift vs solver tolerance, failure records. F9 temporal curves; T9 solver costs/errors. | Existing small DOP853 ARK4 and solver-failure tests. Campaign driver pending. Bounded Euler fluxes depend on dt/limiter branch; generic RHS integration is not its reference. Keep spatial, temporal and solve errors separate; no IMEX backward claim. |

## References and reference selection

* **R1** Jablonowski & Williamson (2006), *A baroclinic instability test case for
  atmospheric model dynamical cores*, QJRMS, DOI
  [10.1256/qj.06.12](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1256/qj.06.12).
  [Author-hosted paper](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/qj_jablonowski_williamson_2006.pdf).
  Original dry initial-value specification, not a moist reference solution.
* **R2** Lauritzen et al. (2012), *A standard test case suite for two-dimensional
  linear transport on the sphere*, [GMD 5, 887–901](https://gmd.copernicus.org/articles/5/887/2012/).
  Source for horizontal protocols, not vertical or fully coupled moist dynamics.
* **R3** Kennedy & Carpenter, *Additive Runge–Kutta Schemes for
  Convection–Diffusion–Reaction Equations*,
  [NASA/TM-2001-211038](https://ntrs.nasa.gov/citations/20030025779).
  Check retained tableau/split against the specification; method order alone
  does not qualify the VFS stage solves or composition admissibility.

Analytic sphere identities and finite-difference/Taylor tests above are independent
mathematical references to be written out with each experiment. No mountain-wave
reference is asserted prematurely. Held–Suarez climate verification remains a
separate future experiment: forcing and runnable example exist, but spin-up,
averaging, uncertainty and reference statistics are not yet specified.

## First runnable measurement

Build the pinned commit in a separate checkout (no release/tag):

```sh
git worktree add --detach /tmp/vfs-verification-baseline 8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648
python -m build --wheel /tmp/vfs-verification-baseline --outdir /tmp/vfs-verification-wheels
python -m pip install /tmp/vfs-verification-wheels/*.whl
# From outside the source checkout, using an absolute repository path:
cd /tmp
python -I /path/to/vfs-a/tools/run_verification.py \
  --repo /path/to/vfs-a --output /tmp/vfs-c1-run \
  --cells 16 64 256 --lloyd-iters 0 --seed 20261030
```

Use a dedicated environment; retain the wheel and its SHA256 alongside campaign
artifacts. Dependencies are recorded rather than pretending a Torch version alone
ensures bitwise portability. The driver checks every installed tracked runtime
file against the baseline and rejects unexpected Python modules. It requires an
installed import, never overwrites an existing output directory, records driver
hashes/checkout status and CPU/backend/distribution details, and saves generated
mesh buffers and random fields. `manifest.json` marks completion/failure and hashes
artifacts; `metrics.json` contains measurements, without acceptance thresholds.
The executed driver and environment collector are archived with the run.
Terminated processes may leave `running` runs;
they are incomplete, never paper evidence.

The geometry experiment has no model, column, timestep or thermodynamics. Future
dycore manifests must additionally record exact vertical coefficients, terrain,
IC choices, physical constants, resolved solver configuration, dt, duration,
cadence and output-variable definitions. Archive actual arrays and reference data
with hashes, not just seeds/URLs. Report weighted norms with units and explicit
zero-denominator handling, refinement sequences, uncertainty and failures. Never
turn portable migration tolerances into scientific acceptance criteria.

See [current paths](../docs/CURRENT_PATHS.md),
[benchmark setup gaps](../docs/VERIFICATION_HANDOFF.md), and
[initial infrastructure results](RESULTS.md).
