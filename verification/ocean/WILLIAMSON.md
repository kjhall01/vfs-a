# Williamson steady rotating flow: measurements with frozen operators

> Status reconciled against main `e959b7005832a5041e082b6e300838e15bfbaa9e`:
> #26 and #27 are merged and their heads are ancestors of main. Draft/unmerged
> language below describes the original publication, not current status.
> The reference solver remains default; the constant-density candidate remains
> explicit. Current investigation: [rotating-flow diagnosis](ROTATING_BALANCE.md).

This is Williamson et al. (1992) shallow-water test case 2, with the rotation
axis aligned with the geographic pole (alpha=0). It measures the explicit
`ConstantDensityDynamics` candidate; the archived ocean remains the default.
No package runtime file changes from the integration parent. No damping,
balanced-state adjustment, clipping, hidden projection or parameter tuning
was added. Initial data are point samples of the continuum solution.

## Reference and equations

Reference: Williamson et al., *A standard test set for numerical approximations
to the shallow water equations in spherical geometry*, JCP 102 (1992), 211–224,
[DOI](https://doi.org/10.1016/S0021-9991(05)80016-6).
The explicit alpha=0 specification and five-day diagnostics also appear in
the [Polaris geostrophic task](https://docs.e3sm.org/polaris/main/users_guide/ocean/tasks/geostrophic.html).
Polaris acceptance thresholds are not adopted here.

For latitude phi and north-pole unit vector k, the exact steady fields are

\[
u=U_0\cos\phi,\qquad v=0,\qquad
h=h_0-\frac{a\Omega U_0+U_0^2/2}{g}\sin^2\phi.
\]

| Symbol | Meaning | Value / units |
| --- | --- | --- |
| a | Sphere radius | 6,371,220 m |
| g | Gravity | 9.80616 m/s² |
| Omega | Planetary rotation rate | 7.292e-5 s⁻¹ |
| U0 | Equatorial zonal speed | 2 pi a / (12 days), m/s |
| h0 | Equatorial water thickness | 29,400/g m |
| h | Total water thickness, one layer | m |
| eta | Free surface relative to flat bottom depth h0 | h-h0, m |

The Cartesian velocity is U0 (k cross r), with r a unit-sphere position.
Its dot product with the mesh edge normal supplies u_e. Thickness is sampled
at cell centers, velocity at Voronoi edge midpoints. These are not exact cell
or edge averages. The physical references are the same samples at all times.
The alpha=0 formulation needs no changed Coriolis buffers or shared geometry.

The vector-invariant momentum equation contains rotation, -g grad(h), and
-grad(|u|²/2). Relative vorticity is 2 U0 sin(phi)/a. The analytic contributions
cancel. A test independently differentiates the analytic Bernoulli function
with autograd and projects its gradient onto the sphere to verify this balance.
The production RHS continues to use its existing analytic energy derivatives.

## Experiment and evidence

CPU float64, one numerical thread; 64, 256 and 1,024 Fibonacci/Voronoi cells,
100 Lloyd iterations, one layer, no coast, forcing or remapping. Each mesh is
run for five days at dt=600, 300 and 150 s with unchanged production SSPRK3.
Temperature=10 degC, salinity=35, and a unit passive tracer are initialized.
The original local campaign archived all actual mesh buffers and initial/final
fields, with daily diagnostics and initial numerical/analytic acceleration terms.
The GitHub publication is a subset; see the size-limited publication note below.

L1 and L2 use cell area for thickness and dc*dv for normal velocity; L2 is an
RMS, in m or m/s. Linf is the largest absolute sample error. Effective mesh
spacing is sqrt(4 pi a²/N). Both absolute errors and observed refinement orders
are reported, without scientific pass/fail cutoffs.

At day 5, dt=150 s:

| Cells | h RMS error (m) | h maximum error (m) | u_e RMS error (m/s) | u_e maximum error (m/s) |
| --- | ---: | ---: | ---: | ---: |
| 64 | 38.0653 | 107.341 | 2.88608 | 7.60563 |
| 256 | 10.8171 | 33.3192 | 0.836508 | 2.59381 |
| 1,024 | 5.14620 | 24.3117 | 0.325253 | 1.37963 |

Pairwise spatial orders are **1.815, 1.072** for thickness and **1.787, 1.363**
for normal velocity. Errors decrease, but this does not demonstrate a uniform
second-order regime. Changing dt from 300 to 150 s at 1,024 cells changes the
endpoint by **0.0275085 m RMS** and **0.00120661 m/s RMS**. These differences
measure timestep sensitivity; they are not rigorous bounds on temporal error.

### Balance and term consistency

| Cells | Initial momentum RHS RMS (m/s²) | Pressure term error RMS (m/s²) | Kinetic term error RMS (m/s²) | Rotation term error RMS (m/s²) |
| --- | ---: | ---: | ---: | ---: |
| 64 | 9.62411e-5 | 5.52457e-5 | 5.95637e-6 | 1.38193e-4 |
| 256 | 6.38649e-5 | 1.68824e-5 | 1.33704e-5 | 6.81004e-5 |
| 1,024 | 5.68013e-5 | 4.00829e-6 | 1.23451e-5 | 5.81204e-5 |

Pressure-gradient error decreases strongly; rotation and kinetic-gradient error
decrease weakly or nonmonotonically in this mesh sequence. Rotation has the
largest term error. The maximum momentum residual actually increases from
2.36374e-4 to 3.28603e-4 m/s². This warrants a separate investigation of rotation,
kinetic reconstruction, mesh geometry, and sampling error. It does not isolate
a single defective operator or justify modifying one in this verification PR.

Initial cancellation-normalized energy rates are **2.666e-16, 4.723e-16,
9.338e-17**; absolute rates are **918, 1,224, 127.5 W**, respectively. The
normalization is the sum of absolute local H_u*u_dot and H_h*h_dot terms.
These energy cancellations coexist with the nonzero balance residuals above.

Across all nine runs and daily samples, maximum absolute relative volume drift
is **4.249e-16**. At dt=150 s, relative five-day energy changes are
**-5.232e-9, -1.220e-8, -2.645e-8** for the three meshes. SSPRK3 does not exactly
conserve energy. The unit tracer remains uniform in these cases; that is not a
general positivity result. hT and hS inventories are passive contents, not a
thermodynamic heat/salt energy closure.

The pre-rendered figure is omitted under the publication size cap. Regenerate it
from the retained JSON with `tools/plot_ocean_williamson.py` (command below).

## Reproduce and inspect

Build/install the integrated wheel, copy tools/tests outside the checkout, then:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I tools/verify_ocean_williamson.py \
  --repo /path/to/repo --output /new/run --cells 64 256 1024 \
  --dt 600 300 150 --days 5 --lloyd-iters 100
python tools/plot_ocean_williamson.py --results /new/run/results.json --output /new/figure.png
```

The published snapshot retains only the 64-cell frozen inputs. For their replay,
use `--cells 64 --dt 600 --days 5 --lloyd-iters 100` and add
`--inputs /path/to/repo/verification/ocean/williamson_results`.
The full sweep above generates new geometry; exact replay of the 256/1024-cell
cases requires their omitted original inputs.
New output directories are required; references are never overwritten.
The source guard pins the structural/atmospheric revisions and exact ocean #24
tree. The manifest records all runtime hashes, environment, driver bytes and
artifact hashes. The original campaign recorded an uncommitted driver snapshot;
its archived SHA-256 exactly matches the subsequently committed driver. A
separate replay from that committed driver reproduces the 64-cell, dt=600 s
five-day fields and metrics exactly; see the compact replay summary in `recovery_results.json`. The detailed replay
manifest is omitted from publication because it includes machine metadata.
Local driver commit `589f466` is published with identical Git tree as
`7e53cd453bb8fea49d654fcf699b0937c6e45c20`. Commit metadata differs because the
GitHub publication API creates the commits; archived source bytes are identical.

`williamson_results/results.json` contains all nine trajectories and orders;
`initial_terms_64.npz` and `initial_terms_256.npz` contain component accelerations
and references; `inputs_64.npz` contains every 64-cell mesh buffer and initial field;
`final_*_dt*.npz` contains the five stored prognostic tensors. The archived
scripts are evidence snapshots, not a second runtime implementation.

Three focused tests verify the independent continuum balance, exact frozen-mesh
replay and rejection of changed parameters/initial fields, and endpoint metric
consistency. They pass on the installed integration wheel, whose broader suite
passes 612 tests with one CUDA skip. Hosted full-suite results remain separate.
Integration #26's push and PR checks passed at `018da8c`; the verification
draft additionally runs a small driver execution in CI. That execution is not
an accuracy acceptance gate or a replacement for the archived five-day sweep.

No coastal, tilted-axis, multilayer shear, active stratification, PV/enstrophy,
GPU/compile, long-run, or general positivity claim follows from this test.
Investigate the weak initial rotation/kinetic consistency before promoting the
candidate default or treating its energy closure as dynamical qualification.

## Recovery validation — 31 October 2026

The interrupted publication was recovered from local evidence commit `acaf12a`.
All 25 artifacts listed in the original campaign manifest retain their recorded
SHA-256 hashes. All 97 runtime Python files match the installed integration wheel;
the verification branch changes no runtime files or historical fixtures.

With tests, tools and examples copied outside the checkout, the full installed-wheel
portable suite passes **615 tests, with one CUDA skip** (55.88 s). An initial
check-directory setup omitted `examples/bounded_moist.py`, causing five missing-file
failures; copying the unchanged examples resolved them. No source or tolerance
change was needed. The final summary is recorded in `recovery_results.json`.

A new 64-cell, dt=600 s, five-day run from the archived inputs exactly reproduces
all five final prognostic arrays, every daily metric, and the initial balance
report. These are CPU/float64, same-environment checks. The original nine-run
campaign and its source snapshots remain untouched. Verification-branch hosted
CI is pending publication; integration #26's hosted checks passed separately.

## Size-limited publication — 31 October 2026

At the user's direction, this recovery omits files larger than a conservative
128 KiB upload cap. This is a publication cap, not a GitHub storage limit.
The exact paths, original byte lengths and SHA-256/Git blob hashes are recorded in
[publication_omissions.json](publication_omissions.json). Four files exceed the cap:
`inputs_256.npz`, `inputs_1024.npz`, `initial_terms_1024.npz`, and `williamson.png`.
They are not split, replaced, re-encoded, or added to the publication history.

All nine trajectories' JSON measurements, all nine final-state arrays, the
64-cell frozen inputs, the two smaller initial-term archives, source snapshots,
tests and plotting driver are retained. Every retained campaign artifact is
byte-identical to the recovered evidence. Automatic approval review blocked the
detailed replay report because it contains local machine/environment metadata.
Both that report (`williamson_replay.json`) and the similar complete campaign
manifest (`williamson_results/manifest.json`) are omitted from publication.
They remain unchanged in the original local checkout. The original artifact-hash
mapping is retained in `publication_omissions.json` without machine metadata;
three of its 25 artifacts are intentionally absent from this partial publication.
The figure was outside that manifest. Local recovery results above describe the
original complete checkout.

The retained 64-cell inputs support the documented frozen replay. A new full
sweep remains runnable without `--inputs`; regenerating omitted inputs does not
establish byte-for-byte reproduction of the original larger-grid runs. Runtime
code, historical regression fixtures and tolerances remain unchanged.

Publication checks: all 22 retained artifacts listed in the original campaign manifest
match their SHA-256 hashes. The 97 runtime Python files match the installed
integration wheel. All three focused Williamson tests pass outside the checkout;
corrected-Gibbs-v1 (142 tensors) and dry-helper (93 tensors) portable comparisons
pass. The 615-pass full-suite result above is the prior recovery run.
