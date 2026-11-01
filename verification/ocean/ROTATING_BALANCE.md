# Rotating-flow diagnosis, with production frozen

Starting main: `e959b7005832a5041e082b6e300838e15bfbaa9e`. #26 merge
`735785c7cf5edae2f363f8c61ca7d0c8409496e3` and #27 merge `e959b700...`
are ancestors. This isolated ocean investigation changes no runtime code,
fixtures, tolerances or defaults. Diagnosis precedes any numerical proposal.

The five-day axis-aligned Williamson case is replayed before operator probes.
The 64-cell mesh is frozen published input; larger meshes are regenerated,
explicitly not a byte-for-byte claim about omitted original mesh archives.
The baseline's exact ocean/atmosphere source checks remain enabled.

The diagnostic driver separates point/cell/dual sampling, gradient truncation,
kinetic reconstruction, vorticity reconstruction, raw/shared-skew/layer-skew
rotation, and local vector moments. Results and limitations follow in a separate
evidence commit. No ablated expression is used to advance a benchmark state.

## Conclusion and limits

The evidence isolates two spatial-consistency limitations, not a time-stepping
failure or a failure of the analytic energy derivative:

1. The existing rotation reconstruction has nonzero local constant-vector
   moments on these meshes. Exact vorticity does not remove its leading error.
2. The cell kinetic-energy derivative has a geometric second-moment defect.
   Differencing the resulting cell-scale kinetic-value errors amplifies them.
   Applying the same gradient to analytic kinetic values converges much better.

The extra skew projection in layer-dependent mass is not the main high-resolution
error. Nor is the shared geometry's metric-skew projection: its increment in
these probes is at roundoff. These findings do **not** prove a universal lack of
convergence or identify an isolated indexing defect. On irregular C grids,
energy-neutral rotation need not reproduce local vectors accurately. A proposed
correction must satisfy both properties, and must be assessed with the kinetic
metric and velocity location. No numerical correction is promoted here.

## Equations and independent probes

Let `I_ec` be the two side-area interpolation weights, `A_c` cell area and
`A_e=dc_e dv_e` (m²). With stored edge-normal speed `u_e` (m/s) and layer
thickness `h_c` (m), the candidate uses

```text
M_e = rho0 A_e I(h)_e
K_c = (1 / (2 A_c)) sum_e I_ec A_e u_e²       [m²/s²]
F_e = I(h)_e u_e                             [m²/s]
h_dot = -D F                                [m/s]
u_dot = -G(g eta + K) + (M R_q - R_q^T M)u/(2 M)
```

`K_c` is `H_h/(rho0 A_c)`'s kinetic part, differentiated at fixed **stored
velocity** and passive contents. It is not a squared least-squares cell wind.
The existing analytic derivatives match independent AD; replacing only K by
an apparently more accurate reconstruction would break the demonstrated work
pairing. The usual grad/div pairing cancels Bernoulli and continuity work.
`R_q` contains the symmetric mean of edge absolute vorticities; the transpose
holds that vorticity fixed. Antisymmetrization cancels rotation work, but does
not enforce accuracy, PV consistency or a Poisson/Jacobi structure.

For a vector parallel transported from the target to nearby edges, local
rotation consistency requires `sum_j R_ej PT(n_j) = n_e × r_e`, up to curvature
truncation. The driver reports the weighted norm of this residual. Likewise,
with edge normals transported to the cell generator, the kinetic second moment
is `Q_c = sum_e I_ec A_e n_ec n_ec^T / A_c`. Reproducing local `|v|²/2` requires
`Q_c` to approach the tangent projector `P_c = Id-r_c r_c^T`. We report its
Frobenius residual, not a subjective pass/fail threshold.

The kinetic-force error is separated exactly as

```text
-G(K_discrete) - (-grad K_exact)
  = -G(K_discrete - K_exact) + [-G(K_exact) - (-grad K_exact)].
```

The vector errors add, **their RMS norms do not**. The second term tests the
unchanged gradient with analytic point values; the first exposes reconstruction
error. This avoids attributing every imbalance to the gradient operator.

Sampling is also explicit. Williamson's stored velocity is at a Voronoi-face
midpoint. Curl instead uses that speed times the *dual* edge length in a dual
triangle circulation. For `v=U0 axis×r`, exact velocity along a dual great-circle
side is constant, `U0 axis·normalize(c0×c1)`. Its curl gives a triangle-average
vorticity, which need not equal a vertex point value. Independent arc sampling
checks this formula. Exact face-average normal speed is the midpoint value times
`sinc(dv/(2a))`. Cell-average h is independently integrated using the existing
geometric Gauss–Duffy rule at orders 6 and 10. None of these diagnostic samples
is substituted into the published benchmark or presented as a balance projection.

Reference: [Williamson et al. (1992)](https://doi.org/10.1016/S0021-9991(05)80016-6),
test 2, as specified in the existing [benchmark](WILLIAMSON.md).
For the distinction between energy neutrality and PV/accuracy, and the C-grid
kinetic pairing, see [Ringler et al. (2010), sections 3.6–3.7](https://www2.mmm.ucar.edu/people/skamarock/Ringler_et_al_JCP_2009.pdf).
These papers do not qualify VFS's particular interpolation or mesh sequence.

## Results (CPU float64, one thread)

The five-day dt=150 s replay gives h/u RMS pairs **38.06527 m / 2.886079 m/s**,
**10.81708 m / 0.8365083 m/s**, **5.146204 m / 0.3252532 m/s** at 64/256/1024
cells. All five 64-cell endpoint arrays reproduce exactly from published frozen
inputs. Regenerated larger meshes reproduce the reported errors, but not bitwise
endpoints: maximum h discrepancies are 2.10e-10 and 2.38e-10 m. The omitted
original larger mesh files are not reconstructed or republished as originals.

Ten-step timing estimates were 2.5/2.9/5.6 seconds for each five-day run before
launch. The extension to 4096 cells is **stationary only**, not a larger-duration
campaign. All numbers, units, source/input hashes and omission reasons are in
[rotating_balance_results](rotating_balance_results).

| Cells, Lloyd iterations | Rotation error RMS m/s² | Kinetic-force error RMS m/s² | Gradient of analytic K error RMS m/s² | Rotation local-vector defect | Kinetic second-moment defect |
|---|---:|---:|---:|---:|---:|
| 64, 100 | 1.38193e-4 | 5.95637e-6 | 2.20407e-6 | .03894 | .03993 |
| 256, 100 | 6.81004e-5 | 1.33704e-5 | 6.73534e-7 | .03703 | .03355 |
| 1024, 100 | 5.81204e-5 | 1.23451e-5 | 1.59914e-7 | .02915 | .03002 |
| 4096, 100 | 5.89089e-5 | 4.41661e-5 | 8.33162e-8 | .04014 | .03499 |

At 1024 cells, substituting exact q in the diagnostic gives rotation error
5.81478e-5 m/s². The layer-metric skew increment is 4.76078e-6 m/s², decreasing
to 1.42284e-6 at 4096 cells. The unchanged raw and geometry-skew stencils have
indistinguishable exact-q errors at reported precision. These ablations rule
out those two skew operations as the dominant plateau in this sequence.

At 1024 cells, 500 instead of 100 Lloyd iterations reduces rotation error to
3.44247e-5 m/s² but kinetic-force error becomes 1.29300e-5 m/s². This is a
**controlled mesh-sensitivity measurement**, not a tuned replacement benchmark.
Unit-speed manufactured rotations around z, x and (1,2,3)/sqrt(14) all show
reconstruction error; there is no evidence that a reversed sign or choice of
geographic axis alone explains the result. Normal/radius dot products and total
sphere area are at floating-point precision. Face/dual midpoint displacement
remains about 2% of dc, rather than disappearing rapidly relative to grid size.

At 1024 cells curl error changes from 4.32287e-7 to 7.91847e-8 s^-1 when exact
dual circulation replaces the face sample **only in the diagnostic**. The
remaining error includes triangle-average versus vertex-point interpretation.
Cell-average minus point h is 1.67227 m RMS; order-6/order-10 quadrature differs
by 2.08e-12 m RMS. Its pressure-force difference is 5.51190e-6 m/s², smaller
than the rotation error. Sampling matters, but does not explain away the
reconstruction plateau. The original dt-halving evidence remains separate:
1024-cell endpoint dt=300→150 differences are .02751 m and .001207 m/s.
Time refinement cannot remove an initial spatial RHS inconsistency.

Finite-step relative energy drift in the reproduced five-day runs is
-5.23233e-9, -1.21951e-8 and -2.64546e-8. Instantaneous normalized work is reported
separately; neither is substituted for the other. Volume/content changes and
minimum h are retained in each daily series. No general positivity claim follows.

## Reproduce and validate

Install the unchanged starting-main wheel; copy tests/tools/examples outside
the checkout. All drivers use the existing exact source guard, including the
explicit atmospheric structural/numerical and ocean revision pins internally.
The published 64-cell frozen input directory is `verification/ocean/williamson_results`.
For the larger replay omit `--inputs`; generated geometry is saved in the new
private run directory. Output directories must not already exist.

```sh
python -I tools/verify_ocean_williamson.py --repo /path/to/repo \
  --output /new/replay64 --inputs /path/to/frozen-inputs --cells 64 --dt 150 --days 5
python -I tools/verify_ocean_williamson.py --repo /path/to/repo \
  --output /new/replay-large --cells 256 1024 --dt 150 --days 5
python -I tools/investigate_ocean_balance.py --repo /path/to/repo \
  --output /new/diagnosis --cells 64 256 1024 --quadrature
python -I tools/investigate_ocean_balance.py --repo /path/to/repo \
  --output /new/stationary4096 --cells 4096
python -I tools/investigate_ocean_balance.py --repo /path/to/repo \
  --output /new/mesh-sensitivity --cells 1024 --lloyd-iters 500
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

Installed wheel: **619 passed, one CUDA skip**; both atmospheric portable tools
and both ocean examples pass. The historical ocean archive has 1400/1416 exact
arrays on this macOS environment; 16 source/forcing arrays have differences up
to 1.73e-18 in native units. Existing portable migration tolerances pass; the
exact report is retained as a difference, not called exact success. Four new
tests check independent arc integration, parallel transport, the sparse
transpose against a dense oracle, and deliberate rotation corruption.
Runtime is byte-identical to starting main, so no changed numerical fixture,
external-wave or temporal method is being qualified here. Accelerator, reduced
precision, long backward rollouts, and stratified balance remain untested.

## Separate numerical proposal and next milestone

A future shared-geometry proposal should test simultaneous local rotation
moments and metric-skew constraints, plus cell kinetic second moments, on frozen
irregular meshes and a second mesh family. Simply replacing the ocean rotation
with a fitted stencil and then skewing it again can undo consistency; replacing
K alone breaks its energy derivative. Any kinetic-metric change must derive
H_u, H_h, continuity flux and momentum together. No such change is implemented.

Affected shared consumers include atmospheric dry/moist rotation, atmospheric
initial-condition rotation inversions, mesh tangential KE diagnostics, and both
ocean equation sets. A shared change needs a separate PR with atmospheric
states/tendencies/ledgers/gradients/ARK4 regression evidence and explicit intended
differences; it must not enter through an ocean-only default change. The new
manufactured-field probes are the minimal reproduction for that proposal.

Meanwhile the independent next milestone is transport admissibility: measure
actual *carrier* outflow per cell/layer, derive the Euler/SSPRK3 restriction, and
probe h/hT/hS/hq consistency. A Celsius temperature need not be positive. Do not
run longer integrations before distinguishing h positivity, salinity and tracer
bounds, and the existing minimum-thickness rejection policy.

## Hosted CI distinction

The initial PR run [41263050551](https://github.com/kjhall01/vfs-a/actions/runs/41263050551)
passed. Push run [41263013697](https://github.com/kjhall01/vfs-a/actions/runs/41263013697)
passed all 164 exact same-environment ocean integration tensors but failed the
historical archive command: five forcing/source arrays differed by up to
1.30104e-18. This predates any runtime change (there are none here), and the
required portable pytest comparison passed.

A separate CI-only commit makes that historical **exact** command explicitly
informational, preserving its nonzero exit, report and step outcome. Exact
same-environment equality remains required. The existing ocean archive pytest
gate still uses its unchanged 512-epsilon per-array migration allowance, including
all 1416 arrays. No fixture, tolerance, atmospheric check or atmospheric
verification policy is changed. A green portable gate is not an exact match.
