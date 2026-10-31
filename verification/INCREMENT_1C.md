# Increment 1c — prescribed horizontal transport (C5)

This is a small transport investigation, not atmospheric or coupled-energy
qualification. PRs #13/#14 are merged. This delivery depends on unmerged #16's
[approved stage extraction](SSPRK3_EXTRACTION_RESULTS.md). Numerical baseline
remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`; the separately approved
interface commit is `ae132773e621671f70bfa6a698eebb1a82cffa94`. This dependent
PR changes no numerical package source, fixtures or tolerances.

Read [results and limitations](INCREMENT_1C_RESULTS.md), particularly the
constant-composition counterexample. Horizontal infrastructure is runnable;
transport accuracy and constant-composition preservation are not qualified.
Vertical/combined transport, terrain-rest construction and large campaigns
remain queued. No 1024/4096/16384-cell runs were launched.

## Original specifications and adaptations

Original sources were read for the boundary proposal:

- Williamson et al., [ORNL/TM-11895, §3.1, equations 75–79](https://www.osti.gov/servlets/purl/5232139),
  published in [JCP 102 (1992), 211–224](https://doi.org/10.1016/S0021-9991(05)80016-6).
- Nair & Lauritzen, [JCP 229 (2010), 8868–8887](https://doi.org/10.1016/j.jcp.2010.08.014),
  [author manuscript](https://www.image.ucar.edu/staff/rnair/NairLauritzen_BM_JCP2010.pdf),
  equations 14–16 (Gaussian hills) and 28–33 (Case 4).

Longitude lambda increases east from +x toward +y; latitude theta increases
north toward +z. Both are radians. Unit position is
`r=(cos(theta)cos(lambda), cos(theta)sin(lambda), sin(theta))`.
R=6,371,220 m, T=12 days=1,036,800 s. u/v denote east/north m/s.

Solid rotation: U=2*pi*R/T,

```
u = U*(cos(theta)*cos(alpha) + sin(theta)*cos(lambda)*sin(alpha))
v = -U*sin(lambda)*sin(alpha)
v_cart = (2*pi*R/T)*(-sin(alpha),0,cos(alpha)) cross r
```

alpha=0 gives equatorial tracks; alpha=pi/4 is our explicitly chosen tilted
variant, not one of Williamson's four recommended orientation values. Reference
points undergo inverse Rodrigues rotation. The Gaussian multispecies field
below replaces Williamson's original cosine bell and shallow-water height.

Nair–Lauritzen Case 4 uses R*=1, T*=5, k=2, lambda'=lambda−2*pi*t*/5:

```
u* = 2*sin(lambda')**2*sin(2*theta)*cos(pi*t*/5) + (2*pi/5)*cos(theta)
v* = 2*sin(2*lambda')*cos(theta)*cos(pi*t*/5)
```

Set t*=5*t/T; multiply u*,v* by 5R/T. In the rotating frame the deformation
is a fixed vector field multiplied by cos(pi*t*/5). Its integrated coefficient
over the full period is zero; background rotation makes one full turn. Thus
material returns exactly to its starting point at T, but numerical mixing need
not reverse. At T/2 the deformation reaches its maximum, while background wind
is still nonzero. At T the deformation velocity has reversed sign. All production
timesteps remain positive. Independent DOP853 trajectories provide intermediate
references, with rtol 2e-11 versus 2e-12 refinement; reference integration backward
to departure points does not reverse any production Euler timestep.

## Initial and reference fields

Flat one-layer column, p_top=5000 Pa, initial ps=100000 Pa, T_air=280 K,
`phi_full_mode="direct_u_gl2_adjoint"`. One layer still has two cell/interface
composition fields. They must both be measured, not assumed equal.

Gaussian hills: `f=sum exp(-5*|r-r_i|²)`, centers (5*pi/6,0), (7*pi/6,0).
VFS water fractions in kg/kg are qv=.01+.001f, ql=.001f, qi=.0001f;
dry is 1−qv−ql−qi. The .01 vapor background is an initial condition, not a floor.
The constant control uses (.01,.001,.0001). The empty-condensate case retains
Gaussian vapor but replaces f in condensates with the sum of compact cosine
bells, `.5*(1+cos(pi*d/.5))` for angular distance d<.5 radians, zero outside.
No phase conversion, vapor clipping or zero-vapor extension is added.

These fractions are **cell averages**, initialized identically at both CP
interfaces with uniform carrier. Independent spherical polygon integration uses
triangles from the cell generator to consecutive dual vertices. A normalized
barycentric map has Jacobian `|det(a,b,c)|/|a+u(b-a)+v(c-a)|³`; a Gauss-Duffy rule
includes the `(1-u)` factor. Divide integrals by independent exact spherical
triangle areas. Report area quadrature error and disagreement with production
areas. Orders 8 and 12 estimate reference error; these are estimates, not rigorous
bounds. An independent global Gaussian integral tests the quadrature. Exact
return references reuse the analytic initial field, not the evolved numerical
field. Compact-support quadrature is less accurate and is reported separately.

## Production boundary and execution

[Driver](../tools/run_transport_verification.py) and
[analytical references](../tools/transport_references.py) stay outside `vfs/`.
At each RHS time, sample Cartesian analytical velocity at the dual-edge midpoint
and dot with the production edge normal. No streamfunction construction or flux
projection is used. Sampled continuum nondivergence is **not** discrete
nondivergence: record `max(abs(mesh.div(u_e)))` in s^-1 at every stage.

`advance` → `bounded_ssprk3_with_info` → prescribed wind replacement →
`BoundedGibbs.euler_with_info` → production transport primitives and graph →
Gibbs candidate reconstruction → bounded transfer/donor remap/common limiter →
entropy/heat and Euler state → authoritative water amounts/area.
The extracted sequence uses RHS times t, t+dt, t+dt/2, all with budget dt. The
second convex combination represents the half-time stage, hence nonmonotonic
RHS times. No RK formula is copied into the harness.

Momentum tendencies are evaluated but the resulting wind is unused: replace
wind before every next Euler call and again with analytical wind at t+dt on
return. The RK machinery combines scratch wind tensors; those values never
enter prescribed transport. Pressure/carrier, entropy, temperature response and
all species evolve normally. Do not reset pressure, density, entropy or carrier.
The one-layer vertical graph flux is measured (zero in these runs), not forced
to zero. This bypasses coupled momentum evolution and the atmospheric stepper's
time-independent entry, not reconstruction, limiting or any species budget.
External wind replacement prevents a coupled dynamical energy-closure claim.

## Measurements and records

For each water species, error e is numerical fraction minus independently
integrated exact fraction, in kg/kg. Report `L1=sum(w*abs(e))/sum(w)`,
`L2=sqrt(sum(w*e²)/sum(w))`, `Linf=max(abs(e))`. Area weights are A_cell/2 at each
of the two CP nodes; carrier weights are their actual mass in kg. Both interfaces
and per-interface errors are reported. No relative-error denominator hides small
ice scales. Uniform-carrier error is also reported as max |ps−100000| Pa.

Initial, T/2 and T fields accompany errors, physical dry/water inventories,
composition extrema, peak loss (reference peak minus numerical peak), variance,
and mean q(1−q). The displacement diagnostic is the angle between area-weighted
excess-vapor centroids, not a fitted peak trajectory; diffusion and two-hill
ambiguity limit its interpretation. The JSON reports constant-composition error
for every case, but only the constant-profile control tests that property.

Every actual Euler stage records time/budget, mass/species inventories, exact
ledger installation, finiteness/positivity, beta range and count, carrier/species
incidence residuals, flux-sum residual, donor substeps, and actual outgoing
integrated carrier divided by the smaller physical endpoint mass. The production
donor routine's own count is recorded; a zero count means its safe-high-order
shortcut was used, not zero transport. Production inventories are cross-checked
against an independent half-layer mass calculation. Scalar quantities in kg can
have roundoff residuals of tens of kg against ~1e18 kg global inventories.

An independent NumPy histogram computes incidence; tests compare it against the
original node-by-node all-face oracle. That oracle is also retained at every
stage for <=16-cell cases, never scanned per node on larger grids. No production
kernel is optimized. Raw Euler inputs/outputs/ledgers are archived for the first,
half-period and last step; scalar audits are flushed after every successful
stage. The last step input and failure manifest permit replay of failures.

Each protected run directory contains frozen-baseline and interface identifiers,
per-file runtime hashes, exact driver/helper archives, installed environment,
geometry and prognostic input archives, reference fields/quadrature estimates,
stage JSONL, snapshots, runtime and artifact-size measurements. Existing output
is never overwritten. Errors become failed manifests and a collection failures
file; later cases continue, and the command exits unsuccessfully if any failed.
No scientific accuracy threshold converts a poor result into a hidden failure.

## Runnable development commands

Install a wheel from #16 or this dependent branch; retain full repository history
for byte-level source checks. Use the test extra. From outside the checkout:

```sh
REPO=/absolute/path/to/vfs-a
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" \
  --output c5-smooth --cells 16 64 256 --steps 120 240
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" \
  --output c5-constant --cells 16 --steps 120 --profile constant
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" \
  --output c5-empty --cells 64 --steps 240 --profile empty_condensate --flows NL2010_case4
```

The first is 18 cases, followed by three constant controls and one compact-cloud
case. dt is T/120=8640 s or T/240=4320 s; only the scalar transport is isolated,
so these are not stable coupled atmospheric timesteps. CPU float64 only. Read
results before extending the campaign; constant-composition behavior needs a
separate scientific/numerical decision, not an unreviewed change in this PR.
