# Semi-implicit ocean mixing research

`LaggedClosure` predicts transport coefficients, then holds them fixed during a
budgeted mixing update. `SubcycledLaggedClosure` composes a prescribed number of
these updates inside one outer forcing/dynamics timestep. This is a different
time discretization from the fully coupled research reference; no claim of
end-coefficient fixed-point convergence is made. Supported runtime/defaults,
JMD95, energy functional, physical sources and fixtures are unchanged.

## Discrete steps

One positive source/sink predictor estimates k for the coefficients. It is an
uncommitted predictor, not a physical energy increment. Native edge momentum
is then diffused once at those fixed coefficients. Its physical shear loss and
backward-Euler time-discretization loss are credited as TKE production P using
the native interpolation transpose and metric. Shared-edge column exchanges
are retained exactly as in the reference formulation.

With M the layer mass, transport conductance G and closed-boundary divergence D,
the default ordered split is:

    M*(k_transport-k0) = P + D(dt*G*gradient(k_transport))
    (M + d)*k_final = M*k_transport - B
    heat = d*k_final

Here d=dt*M*Ceps*sqrt(k_predict)/L is the linearized dissipation coefficient.
B is the actual conservative-path JMD95 buoyancy work of the proposed scalar
exchange, with the existing midpoint energy-flux allocation. Summing the steps
closes the local TKE ledger. Dissipated energy is deposited using the actual-H
thermal inversion; final full-H and entropy guards remain unchanged.

Heat, salt and passive scalar diffusion share a tridiagonal factorization and
multiple right-hand-side solve. Momentum is cached during scalar funding retries;
TKE transport factors and transported availability are reused. The banded
factorization is batched across columns and uses no dense solve or Python loop
over columns. Thermodynamics still needs further optimization.

If a layer cannot pay B, the conductances on its adjacent scalar interfaces are
reduced using the ratio of available TKE to actual positive work, with a numerical
safety factor (default .9). Scalars are re-solved implicitly, preserving their
conservative flux form and positivity structure. At most eight such reductions
are attempted. Any remaining unfunded columns explicitly defer scalar mixing
for that substep; momentum, TKE transport and dissipation still proceed. Every
reduction and deferral is reported. No negative TKE is clipped or floored and no
unfunded buoyancy work is concealed as heat.

This limiter changes the effective mixing. Successful conservation alone is not
accuracy evidence: its activity, safety-factor sensitivity and timestep behavior
are part of the qualification. The alternative `funding='column'` is a control
that funds B through a simultaneous frozen linear TKE solve; `funding='local'`
retains the initial adjacent-interface backoff control. Neither is the default.
`CoupledReport.fixed_point_error=0` is a compatibility placeholder here, not an
assertion of fully implicit nonlinear convergence.

## Reproduction

Run outside the checkout against an installed wheel:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/lagged_matrix.py --output results
python validation/ocean_column/analyze_lagged.py --input results --output summary.json
python validation/ocean_column/lagged_mesh.py --output mesh.json
python validation/ocean_column/lagged_mesh.py --seed .0001 --speed .004 --duration 300 --output mesh-weak.json
python validation/ocean_column/lagged_mesh.py --cells 162 --methods coupled2 lagged8 --seed .0001 --speed .004 --duration 300 --output mesh-match.json
python -I -m pytest tests --fixture-mode=portable -q
```

The spherical tests use 42 and 162 cells, eight unequal-thickness layers and
native shared-edge velocities. They advance vertical mixing only; they are not
global horizontal-dycore simulations. Each 120- or 300-second comparison uses
16 coupled substeps as its numerical reference. The one-hour forced-column
comparison uses coupled dt18.75 as its numerical reference; neither is an analytic
solution or proof of asymptotic convergence.

## Column results

The default ordered split completes the timestep, substep, safety-factor, cooling,
rain/heating and 32-layer controls. No column-wide scalar deferral occurs in
those controls, including the 12-hour run. Interface reductions remain active:
they are not being hidden by the completion status.

One-hour errors against the coupled dt18.75 numerical reference:

| Outer dt / mixing substeps | Full trials | Velocity relative L2 | TKE relative L2 | Temperature RMS (C) |
|---|---:|---:|---:|---:|
| Coupled dt300 | 256 | 0.010067 | 0.40339 | 0.00029213 |
| Semi-implicit dt300 / 1 | 36 | 0.009881 | 0.45406 | 0.00022325 |
| Semi-implicit dt300 / 4 | 143 | 0.006245 | 0.21936 | 0.00010273 |
| Semi-implicit dt300 / 16 | 570 | 0.004874 | 0.18543 | 0.00004774 |
| Semi-implicit dt150 / 1 | 72 | 0.005249 | 0.20625 | 0.00016621 |
| Semi-implicit dt75 / 1 | 143 | 0.002364 | 0.08850 | 0.00009070 |
| Semi-implicit dt18.75 / 1 | 570 | 7.24e-7 | 0.000948 | 0.00000541 |

The coupled fine reference uses 2363 trials. At equal outer dt18.75, the two
methods differ by 0.095% in TKE relative L2. This is agreement between numerical
methods, not an analytical validation. Mixing subcycles at outer dt300 do not
reproduce a dt18.75 forced trajectory: wind forcing and rotation still occur on
the coarse outer timestep. That splitting error remains visible in the table.

Changing the limiter safety factor from .9 to .5 changes TKE by 5.31% relative
L2 at dt300, but by 0.0522% at dt18.75. Corresponding temperature RMS differences
are 3.92e-4 C and 8.00e-7 C. Numerical limiter sensitivity diminishes in these
controls but is not eliminated or qualified for arbitrary states.

The retained alternatives show why the funding formulation matters. The initial
local-backoff control takes 105 trials and defers scalar mixing in 8 of 12 steps;
its TKE error against the fine reference is 143%. The simultaneous column-funding
control avoids deferrals but has 153% TKE error. Their evidence remains included;
neither is recommended as the default research candidate.

## Twelve hours

With outer dt300 and four mixing substeps, all 144 outer steps complete, using
1727 full trials over 576 mixing substeps (about three trials per substep).
There are 2868 limited interface/substep events and zero column-wide deferrals.
The smallest scalar coefficient fraction is 6.94e-5; this occurs in a funding-
limited interface and is an important limitation, not a claim of unmodified
mixing everywhere.

Final energy residual is -1.20e-10 J/kg; salt and passive relative inventory
changes are 1.04e-15 and 5.33e-16. The coupled dt300 reference used 8381 trials.
Recorded wall time is 316.39 s versus the previously recorded 869.65 s, but jobs
overlapped differently: these wall times are not an isolated speed benchmark.
The trial-count reduction is 4.85-fold.

At 12 hours the semi-implicit and coupled dt300 endpoints differ by 0.543% in
velocity relative L2, 0.984% in TKE relative L2 and 0.001289 C temperature RMS.
There is no fine-step twelve-hour reference here. No 48-hour or climate-run
qualification follows from these results.

## Native mesh cost and accuracy

All tested 42/162-cell, eight-layer mesh cases retain nonnegative TKE/tracers and
close total energy to approximately 1.6e-11 J/kg or better. Both energetic
(seed k=.001, edge speed scale .2 m/s) and weaker-turbulence (k=.0001,
.004 m/s) initial conditions are included. No column-wide deferral occurs.

The most informative cost/accuracy comparison is the 162-cell weak case over
300 seconds, using 16 coupled substeps as the numerical reference:

| Method inside the 300-s update | CPU seconds | Trials | Velocity relative L2 | TKE relative L2 | Temperature RMS (C) |
|---|---:|---:|---:|---:|---:|
| Two coupled substeps | 8.474 | 53 | 4.97e-5 | 0.14789 | 1.36e-4 |
| Eight semi-implicit substeps | 5.385 | 20 | 3.13e-5 | 0.02716 | 2.92e-5 |

In this comparison the semi-implicit update is closer in all three reported
fields and uses 36% less process CPU time (1.57-fold speed ratio). This is a
single measured mesh/input comparison, not a universal throughput claim. CPU and
wall times are both retained; other jobs ran concurrently. The same studies also
show tradeoffs: four semi-implicit substeps improve TKE/temperature error versus
one coupled step in the weak case but have larger velocity error. All methods,
not just the favorable pair, are retained in the evidence.

These tests exercise native shared-edge momentum and cell energy allocation.
They do not run horizontal transport, pressure gradients, rotation, ALE remap,
or a distributed global forecast. Larger batches, realistic vertical resolution,
coasts/dry columns, GPU performance and backward propagation remain unqualified.
The current retry loop recomputes scalar thermodynamics across the whole batch;
active-column work selection and compiled banded kernels remain optimization
opportunities. Fixed iteration bounds are useful but do not establish global
model throughput.

## Regression and recommendation

Installed unchanged wheel outside checkout: **1030 passed, one CUDA skip**,
one existing tensor-to-scalar warning, 250.78 s. Seven new tests include independent
dense multi-RHS comparisons, stable/unstable unequal-column native budgets,
subcycle aggregation, analytic linearized decay, exact-zero TKE and explicit
funding deferral. Corrected-Gibbs and dry-helper portable comparisons pass.

This is a promising research candidate for further integration work, with an
explicit accuracy/cost tradeoff. Keep the coupled solve as a numerical reference;
do not merge all experimental alternatives or change supported defaults based
only on this campaign. The small-mesh comparison supports the direction, while
limiter activity, coarse-step errors and global scaling remain constraints.
All uploaded artifacts are below 128 KiB. No checkpoint binaries are uploaded.
