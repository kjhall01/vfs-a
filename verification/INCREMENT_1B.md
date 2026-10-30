# Increment 1b — energy derivatives, work and actual stage budgets

This increment depends on [increment 1a / PR #13](https://github.com/kjhall01/vfs-a/pull/13).
The numerical baseline remains **8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648**.
No runtime source, defaults, fixtures, limiter or RK expression changes.

Run against the installed baseline wheel, outside the checkout:

```sh
cd /tmp
python -I /path/to/vfs-a/tools/run_energy_verification.py \
  --repo /path/to/vfs-a --output /tmp/vfs-energy-16 \
  --cells 16 --levels 3 --dt 1 --directions 2 --seed 20261030
python -I /path/to/vfs-a/tools/run_energy_verification.py \
  --repo /path/to/vfs-a --output /tmp/vfs-energy-32 \
  --cells 32 --levels 4 --dt 1 --directions 2 --seed 20261030
```

The shared manifest lifecycle verifies every installed runtime file against the
baseline, records environment/configuration and completion/failure, refuses to
overwrite runs, and hashes artifacts. It archives both executed drivers and the
environment collector. Case `.pt` files contain geometry/vertical coefficients,
initial/final tensors, actual stage states/diagnostics, work tendencies and C2
directions. `metrics.json` reports all measurements, including signed residuals;
there are no new scientific pass thresholds. Partial/failed runs remain labeled.

## Cases and physical interpretation

Each run uses Fibonacci geometry, zero Lloyd iterations, radius 6371220 m, fixed
5000 Pa lid and `direct_u_gl2_adjoint` column geometry. All tensors are CPU float64;
Torch uses one thread. Actual arrays, seeds, physical constants and configuration
are recorded. Surface geopotential is 1000*x m²/s² and ps=100000*(1+.01*x) Pa,
where x is the cell's unit Cartesian coordinate. These diagnostic states are
**not a terrain-rest construction or a balanced benchmark**.

M-B is the primary bounded SSPRK3 configuration. Smooth composition has separate
positive vapor/liquid/ice scales; sharp and reversed cases have a cloud edge and
zero ice, activating the limiter. C2 uses only smooth positive composition;
changing an absent species in both signs would be inadmissible. D-E is a dry
SSPRK3 control. D-I/M-I ARK4 measurements remain separate, pending work; there is
no inferred ARK4 or timestep-backward evidence here.

## C2-DER-001: independent derivative sweeps

At fixed geometry, independently differentiate the production scalar internal,
potential, kinetic, fixed-lid reservoir and total boundary energy with autograd.
This does not call the production analytic energy gradient. Compare total-energy
AD directional derivatives with `boundary_energy_rate` (moist) or
`boundary_energy_rate_closed` (dry), and each component's derivative with
`[E(x+epsilon*d)-E(x-epsilon*d)]/(2*epsilon)`. The analytic rate method contracts
with a direction here, so its result is J per unit epsilon, **not watts**.

Use two saved random directions per prognostic block; vapor, liquid and ice are
separate blocks. Epsilon is dimensionless, 1e-2 through 1e-7. Directions have RMS
10 m/s for edge/full wind, 1000 Pa for cell surface pressure, and 10% of the
block's mean absolute content for interface entropy/theta and each species.
Other prognostic fields remain fixed. Entropy stays fixed when varying species.

Report both signs of Taylor remainders (J), signed FD-minus-AD differences,
absolute-relative derivative errors, and analytic-minus-AD differences. Relative
error uses |AD directional derivative| with **null for zero denominator**, not a
hidden floor. Small directional derivatives and subtraction of planetary energies
can produce poor relative FD accuracy. Keep all epsilon rows; do not report only
the most favorable epsilon or treat roundoff plateaus as operator defects.
This checks discrete energy derivatives, not continuum thermodynamics or rollout
sensitivities. No saturation adjustment or moisture floor is introduced.

## C3-WORK-001: component work closure

Production `tendency_terms` supplies momentum pressure-gradient, rotation,
vertical and Bernoulli terms, scalar horizontal/vertical transport and pressure
continuity. Alias entries (`total`, `advective`, duplicate zero diffusion slots)
are not counted twice. Water has only a total centered transport diagnostic.
The observer reports the difference between assembled components and the actual
combined RHS; separate momentum solves can differ at roundoff.

For M-B, `transport.evaluate(state, dt)` supplies the actual budgeted RHS,
limiter force, entropy heat source and released face heat. Report centered base
work, bounded momentum change excluding limiter force, limiter mechanical work,
limited-minus-centered entropy/water transport, and heat allocation separately.
The inferred bounded momentum change is a subtraction diagnostic; it is not a
new force implementation. Contract every piece with independent AD gradients
for all five energy components, in W, plus the analytic total contraction.

The closure denominator is the sum of absolute component total-energy powers;
report signed total W, W/m², component assembly residuals, and heat-source work
minus allocated face heat. The fixed-lid reservoir participates explicitly.
Initial instantaneous closure is **not** the one-step energy change, which is
reported separately in J. No Held–Suarez forcing is included in these cases.

## C4-LEDGER-001: observe the actual three Euler stages

A local observation wrapper surrounds `euler_with_info` during the normal public
`model(state, dt)` call. It returns the original objects unchanged and is removed
after the call. No RK coefficients/formulas are copied. Tests require observed
and unobserved final states to be bitwise equal. Each recorded Euler budget is
exactly the outer dt; the second/third inputs are the states chosen by production
SSPRK3. The final convex-combination state is archived separately.

Archive initial/output states and **all** returned stage diagnostics, including
amounts, integrated flux, target carrier, beta and donor substeps. Species order
is dry/vapor/liquid/ice. Amounts/flux integrals are Pa m² in production; dividing
by gravity gives kg. Wind lives on edge/full levels; pressure on cells; entropy
and water contents on cell/interfaces (water has a final three-species axis).

Independently compute carrier from adjacent half-layer pressure thicknesses and
water from prognostic content times area/g; physical dry is their difference.
For every node, independently sum incoming minus outgoing recorded transfers
using endpoint masks, without the production `index_add` divergence. Compare
carrier and species endpoint balances, species-flux sum vs carrier, ledger sum
vs target, physical dry vs ledger dry, and production inventory vs independent
sums. Report absolute residuals and per-species scales/normalized residuals;
zero ice has a null normalized denominator. This different reduction order is
not expected to reproduce the authoritative ledger bitwise.

Keep the **exact** check that Euler water equals returned ledger water divided
by area. Do not reconstruct it from RHS. Also check finite recorded data, limiter
count/mask agreement, beta in [0,1], three stages and their budget dt. Report raw
min/max fractions, positive-carrier minimum and each species' floor margin;
never clip negative roundoff or invent a scientific positivity tolerance.
Global species sums and stage-final bounds are measured independently of the
exact storage check. The final convex-combination inventory is reported too.
These are one-step observations, not completed C10 integrated-budget evidence.

## Status and next work

The small-case drivers and corruption/independence tests are implemented.
[Results](INCREMENT_1B_RESULTS.md) distinguish local and hosted execution.
Remaining 1b campaign work includes broader state/direction/resolution samples,
reference-quality figures, uncertainty assessment and case-specific physical
interpretation. A manifest or successful execution alone is not a paper claim.

**Next queued:** prescribed horizontal, vertical and combined transport protocols
and drivers (C5), followed by physically consistent flat/terrain-rest construction
(C6). C10 integrated budgets, C11 Held–Suarez statistics and C12 restart equivalence
remain pending; model state_dict alone is not a complete restart format.
