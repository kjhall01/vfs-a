# Sharp-front Gibbs/entropy ocean research candidate

Parent: draft #37, `bf5ad5cb8887ec05d26e0566dde0c636daf0676a`.
The frozen candidate conserves inventories and enforces global admissibility,
but its one-circuit sharp front overshoots SP=[34,36] and a passive maximum of
one. This dependent research composition changes scalar transport. **Rotation,
mesh construction, Hamiltonian, JMD95 density and caloric reference are fixed.**
No installed runtime or default changes. The parent remains independently
reproducible; this is not a replacement of its archived results.

## What changes

`tools/ocean_front_research.py` wraps the parent through explicit functions.
It uses an ocean-owned local flux-corrected target, followed by an active
mechanical-work constraint. No atmospheric imports, forward AD-generated
forces, state clipping, salt floors or extra heat reservoir are introduced.
The caloric physics is still the explicitly GSW-derived surface potential.

Let V be the actual full-Euler carrier volume, Q_L the positive donor amount,
and A_e the integrated antidiffusive flux from donor to the Gibbs high trace.
For each stored concentration (entropy, SP, passive), calculate q_min/q_max over
that stage's cell and incident neighbors. Sum all incoming/outgoing correction
magnitudes separately, and limit them by V*q_max-Q_L and Q_L-V*q_min. Each
face uses the smaller allowance of its receiving and losing endpoints.
This is the same carrier/amount-budget idea already used in atmospheric and
ocean transport, independently implemented for these ocean variables.

Salt/passive target antidiffusion is additionally capped by the parent's
per-incidence global-admissibility coefficient. Reducing individual FCT
coefficients preserves their local-bound proof. The initial salt must still
satisfy **0<SP<42**, passives must be nonnegative, and every complete Euler
stage must remain in the nonlinear caloric domain. There is no automatic retry.

Unlike the parent, the new **target** limits entropy and salt independently;
this is needed to control the thermal as well as salinity front. It does not
simply attach entropy changes to salt changes. The actual work of both changes
is explicitly paired with momentum or nonnegative heating through the same
Hamiltonian. The parent composition-attached correction remains the fallback.

## Why strict active local bounds are not promised

At a local extremum, an available local correction budget can be zero. The
resulting upwind trace correction need not vanish as one face's velocity tends
to zero. If its scalar energy work is positive and proportional to velocity,
compensating it mechanically can produce a finite force jump at zero velocity.
Blindly replacing the parent's global budgets with local bounds would lose an
important property of #36. Passive fields do not contribute to the Hamiltonian
and do not have that conflict.

For each face/layer define:

- W_b: actual scalar correction power for the parent's admissible flux;
- W_t: correction power for the local-bound target, relative to the Gibbs flux;
- K_e = (H_u)_e u_e/2: that face/layer's diagonal kinetic energy;
- C_e = max(max(W_b,0), K_e/dt).

Set lambda=1 when W_t<=W_b. Otherwise use

```text
lambda = min(1, (C_e-W_b)/(W_t-W_b)).
active_trace_change = parent_change + lambda*(target_change-parent_change).
```

This blends the two active **flux corrections**, not updated states. Passive
fluxes use their full local-bound target independently. There is no division
by velocity and no velocity floor. Because W_b<=C_e, lambda is nonnegative.
The final positive work cannot exceed C_e in exact arithmetic. In particular,

```text
|new mechanical force correction|
    <= max(|parent mechanical force correction|, |u_e|/(2 dt)).
```

The parent already has a continuous single-edge zero limit with other faces
limited. Hence the new force also tends to zero, even if its target remains
limited. This is a numerical work constraint, not a physical mixing closure.
It bounds the correction's instantaneous demand; it is **not** an exact
finite-step energy theorem, a gravity-wave CFL, or a bound on total acceleration.
It can retain parent work larger than K_e/dt; it does not repair parent behavior.

Salt coefficients stay between the parent's global-admissibility coefficient
and the smaller target coefficient, so the per-incidence global bounds survive
face-dependent blending. **Local SP and entropy bounds can be relaxed wherever
lambda<1.** Entropy also receives heat, and local entropy bounds do not imply
temperature bounds. No universal active maximum-principle claim is made.
Passives retain the local bounds without this relaxation.

## Energy and entropy bookkeeping

For the actual final trace correction delta_q, use the exact same scalar
covector jump as the Hamiltonian-derived pressure coupling:

```text
W_e = carrier_e * jump(H_scalar / cell_area) dot delta_q_e.
heat_e = max(-W_e, 0).
```

Positive W is canceled mechanically; negative W becomes nonnegative heat,
allocated to endpoint entropy contents using their positive H_entropy
coefficients. Advective entropy fluxes telescope. The global entropy inventory
therefore has the explicit nonnegative source budget at each Euler stage and
under the SSPRK3 stage weights. Every full Euler state is checked before an
SSP combination can hide an invalid intermediate state. The limiter remains
piecewise differentiable, not globally smooth across min/max switches.

## Measurements

CPU float64; fixed source hashes are included with the raw results in
[sharp-front/](sharp-front/). Scientific results have no empirical acceptance
thresholds. Mathematical invariants and independent derivatives have tests.

The isolated prescribed-carrier ring uses exactly #37's profiles, one circuit,
CFL=0.2 and 32/64/128 cells. Momentum/heating feedback is disabled here, so this
measures the local-bound target, not its active work fallback. At 64 cells:

| Metric | Parent | Local-bound target |
| --- | ---: | ---: |
| Sharp SP minimum | 33.23520 | 34.00000010 |
| Sharp SP maximum | 36.77285 | 35.99999991 |
| Sharp SP L1 error | 0.320575 | 0.126818 |
| Sharp SP RMS error | 0.401504 | 0.280145 |
| Sharp SP total variation (initial 4) | 19.072 | 4.000000 |
| Sharp passive maximum (initial 1) | 1.278983 | 0.99999995 |
| Smooth SP RMS error | 0.00570419 | 0.00969985 |

At 32/64/128 cells the new sharp SP L1 errors are 0.201150, 0.126818,
0.0799262. RMS errors are 0.351310, 0.280145, 0.223038. The fronts are more
spread out but no longer have the parent's overshoots in this isolated test.
Smooth SP RMS errors are 0.0292852, 0.00969985, 0.00302874, giving observed
orders about 1.59 and 1.68. **This sacrifices some smooth-extremum accuracy**
relative to the approximately second-order parent; no second-order claim is
made for this candidate on the tested smooth profile.

Independent-AD energy-work tests and analytic audit cases cover fronts, tilted
layers, nearly fresh salt, upper-salt states and closed coast masks. Maximum
normalized spatial work residual in the recorded audits is 5.06e-17; maximum
relative finite-step entropy-source budget residual is 8.29e-17.

A single edge crosses zero while 218 other salt faces remain limited. At
speeds +/-1e-9 and +/-1e-12 m/s, its force is respectively -/+5e-11 and
-/+5e-14 m/s²; it is exactly zero at rest. The work constraint activates on
that edge. Tests also verify the derivative -1/(2dt) inside this active branch,
passive/active independence, uniform passive transport and no-passive states.

## Fully coupled short front trajectory

A separate experiment advances the complete active three-layer JMD95 system
for two hours, with all momentum feedback and heat allocation enabled. Both
parent and candidate use the identical fixed 32-cell, radius-100-km mesh with
two Lloyd iterations. Layers start at [500,1000,1500] m; theta=[17,10,4] plus
+/-2 degrees across y=0; SP=35+/-1 across x=0; the passive is a 0/1 step.
Initial velocity is the same small nonuniform field for both. This is a short
nonlinear integration/budget experiment, **not** a fine-mesh accuracy or
balanced-jet study. It has no exact evolving analytical reference.

| Two-hour endpoint | Parent, dt=30 s | Candidate, dt=30 s | Candidate, dt=15 s |
| --- | ---: | ---: | ---: |
| SP range | 33.742824–36.171428 | 34–36 to roundoff | 34–36 to roundoff |
| Passive maximum | 1.084369 | 1 to roundoff | 1 to roundoff |
| Temperature range (initial 2–19) | 1.684285–19.540377 | 1.952771–19.100471 | 1.952764–19.100490 |
| Energy change, J/m² | -42.74569 | -43.04880 | -5.51745 |
| Entropy change, J/K/m² | -5.19e-6 | 17139.77761 | 17135.98630 |

The candidate completes both runs without stage rejection. The work fallback
is not active in these trajectories; it is exercised separately by the
near-zero stress tests. Halving dt reduces energy drift by about 7.80, but
this two-level comparison alone is not a general temporal-order proof.
Salt/passive excursion beyond initial bounds is at accumulated roundoff scale.
**Temperature still leaves its initial range.** Neither the entropy-coordinate
local target nor nonnegative heating supplies a temperature maximum principle.
The larger entropy increase is explicitly accounted numerical mixing/heating,
not a calibrated subgrid mixing model; it must remain visible in comparisons.
No long-rollout, GPU, coupled atmosphere, ALE or full-backward qualification
follows. Rotation remains frozen, per the development decision.

## Reproduce

Install the parent wheel; copy tools/tests outside the checkout, using
Python 3.12, torch 2.15.0+cpu and the same independent gsw 3.6.23 reference.

```sh
python -I tools/probe_ocean_front.py --output front-measurements.json
python -I tools/probe_ocean_front.py --trajectory --output active-front.json
python -I tools/probe_ocean_front.py --trajectory --scheme parent --output parent-active-front.json
python -I tools/probe_ocean_front.py --trajectory --dt 15 --output active-front-halfdt.json
python -I -m pytest tests/ocean/test_front_candidate.py --fixture-mode=portable -q
```

CI runs a smaller transport sweep and a two-step active trajectory. Field arrays,
full logs and wheels are omitted; all committed artifacts are below 128 KiB.

## Validation

Installed-wheel tests run outside the checkout: **736 passed, one CUDA skip**.
All 164 legacy ocean integration tensors are exactly unchanged. Atmosphere portable
comparisons pass for 142 moist and 93 dry tensors. Fourteen new focused tests exercise
independent Hamiltonian derivatives/work, entropy accounting, local passive bounds,
global salinity admissibility and zero crossings with other faces still limited.
Historical exact archive differences remain separate; no fixtures or tolerances changed.
