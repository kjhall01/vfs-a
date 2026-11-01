# Actual-carrier timestep restrictions and passive-content bounds

Dependent on the [rotating-balance diagnosis](ROTATING_BALANCE.md), draft #28.
Production baseline remains `e959b7005832a5041e082b6e300838e15bfbaa9e`.
Driver/test commit: `2fa55e4`. No runtime, default, fixture or tolerance change.
This is a small global one-layer stress probe, not a physical benchmark or a
new transport method. The same argument applies per layer when there is no
interlayer exchange; coastal masks and optional reference processes are outside
this probe. Positive/negative flux parts below are an incidence decomposition,
not clipping of prognostic variables.

## Euler condition and units

For cell i, volume `V_i=A_i h_i` (m³) and signed face volume rate
`Q_e=dv_e I(h)_e u_e` (m³/s), let O_i and N_i be outgoing and incoming rates.
The actual centered carrier flux, not a donor h or a guessed speed CFL, gives

```text
V_i^+ = V_i - dt O_i + dt N_i
(V q)_i^+ = (V_i - dt O_i) q_i + dt sum_in Q_ji q_j.
```

Thus `dt O_i <= V_i` makes the donor and incoming coefficients nonnegative.
If `V_i^+>0`, the new concentration is a volume-weighted convex combination of
old concentrations. It preserves any interval `[q_min,q_max]`, including
**signed Celsius temperature**, provided all initial concentrations lie there.
This is simultaneously the hT, hS and hq condition because all use the same F.
It establishes neither wave stability nor a general unconditional bound.
Strict h positivity without inflow needs `dt O_i < V_i`. The implementation
also requires `h > 1e-3 m`; a sufficient no-inflow margin is instead
`dt O_i < A_i(h_i-1e-3 m)`. No prognostic floor is inserted.

For centered h, outgoing transport can be dominated by a much thicker neighbor.
Consequently an `|u| dt/dx` estimate can be small while `dt O_i/V_i` exceeds one.
The cell with zero initial velocity has no finite **transport** bound, but
pressure acceleration can still impose a wave/stage restriction. No stable
wave-CFL constant is established by this experiment.

The existing SSPRK3 uses three dt Euler substeps and convex combinations of
**contents**, not concentrations. If every Euler substep satisfies the condition
and maintains positive h, the convex combinations also preserve concentration
bounds (with weights proportional to h). An initial-state check alone does not
establish that condition at later stages. The observer delegates all RK formulas
to production `ssprk3`; it never adjusts states or copies the RK algorithm.
Final-state bounds can hide inadmissible intermediate Euler states.

## Manufactured counterexamples and results

Use the archived 32-cell ocean fixture mesh, one layer, omega=0, fixed bottom
1000 m. Cell 0 has h=0.1 m and T=-2 degC, S=35, q=1. Other cells have h=1000 m,
T=10 degC, S=0, q=0. One edge exports at 1 m/s; a second optionally imports the
same volume rate. These sharp fields are intentionally not in pressure balance.
They test an admissibility claim for valid input, not a useful ocean simulation.

Initial carrier bound is **862.483090083 s**. The Euler tests use .1/.5/.9/1.1
of that bound. At .9, thickness and all initial concentration intervals are
preserved. At 1.1:

| Configuration | Public result | Thickness | Salinity/tracer/temperature |
|---|---|---|---|
| Outflow only, Euler | rejects | raw Euler minimum -0.01 m | no repaired state returned |
| Outflow only, SSPRK3 | rejects at subsequent RHS validation | first Euler minimum -0.01 m | input remains unchanged |
| Equal inflow/outflow, Euler | **accepted** | minimum 0.1 m | S=-3.5, q=-0.1, T maximum 11.2 degC |
| Equal inflow/outflow, SSPRK3 | accepted | final minimum .657356 m | final bounds hold, but first two Euler stages have S minima -3.5 and -.287723 |

The throughflow Euler overshoot occurs with a thin-cell outgoing speed Courant
number only **0.000219978**. Its actual carrier outflow fraction is 1.1. Total
volume and conservative contents still cancel in the incidence sums. Conservation
and positive h therefore do not imply salinity admissibility or tracer bounds.
Negative Celsius T itself is **not** a failure; the temperature failure is
leaving its initial interval [-2,10] in this unforced donor problem.

[Raw small-case results](admissibility_results/results.json) retain every stage,
minimum h, concentration range, actual outflow fraction, graph-divergence
residual, inventory rate and absolute local rate scale. Content rates have
units m³/s, m³ degC/s, m³ salinity-unit/s and m³/s respectively. Finite-step
mechanical energy changes are also recorded; they are not transport bounds or
instantaneous closure claims. Failures are kept, not filtered out of the result.

## Validation and reproduction

Installed unchanged wheel, copied tests/tools/examples outside checkout:
**627 passed, one CUDA skip** locally (macOS ARM64, CPU float64). Eight focused
tests cover the sufficient condition, explicit violating cases, conservation,
no input mutation/repair, stage recording, and exact delegation including
gradients for all five prognostic tensors. No accelerator, reduced-precision or
long-backward claim. The parent milestone's atmospheric portable and ocean
reference comparisons remain applicable because runtime is identical; the
full suite reruns them. No reference file is changed.

```sh
python -I tools/investigate_ocean_admissibility.py --repo /path/to/repo \
  --output /new/nonexistent/admissibility-run
python -I -m pytest tests/ocean/test_admissibility_investigation.py -q
python -I -m pytest tests --fixture-mode=portable -q
```

The exact runtime guard stays enabled. Original fixture geometry and fields are
unchanged; only the documented manufactured fields are formed in memory. Private
manifests retain complete source/input/driver hashes. The public subset excludes
detailed machine metadata and any artifact over 128 KiB, with an omission list.

## Proposed next implementation boundary (not implemented)

A small explicit **stage-admissibility check** for the constant-density candidate
could reject a dt when any actual Euler carrier outflow exceeds its available
volume (including the configured minimum-h margin). It should reuse the existing
RK sequence, explain the rejected cell/layer and bound, and preserve accepted
step arithmetic/gradients. It must not silently shrink dt, clip state, swap fluxes,
claim wave stability or alter the archived default. Variable-stage h/velocity
means a single initial timestep estimate is insufficient. Review this acceptance
policy separately from changes to spatial rotation or active forces.

Before active stratification, derive a hydrostatic Boussinesq potential energy
and its derivatives with respect to **h, hT, hS**, at fixed stored contents.
Then pair the active scalar work with the same content flux traces. Attaching
the archived baroclinic term to the new mechanical energy is not justified.
Start with a linear EOS, fixed bathymetry, no remap/mixing/forcing, and test the
constant-density limit and flat/topographic rest before internal waves or
baroclinic flow. The rotating reconstruction limitation remains independent and
must not be hidden by a new thermodynamic energy budget.
