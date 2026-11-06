# Ocean-owned bounded transport: first implementation pass

Base: main `5841cda8b4585418b18f0ba0486315870b1a59b2` (merged JMD95 #31).
Independent of the unmerged ALE #32. This is horizontal layer transport, not a
vertical remap or a completed bounded, energy-closing active ocean.

## What is implemented

`vfs/ocean/transport.py` provides `HorizontalTransport` and an inspectable
`TransportLedger`. The donor reference and FCT (flux-corrected transport)
candidate both evolve volume and stored T/S/passive contents conservatively.
The installed module imports only Python and torch, never atmosphere or tools.
All existing runtime files, defaults, fixtures and tolerances are unchanged.

The design explicitly reuses the atmospheric work at the base commit:

| Atmospheric implementation | Ocean adaptation |
| --- | --- |
| `bounded_transport.py`: one carrier graph | One graph of cell/layer volumes and horizontal edge/layer fluxes |
| Positive donor species amounts | Remaining-volume plus incoming-donor content form |
| All-incident-face correction budgets | Local concentration min/max capacities, summed over every incident face |
| Common species limiter, species sum to total mass | Independent field limiters: potential T and practical S do not sum to one |
| Authoritative Euler amounts | Install the ledger amounts directly, not old contents plus a diagnosed RHS |
| `bounded_moist.py`: three full Euler budgets | Research SSPRK3 checks all three full-dt Euler endpoints before convex combinations |
| Gibbs/entropy heat release and carrier-transpose work split | Not copied: the ocean JMD95 dynamic-enthalpy functional does not supply that caloric/entropy closure |

Ocean volume is the Boussinesq carrier. Its conservation also conserves reference
mass `rho0*volume`; this does not claim conservation of the integral of variable
JMD95 density. T/S inventories mean integrals of `h*T` and `h*S`. In particular,
potential-temperature content is not a newly defined calorimetric heat budget.
Negative Celsius is allowed. Initially nonnegative passive fields are bounded
below by zero; signed passive fields are also supported without an artificial
floor. Negative salinity inputs are rejected.

## Carrier and limiter

For cell/layer node i, volume V_i=A_i h_i and concentration q_i=Q_i/V_i.
The oriented face carrier is F_e=dv_e I(h)_e u_e, with units m3/s. Let G=dt F.
Before any update, require

```
out_i = sum of positive outgoing G over every incident face
out_i <= cfl_safety * (V_i - A_i*h_min),    0 < cfl_safety < 1
```

The default safety is 0.9 and h_min is 1e-3 m. `timestep_limit(state)` reports
this carrier-only bound; it is not a surface/internal gravity-wave stability
bound. Net incoming/outgoing cancellation is not a substitute for the bound.
The implementation rejects excessive outflow; no subcycling or automatic retry
is hidden inside a step.

The donor endpoint uses the nonnegative remaining carrier plus incoming donor
amounts. The centered high flux is G*(q_i+q_j)/2. Let A_e be high minus donor
integrated scalar flux. For every field, the local bounds include the node and
all its active incident neighbors. Their products with **new** volume bound the
new amounts. Sum positive and negative antidiffusive demands over all faces,
then limit each correction by both endpoint capacities. The result is

```
Q_new = Q_donor + incoming_minus_outgoing(beta * A)
```

Pairwise flux cancellation conserves every stored inventory to roundoff. The
donor convex combination and endpoint budgets give the local bound in exact
arithmetic. A 64-epsilon contraction of beta only when it is limited provides
roundoff headroom; the state is never clipped. Nonfinite results, a violated
bound, or negativity from nonnegative local inputs are rejected. Closed coasts
are excluded from both transfers and neighbor bounds, so dry dummy composition
cannot set a wet-cell limiter. Geometry and state must already match dtype/device.

Minimal transport-only use:

```python
from vfs.ocean.transport import HorizontalTransport

transport = HorizontalTransport(mesh, domain=domain, scheme='fct')
dt_carrier = transport.timestep_limit(state)  # also choose a wave-stable dt
ledger = transport.advance(state, dt)
transported = ledger.install(state)  # velocity retained; not a full dynamics step
```

The ledger exposes integrated donor/high/limited fluxes, limiter coefficients,
amount bounds, carrier, outflow fractions and trace changes. It retains tensor
graphs. Branching at extrema, velocity reversal and limiter activation makes
this piecewise differentiable, not globally smooth. No forward AD is used.

## Active energy accounting and the zero-velocity obstruction

`tools/probe_ocean_bounded_transport.py` composes ledger contents with the
**existing centered JMD95 momentum tendency**, in research Euler and SSPRK3
maps. It validates each endpoint. This composition is deliberately not installed
as a runtime dynamics class or selected as an Ocean default.

Write C_T=(dH/d(hT))/A and C_S=(dH/d(hS))/A. Replacing centered scalar transfers
while leaving momentum unchanged creates the spatial work residual

```
R = sum_faces [ (C_T,j-C_T,i)*(J_T,limited-J_T,centered)
             + (C_S,j-C_S,i)*(J_S,limited-J_S,centered) ] / dt
```

The driver measures R independently against the full analytic derivative/RHS
contraction. Tests additionally use autograd of the energy as an independent
derivative. Finite-Euler and SSPRK3 energy changes are reported separately:
neither is equal by definition to dt*R, and time discretization contributes.
The measured R is negative in these cases; no general sign theorem is claimed.

One can formally cancel R by replacing the pressure's scalar traces with the
limited transport traces. The driver diagnoses this all-mechanical correction
without applying it. As velocity changes sign, a donor trace can change between
neighbor values even as the scalar flux tends to zero. Thus the corresponding
force can jump by a finite amount. Avoiding an explicit division by u does not
remove that discontinuity. At zero speed the energy constraint alone does not
determine a continuous force.

The chosen experimental composition preserves the existing force continuity,
reports the unclosed work, and adds no fictitious thermal reservoir. A physical
thermodynamic treatment of limiter work is still required before claiming a
bounded energy-closing active timestepper.

## Measured evidence

CPU float64, torch 2.15.0+cpu. The compact [JSON report](bounded-transport-results.json)
contains installed-runtime and driver SHA256 hashes, all ring resolutions,
active cases, and zero-velocity sweeps. There are no scientific pass cutoffs.

Prescribed periodic ring, one-quarter period, Courant approximately 0.2, SSPRK3:

| Cells | Smooth donor L1 | Smooth FCT L1 | Sharp donor L1 | Sharp FCT L1 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 3.630e-2 | 4.192e-3 | 1.396e-1 | 6.306e-2 |
| 64 | 1.888e-2 | 1.160e-3 | 9.922e-2 | 3.985e-2 |
| 128 | 9.629e-3 | 3.135e-4 | 7.034e-2 | 2.514e-2 |
| 256 | 4.861e-3 | 8.359e-5 | 4.980e-2 | 1.585e-2 |

Smooth FCT L1 refinement orders are approximately 1.85, 1.89 and 1.91; limiter
effects at extrema preclude asserting universal second-order behavior. The sharp
front remains within [0,1]; its convergence is slower. Ring inventory errors
are at most 6.2e-16 in absolute mean concentration. These are transport accuracy
measurements on a prescribed ring, not spherical balanced-jet/wave qualification.

Eight spherical active tests combine donor/FCT, global/closed-coast domains,
and ordinary/zero-salinity fronts. Each advances one 10-second SSPRK3 step at
32 cells and three layers. All full Euler endpoints remain admissible; inventory
relative errors are at most 3.1e-16. The residual agrees with face work to
roundoff relative to the absolute local work. The all-mechanical diagnostic
cancels that work to roundoff but fails the velocity-continuity requirement.
For the ordinary global front, FCT limiter work is -0.00344 W/m2, versus
-0.00760 W/m2 for donor transport. The extreme zero-salinity global front gives
-0.00926 W/m2 for FCT. These are instantaneous synthetic-case values, not a
climate energy-drift estimate. The centered Euler update makes that zero front
negative (minimum S=-2.42e-6), while the bounded stages retain S>=0.

At velocities +/-1e-12 m/s, the maximum centered-momentum jump is 2.41e-16 m/s2;
the counterfactual all-mechanical jump is 2.99e-6 m/s2. The donor sweep has a
nonzero limiting jump. FCT coefficients also become roundoff-sensitive in this
extremely small-transfer regime; its diagnostic correction does not approach
zero in the sweep. The actual transported scalar rates do approach zero.

## Verification and reproduction

Build/install the wheel and copy `tests`, `tools`, `examples` outside the checkout:

```sh
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/probe_ocean_bounded_transport.py --output bounded-transport.json
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
python -I examples/ocean_jmd95.py
```

Installed-wheel portable suite: **659 passed, one CUDA skip**, including 17 new
transport tests. The reviewed legacy-ocean capture is exactly equal for all 164
tensors. Corrected atmospheric and dry portable fixture comparisons pass.
Historical exact discrepancies remain separately reported by existing CI; no
fixture or tolerance is replaced. CI runs the installed-wheel suite and a quick
measurement campaign after restoring the candidate wheel. All newly committed
files are below 128 KiB; no binary fields are uploaded.

Coverage includes local bounds, per-layer inventories, mesh orientation, coast
isolation, uniform independent T/S/tracers, negative Celsius, zero/near-zero S,
hidden excessive outflow with positive final thickness, rejection at a second
full Euler stage, independent energy work, velocity reversal, one fixed-branch
directional derivative, and no forward autograd. GPU, float32, compile, long
backward runs, long active trajectories and performance are unqualified.

## Next boundary

1. Derive a thermodynamically meaningful ocean treatment of limiter/mixing work,
   including what energy/entropy variables make that treatment possible. Keep
   independent T/S inventories and velocity continuity explicit constraints.
2. Test that proposal against these ledgers and independent work contractions
   before exposing a bounded active runtime stepper. Preserve the centered
   JMD95 model as an energy-paired control.
3. Compose separately with ALE #32, with separate horizontal, remap and time
   integration energy accounts. Then extend stratified wave and balanced-jet
   tests over many periods and resolution/dt sweeps.

This pass does not resolve the existing rotation/kinetic reconstruction issue,
provide vertical mixing or surface exchange, or implement ocean-atmosphere
coupling. It establishes the reusable bounded transport and the precise work
obstruction that the next thermodynamic derivation must address.
