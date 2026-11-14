# Temperature-content coordinate prototype

This research wrapper advances `OceanState.temp_h = h*theta` instead of entropy
content. It uses the same JMD95 EOS, quadrature, surface caloric reference and
entropy model's transport/force pairing. No supported defaults change. It does
not revert to the earlier centered JMD95-only model or drop caloric energy.

```python
from model import TemperatureOcean  # this directory on sys.path
candidate = TemperatureOcean(existing_stratified_ocean)
physical = existing_stratified_ocean.check_state(entropy_state)  # initial conversion only
physical = candidate.step(physical, dt=60.)
```

Run from the repository with VFS installed:

```sh
python validation/temperature_coordinate/run.py --output result.json
python validation/temperature_coordinate/run.py --compile --output compiled.json
python validation/temperature_coordinate/run.py --front --output front.json
python -m pytest tests/ocean/test_temperature_coordinate.py -q
```

## Coordinate transformation

Let Q=h*T, C=h*S and E=h*eta(T,S). Differentiation gives

    dE = eta_T dQ + eta_S dC + (eta - T eta_T - S eta_S) dh

Therefore the transformed temperature-content tendency is

    Qdot = [Edot - eta_S Cdot + (T eta_T + S eta_S - eta) hdot] / eta_T.

The existing RHS evaluates Edot, Cdot, hdot and udot at the current T,S without
any inverse: E is computed by a forward Gibbs evaluation, and the physical state
is supplied explicitly. This temporary entropy view does not store or integrate
entropy as the prognostic coordinate. It intentionally reuses validated flux,
limiter and work accounting; a later direct implementation could fuse that work.

The energy remains exactly the same function of the physical state:

    H = H_JMD95(u,h,Q,C) + sum(rho0*A*h*h0(T,S)).

Its additional caloric derivatives in temperature coordinates are

    H_Q += rho0*A*h0_T
    H_C += rho0*A*h0_S
    H_h += rho0*A*(h0 - T*h0_T - S*h0_S).

The inverse chain-rule transformation of the tendencies preserves grad(H) dot
RHS algebraically. This is inherited spatial work closure, not a new Poisson or
Jacobi theorem. Analytic derivatives are independently checked against autograd,
and a JVP independently verifies the coordinate map on the actual limited RHS.
No inverse solve is used in the forward dynamics step.

Q is NOT a conserved passive content: it contains the terms needed to represent
entropy transport and irreversible heating in temperature coordinates. Replacing
it with conservative donor transport of T would change the model. Salt, layer
volume and passive tracers retain their authoritative bounded Euler updates.
Temperature domain checks still reject stages; no temperature clipping is added.

## Scope and limitations

SSPRK3 in temperature coordinates differs from SSPRK3 in entropy coordinates.
Exact finite-timestep energy or entropy conservation is not claimed. Positivity
checks on short cases are not global stability or monotonic-temperature proofs.
The short smooth campaign may be roundoff dominated; the stronger sharp-front
case is retained to expose finite-coordinate differences. Three repeated timing
samples after warmup are exploratory CPU results, not a scaling campaign.

Temperature-native layer heating and research TKE mixing are now available in
`physics.py`; see [their campaign](../../verification/ocean/temperature-mixing/README.md).
The optional [complete research composition](COMPOSITION.md) adds surface
forcing and energy-corrected ALE, with a small composed backward smoke test. Heating still needs a nonlinear finite-energy solve; mixing in
reference enthalpy still needs enthalpy-to-T inversion. The coordinate change
alone cannot remove those solves. These helpers do not yet constitute a
supported full-ocean timestep.
The benchmark's `--compile` applies the same existing thermodynamic compilation
to both coordinate formulations; no extra compiler optimization is credited.

## Recorded first-pass results

CPU float64, one thread, 128 cells × 16 layers, order-4 quadrature. Compiled
warm-start entropy dynamics: median 0.11771 s; compiled temperature dynamics:
0.08033 s (1.47×), three samples after warmup. Temperature inverse calls per
step: 6 -> 0. This is dynamics only and is additional to kernel compilation.
Uncompiled smooth samples were noisy (0.208–0.464 s for entropy); raw samples
are retained, not used for the headline gain. Timing jobs were sequential with
no intentionally overlapping test jobs in these final recorded campaigns.

The smooth 120-second comparison is roundoff dominated. A synthetic stronger
front (4/18°C, salinity 36/34, original velocities ×10000) gives:

| dt, s | Temperature-coordinate energy drift, J/kg | Entropy-coordinate drift, J/kg | Between-coordinate temperature RMS, °C |
|---|---:|---:|---:|
| 60 | -2.76e-8 | -4.52e-8 | 1.21e-10 |
| 30 | -3.86e-9 | -7.58e-9 | 3.05e-11 |
| 15 | -5.96e-10 | -1.52e-9 | 7.61e-12 |

This shows decreasing short-run discrepancies; it is not a measured asymptotic
order for the nonlinear limited scheme. Both use the same energy including the
surface caloric term. Energy differences subtract large totals and the smooth
case is at the roundoff floor. No analytic wave/jet or global accuracy reference
is provided by these coordinate comparisons.

Four new tests cover independent energy derivatives and RHS work (open/coastal),
coordinate JVP, unchanged velocity tendency, an inverse-disabled step, conservative
inventories, sharp-front backward smoke and domain rejection. Together with the
ten existing quadrature tests, 14 focused checks pass. This is not a rerun of the
full suite or full-model differentiability qualification. Source and compact raw
results are retained under verification/ocean/temperature-coordinate.
