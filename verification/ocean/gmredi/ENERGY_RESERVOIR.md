# Shared-Hamiltonian mesoscale reservoir qualification

This extends the initial GM–Redi prototype at commit
`b76c0b4266b6384e0b83bf5eafe04db3e9f8539d` on `feat/ocean-gmredi` (PR #82).
Equations, usage and report definitions are in
[GMREDI.md](../../../validation/temperature_coordinate/GMREDI.md).

## What changed

`EddyState.eddy_h` is an explicit transported cell-layer energy content, separate
from turbulent TKE. `RoquetHamiltonian` assembles the existing Roquet mechanical,
thermodynamic and TKE terms plus mesoscale energy. `model.total_energy` uses that
functional; its exact finite work and midpoint conjugates fund both GM and Redi.
There is no surrogate buoyancy energy in the exchange calculation.

Each stage constructs a conservative energy flux using the scalar transfers and
Hamiltonian midpoint conjugates. Remaining local work exchanges with the local
reservoir. A negative final reservoir rejects the step without input mutation or
borrowing from remote cells. The flux definition fixes a spatial production
allocation; global energy conservation alone does not uniquely determine it.

Redi now rebuilds slopes after GM. This removes the previous finite-step lagging
error in the salinity-only density front. Redi's quadratic variance property
still does not imply energetic neutrality: cold and warm examples below require
opposite energy exchanges. Negative rotated-diffusion updates remain a separate
explicit failure, not an energy-accounting repair.

## Tests and whole-model composition

**22 focused tests pass.** These include the original geometry/transport checks
and independent full-H derivatives with respect to velocity, thickness, CT/SA
contents, TKE and mesoscale energy, both open-ocean and coastal. The eddy-energy
conjugate equals rho0*A; other derivatives match resolved dynamics at fixed
independent energy contents. The finite-work identity is checked against an
independent autograd midpoint derivative. Forward fluxes do not call autograd.

Additional tests exercise both signs of Redi work with the published default
Roquet coefficients; insufficient local funding; energy elsewhere not paying a
local deficit; post-GM slope refresh; no-passive state layouts; and one-step
reverse-mode sensitivity against centered finite differences.

A complete forced case includes resolved dynamics, surface stress, shortwave,
heat/freshwater exchange, vertical mixing, GM–Redi and ALE. Mesoscale content is
conserved by transport/remap separately from SGS conversion. The total budget
includes the existing declared external fluxes. The pre-existing RK/physics/ALE
energy tolerance remains 1e-7 J/kg; its error is not absorbed into the reservoir.
This is not a claim of exact finite-step Hamiltonian preservation by SSPRK3.

Final installed-wheel **outside-checkout** portable suite: **1180 passed,
2 skipped**, 203.31 s (`reservoir-installed-wheel-tests.log`). Corrected-Gibbs
baseline: **142 tensors passed** (`reservoir-baseline.json`); dry helpers:
**93 tensors passed** (`reservoir-dry-helpers.json`). Production sources and the
installed wheel are unchanged from the preceding qualification, SHA256
`555f50ff1b8e42ccd41a63bf7cc3dffa417fafc1fadd985222de699e01ad18e8`.
Research modules/tests were copied outside checkout and use that installed VFS.

## Finite-duration energy and refinement experiments

All use a 32-cell, eight-layer, 100 km-radius sphere, 1000 m depth, CPU float64,
PyTorch 2.15.0+cpu, one thread, initial e=0.1 J/kg and prescribed coefficients
1000 m²/s. These parameters accelerate an isolated transport experiment, not
an Earth climate simulation. Dynamics/forcing are absent in these front runs.

The density-front experiment runs two days at dt=3600,1800,900,450 s. Constant
CT and a stable SA profile with lateral variation isolate GM flattening and
Redi tracer smoothing (`reservoir-front32.json`). At dt=3600 s:

| Stage / run | Resolved+TKE change (J/kg) | Reservoir change (J/kg) |
|---|---:|---:|
| GM alone | −5.5863564e−4 | +5.5863564e−4 |
| Redi alone | 0 | 3.70e−18 |
| GM + Redi | −5.5863564e−4 | +5.5863564e−4 |

Pure Redi leaves the density front unchanged. With refreshed slopes, combined
Redi also leaves the post-GM salinity profile unchanged. The energy release now
remains in the full state rather than disappearing. Across all density-front
runs, endpoint finite-H residuals are at most **1.56e−17 J/kg**. The minimum
reservoir is 0.073915 J/kg; no positivity/funding repair was used.

Cold and warm experiments run six hours at dt=600,300,150,75 s. CT/SA now both
vary, using the unmodified published EOS coefficients. At dt=600 s:

| Redi-only case | Resolved+TKE change (J/kg) | Reservoir change (J/kg) |
|---|---:|---:|
| Cold front | +1.09829695e−5 | −1.09829695e−5 |
| Warm front | −6.40948354e−6 | +6.40948354e−6 |

Thus Redi debits as well as credits the reservoir; a zero-funded cold case
rejects. Full GM+Redi runs also complete. Across all 36 runs, the largest
**initial-to-final finite-H residual** is **3.73e−16 J/kg**. Maximum local
stage first-law residual is **6.66e−17 J/kg**. Reference-relative temporal errors
are retained in each JSON; first-order convergence is consistent with the
explicit splitting. No spatial convergence claim is made.

These residuals use exact polynomial differences of the stored initial/final
states, independently of summing stage residuals. Direct subtraction of the
large full-H totals is also recorded; its largest difference is 8.15e−12 J/kg,
limited by cancellation/roundoff in the large reference thermal energy. The
small finite-work numbers should not be read as extra precision in that direct
subtraction.

## Reproduce and remaining scope

```sh
python -m pytest -q tests/ocean/test_gmredi.py tests/ocean/test_eddy_reservoir.py
python validation/temperature_coordinate/eddy_campaign.py --front density \
  --days 2 --dt 3600 1800 900 450 --eddy-energy .1 --output reservoir-front32.json
python validation/temperature_coordinate/eddy_campaign.py --front cold \
  --days .25 --dt 600 300 150 75 --eddy-energy .1 --output reservoir-cold32.json
python validation/temperature_coordinate/eddy_campaign.py --front warm \
  --days .25 --dt 600 300 150 75 --eddy-energy .1 --output reservoir-warm32.json
```

This is conservative energetic bookkeeping with explicit local feasibility
checks. It does not yet supply a calibrated eddy-energy production/transport
closure, energy-dependent coefficients, EKE/EAPE partition, backscatter or
dissipation. The specified midpoint energy flux can redistribute the reservoir
locally even when the net conversion is small; its spatial distribution needs
physical qualification. Zero reservoir is not guaranteed to bootstrap GM.
Positivity-preserving Redi, exact neutral directions, boundary tapering, GPU,
float32, compiled kernels, long backward rollouts, target-resolution performance
and long climate integrations remain unqualified. Earlier transport-only
results are preserved in the historical README and `front32.json`.
