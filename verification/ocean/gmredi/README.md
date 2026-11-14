# Historical GM–Redi transport-only qualification

This report describes the initial unfunded prototype at commit
`b76c0b4266b6384e0b83bf5eafe04db3e9f8539d`. The current implementation includes
an explicit reservoir and refreshed Redi slopes; see
[the energy-reservoir qualification](ENERGY_RESERVOIR.md). Reproduction of the
historical numbers below requires that initial commit.

Base: `validation/ocean-timestep-campaign`, commit
`73fd69936367a8811a5788db9f94b1636eb1398b`. Implementation and runnable examples:
[GMREDI.md](../../../validation/temperature_coordinate/GMREDI.md).
No existing production expressions/defaults or frozen fixtures are changed.

## Front relaxation and timestep refinement

`front32.json` records a 32-cell, 8-layer, 100 km-radius spherical domain,
1000 m deep, run for two days with prescribed coefficients 1000 m²/s.
CT=10°C; SA has a stable linear vertical profile plus a smooth lateral front.
A positive passive tracer varies horizontally and vertically. Resolved
velocity is zero and only eddy transport is stepped. This is an operator
experiment, not a baroclinic-instability or Earth climate validation.

At dt=3600 s:

| Operator | Horizontal SA variance, initial → final (g/kg)² | Passive variance, initial → final | Energy change (J/kg) |
|---|---:|---:|---:|
| GM | 0.00749577 → 0.00704546 | 0.01351649 → 0.01348224 | −5.58636e−4 |
| Redi | 0.00749577 → 0.00749577 | 0.01351649 → 0.01266917 | 0 |
| GM + Redi | 0.00749577 → 0.00704579 | 0.01351649 → 0.01263733 | −5.57403e−4 |

GM reduces horizontal density contrast and releases potential energy. Pure
Redi preserves this particular density front while smoothing the passive tracer.
Because slopes are frozen before the GM substep, combined Redi contributes
+1.25459e−6 J/kg; the split-step lagging effect is explicitly diagnosed, not hidden.
Inventory drifts at this dt are at most 5.2e−17 relative to initial inventories.
Maximum GM outflow fraction is 0.00096144, Redi step number 0.0150451.
Maximum local circulation divergence residual is 5.82e−11 m³/s.

Errors relative to the dt=450 s solution:

| Operator / field | dt=3600 s | dt=1800 s | dt=900 s |
|---|---:|---:|---:|
| GM / SA RMS (g/kg) | 1.12344e−6 | 4.81258e−7 | 1.60384e−7 |
| Redi / passive RMS | 1.30048e−6 | 5.57107e−7 | 1.85662e−7 |
| GM + Redi / passive RMS | 1.46265e−6 | 6.26601e−7 | 2.08825e−7 |

Successive-difference ratios are approximately 2, consistent with first-order
time convergence. The finest run is a numerical reference, not an exact solution.
No spatial convergence or neutral fidelity in nonlinear T/S fronts is established.

## Focused checks

13 focused tests cover open-ocean and coastal conservation, divergence-free
circulation, closed boundary fluxes, constant-tracer preservation, tilted-layer
metric cancellation, a reference-density nullspace, variance nonincrease,
zero-coefficient identity, weak-stratification shutoff, explicit step rejection,
negative-tracer rejection without mutation, independent finite Hamiltonian work,
and composition with three resolved ocean steps.

A localized passive tracer **does go negative** under this rotated Redi stencil,
even at dt=1 s in the test. Rejection is tested; positivity preservation is not
claimed. A bound on variance cannot replace a maximum-principle guarantee.

CPU float64, PyTorch 2.15.0+cpu. GPU, float32, compiled eddy kernels, backward
rollouts, 16,384×40 performance, long integrations and climate behavior are
unqualified. There is no prognostic eddy-energy closure or energy compensation.

## Reproduction

```sh
python -m pytest -q tests/ocean/test_gmredi.py
python validation/temperature_coordinate/eddy_campaign.py \
  --output verification/ocean/gmredi/front32.json
```

The unchanged production wheel is installed separately. The tests/research
harness are also exercised outside checkout so importing VFS cannot fall back
to the source tree. Final outside-checkout portable regression: **1171 passed, 2 skipped** in
196.70 s; see `installed-wheel-tests.log`. Corrected-Gibbs fixture comparison:
**142 tensors passed** (`baseline.json`); dry-helper comparison: **93 tensors
passed** (`dry-helpers.json`). Installed wheel SHA256:
`555f50ff1b8e42ccd41a63bf7cc3dffa417fafc1fadd985222de699e01ad18e8`.

The first attempt used a user-site wheel: 1164 passed, 2 skipped and seven
installation-isolation checks failed because `python -I` hides the user site.
Reinstalling the same production sources and dependencies into a virtual
environment resolved all seven failures; no test expectations were changed.
The final run includes the content-based finite-work diagnostic and its
independent midpoint-gradient test.

## References

- Gent & McWilliams (1990), *Isopycnal Mixing in Ocean Circulation Models*,
  [doi:10.1175/1520-0485(1990)020&lt;0150:IMIOCM&gt;2.0.CO;2](https://doi.org/10.1175/1520-0485(1990)020%3C0150:IMIOCM%3E2.0.CO;2).
- Redi (1982), *Oceanic Isopycnal Mixing by Coordinate Rotation*,
  [doi:10.1175/1520-0485(1982)012&lt;1154:OIMBCR&gt;2.0.CO;2](https://doi.org/10.1175/1520-0485(1982)012%3C1154:OIMBCR%3E2.0.CO;2).
- [MITgcm GM/Redi documentation](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/gmredi.html)
  for the continuous skew-flux/bolus and rotated diffusion formulations. The
  VFS discretization here is a research implementation, not an MITgcm port.
