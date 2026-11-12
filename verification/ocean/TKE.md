# Initial ocean TKE qualification

Historical evidence for the first column-allocation implementation in #51.
The [local interface version](LOCAL_TKE_BUOYANCY.md) supersedes that allocation;
these original measurements and source hashes remain unchanged.

The optional [TKE API](../../docs/OCEAN_TKE.md) adds a prognostic h*k reservoir
without changing existing ocean or atmospheric kernels. These checks establish
specific discrete budgets and basic numerical behavior, not calibrated turbulence.

## Reproduce

Install a wheel, then run from outside the checkout with the copied examples,
validation and test directories:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I -m pytest -q tests/ocean/test_tke.py
python -I examples/ocean_tke.py
python -I validation/ocean_tke/run.py --output /tmp/tke-results.json
```

No external dataset, cloud access, plotting package or GOTM dependency is needed.
[Recorded results](tke/column-results.json) include installed-source hashes, driver
hash, Torch version, transfer ledgers and numerical measurements. CPU float64,
Torch 2.15.0+cpu; deterministic 32-cell Fibonacci mesh with two Lloyd iterations
and compatible rotation. This mesh supports independent-column accounting tests;
it does not establish fine-mesh ocean circulation accuracy.

## Repeated vertical mixing

Closed 100 m columns, 4/16/32 layers, temperatures 15→12 C IPTS-68 and salinity
34.8→35.2 from top to bottom, edge-normal velocities 0.3→-0.3 m/s, initial
k=0.01 m²/s². L=2 m, C_mu=0.1, C_epsilon=0.7, Pr=sigma_k=1. Twenty 10 s steps,
vertical physics only. These are illustrative parameters, not a calibration.

| Layers | Combined H change (J/kg) | Maximum single-step column residual (J/kg) | Final minimum k (m²/s²) |
| ---: | ---: | ---: | ---: |
| 4 | 1.03e-11 | 2.41e-11 | 4.87e-4 |
| 16 | 1.03e-11 | 1.11e-11 | 4.78e-4 |
| 32 | 0 | 8.58e-12 | 4.75e-4 |

Column residuals account for the explicitly reported shared-edge kinetic exchange.
Actual global H differences are measured separately; zero printed difference
means below the resolution of that subtraction, not mathematical exactness.
Transfer-level residuals avoid subtracting the full background where possible.
Salt relative change is zero in these cases; passive inventory relative changes
are <=3.14e-16. Column entropy increases and k stays positive. The finite-step
momentum damping is recorded separately from shear production.

A subsequent test executes the usual dynamics step and ALE with TKE packed as an
additional passive content. TKE transport/remap residuals are below 9e-19 J/kg in
the recorded examples. The combined remap energy change is <=1.03e-11 J/kg.
These checks are not a repeated dynamics-plus-ALE climate qualification.

## Independent and regression checks

Seventeen offline tests cover:

- Tridiagonal diffusion against an independently assembled dense solve; weighted
  inventory conservation, bounds, exact constant-field behavior.
- Native resolved KE loss versus TKE gain, integrated edge momentum and the
  backward-Euler energy identity.
- Analytic fixed-length TKE decay, positive heating and full-energy checks.
- Stable and unstable buoyancy-work signs, salt/passive conservation.
- Zero-TKE no-op without a floor; disabled closure reproducing the resolved
  dynamics bitwise in the regression case.
- Nonuniform TKE with zeros through bounded transport and ALE; absent passives.
- Closed coasts, dry placeholders, one layer, invalid inputs and explicit
  rejection when mixing exceeds the available TKE reservoir.
- Smooth directional derivatives through momentum/TKE and thermal/buoyancy paths.
- Unequal neighbouring thicknesses checked against the existing column
  Hamiltonian decomposition, including the zero-global-sum kinetic exchange.

A separate two-box diffusion reference has an exact continuous exponential
solution. Max errors at 4/8/16/32/64 steps are 0.01013/0.005218/0.002649/0.001335/
0.0006699, consistent with first-order backward Euler. This is not an independent
physical validation of the nonlinear turbulence closure.

See [validation summary](tke/validation.txt) for the installed-wheel suite and
unchanged portable fixture comparisons. Historical fixtures/tolerances are not
modified; exact historical reproducibility is distinct from portable checks.

## Outstanding physical qualification

The closure is one-equation, prescribed-length and unforced. It lacks stability-
dependent length scales, boundary TKE fluxes, Langmuir effects, bottom drag and
horizontal EKE. Buoyancy energy is allocated within a column rather than through
a local layerwise production law. The next physical tests should address
entrainment and mixed-layer deepening, shear/buoyancy competition, a local
buoyancy-work allocation and wind-forced Ekman dynamics. Long integrations,
repeated difficult remaps, GPU/float32 and long backward trajectories remain
unqualified. Energy closure alone cannot certify these behaviors.
