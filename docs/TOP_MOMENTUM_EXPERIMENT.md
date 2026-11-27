# Experimental first-interface donor momentum

Centered transport remains the default. This experiment covers the first
interior atmospheric interface only; the Hamiltonian, mass, species and
cloud transfer formulas are unchanged. To opt in when constructing a moist
model for a test, use `BoundedSSPRK3(top_momentum_upwind=True)`; the coupled
SSPRK(4,3) stepper uses the same bounded Euler stages when called on this model.
The coupled campaign configuration itself does not expose or enable the option.

Let F be the signed downward edge carrier in kg/s at the first interior
interface and a, b the top and second-layer edge-normal winds. The additional
downward momentum flux, donor minus centered, is |F| (a - b) / 2. It enters the
top and second layer's momentum budgets with opposite signs. Dividing the
impulses by their respective kinetic metrics gives the acceleration. The work
of this force pair is -|F| (a - b)^2 / 2. Divide the corresponding positive
heat between the two adjacent columns, at the first interior CP interface,
by the discrete energy derivative with respect to entropy content. This
closes the extra tendency's energy budget; it does not imply exact finite-step
energy for the Runge–Kutta scheme. The enabled bounded-stage report exposes
`top_momentum_heat_W` as one nonnegative power per edge.

## Reproduce the short restart experiment

Use a separate checkout at `3605b88` with the experimental patch applied.
Use the day-43.9027778 Earth shelf NetCDF restart (model second 3,793,200) and
the original ordered NumPy mesh. This reads the checkpoint without modifying
it. It does not advance PyAM2.1, radiation, ocean or ice, and does not confer
production restart compatibility.

```sh
python tools/replay_top_momentum.py "$RESTART" --mesh "$MESH" --scheme centered > centered.jsonl
python tools/replay_top_momentum.py "$RESTART" --mesh "$MESH" --scheme donor > donor.jsonl
```

At 600 s, four simulated hours, CPU float64 with the four-stage integrator:

| Measurement | Centered | Top donor |
| --- | ---: | ---: |
| Maximum sampled edge wind over replay (m/s) | 324.97 | 265.92 |
| Maximum edge wind at centered peak time (m/s) | 324.97 | 194.53 |
| End-hour-four top-level edge-wind 99th percentile (m/s) | 149.72 | 105.52 |
| Four-hour atmospheric numerical energy residual (J/m²) | -466.40 | -436.51 |

Samples are every 600 s, including the initial 256.87 m/s. This is a matched
one-restart result, not evidence for long-term climate validation. The day-49
spike, coupled radiation, moist baroclinic-wave fidelity and GPU behavior have
not been qualified. The experimental code changes the full source fingerprint:
keep an ongoing campaign running its current source. No compatibility waiver
for resuming an existing campaign was added or tested.
