# Coupled freshwater verification

This pass adds prescribed freshwater exchange to the shared ocean–ice–snow
Hamiltonian. [API and boundary conventions](../../../../validation/sea_ice/FRESHWATER.md)
explain routing, thermal references, momentum, incorporation work and limitations.
All results below use CPU float64, PyTorch 2.15.0+cpu, one torch thread.

## Installed-wheel regression

A new wheel was built and installed, then `tests`, `tools`, `examples` and
`validation` were copied to an external harness. Commands ran there with `-I`,
so the package came from site-packages and research code from the explicit copy.
No packaged numerical source changed in this pass.

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable --report baseline-portable.json
python -I tools/check_dry_helpers.py --mode portable --report dry-portable.json
```

- [Full portable suite](installed-wheel-pytest.log): **1330 passed, 2 skipped**.
- [Corrected-Gibbs comparison](baseline-portable.json): all 142 tensors pass.
- [Dry-helper comparison](dry-portable.json): all 93 tensors pass.
- 22 new freshwater tests cover individual and combined mass/momentum/full-H
  ledgers, dry/covered/open routing, salt/tracer/reservoir retention, explicit
  ice-loading work, snow-first sublimation, complete cover loss, river mass/heat
  mapping, invalid forcing and exhaustion, no-flux equivalence, complete-step
  composition and a smooth rain-temperature derivative versus finite difference.
  Forward tests prohibit `torch.autograd.grad` inside the boundary.

The established portable tolerances were unchanged. The warnings in the suite
log concern pytest plugin configuration and an existing tensor-to-scalar test.
No GPU or long-horizon backward qualification is implied.

## Six-hour experiments

Seven integrations complete: **1980 steps, 42 aggregate model hours**. Each
case directory contains `summary.json` and every-step `series.csv`; the top-level
case logs retain progress and final summaries. Each summary records the exact
SHA256 of the sea-ice source files used. Those hashes match this implementation.

All runs use the existing idealized ocean initial state, moving category ice,
a 2 m/s prescribed eastward wind and a separate -10 W/m² heat flux. Mixed forcing
uses rain 2e-5 kg/m²/s at 5°C, snowfall 1e-5 kg/m²/s, open-water evaporation
2e-5 kg/m²/s, covered-area sublimation 2e-6 kg/m²/s, and two rivers with discharge
2e5/1e5 m³/s at 5/10°C. Rain and rivers arrive stationary; snow is co-moving.
River recipients are explicit wet cells, adjacent to dry cells in coastal runs.
Single-process cases retain the same wind/heat forcing and omit other freshwater
processes. These are deliberately idealized numerical experiments.

| Case | Cells | dt (s) | Steps | Peak energy residual (J/m²) | Peak relative water residual | Peak relative salt residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [mixed32](mixed32/summary.json) | 32 | 120 | 180 | 1.042e-06 | 3.821e-16 | 2.952e-16 |
| [mixed32-dt60](mixed32-dt60/summary.json) | 32 | 60 | 360 | 1.745e-06 | 9.528e-16 | 4.429e-16 |
| [mixed32-dt30](mixed32-dt30/summary.json) | 32 | 30 | 720 | 1.079e-06 | 1.053e-15 | 5.905e-16 |
| [coast128](coast128/summary.json) | 128 | 120 | 180 | 7.369e-07 | 2.394e-16 | 3.935e-16 |
| [evaporation](evaporation/summary.json) | 32 | 120 | 180 | 6.517e-07 | 4.353e-16 | 1.476e-16 |
| [sublimation](sublimation/summary.json) | 32 | 120 | 180 | 6.971e-07 | 7.538e-16 | 2.952e-16 |
| [runoff-coast](runoff-coast/summary.json) | 32 | 120 | 180 | 1.603e-06 | 4.948e-16 | 1.964e-16 |

Energy residual means `[H(t)-H(0)-accumulated external energy]/wet area`.
Water uses combined liquid+frozen mass after subtracting the diagnosed external
water supply, normalized by initial combined mass. Salt is normalized by its
initial content. Snow labels do not add a second frozen-mass reservoir. The
maximum cell-local boundary energy residual over all runs is below 4.39e-7 J/m².

Mixed-case dt refinement (120/60/30 s) gives final maximum ice speeds
0.0718010/0.0719323/0.0719655 m/s and final total frozen masses
1.18109760e17/1.18110851e17/1.18111396e17 kg. Successive differences shrink, but
these selected diagnostics do not establish an overall convergence order.
The energy residual is not monotonic under refinement at this near-roundoff
level. Exact finite boundary accounting does not eliminate splitting or
integration error in the coupled trajectory.

Reproduce from an installed wheel and matching research files:

```bash
python -I validation/sea_ice/run_freshwater.py --hours 6 --dt 120 --output mixed32
python -I validation/sea_ice/run_freshwater.py --hours 6 --dt 60 --output mixed32-dt60
python -I validation/sea_ice/run_freshwater.py --hours 6 --dt 30 --output mixed32-dt30
python -I validation/sea_ice/run_freshwater.py --coast --cells 128 --hours 6 --dt 120 --output coast128
python -I validation/sea_ice/run_freshwater.py --case evaporation --hours 6 --dt 120 --output evaporation
python -I validation/sea_ice/run_freshwater.py --case sublimation --hours 6 --dt 120 --output sublimation
python -I validation/sea_ice/run_freshwater.py --coast --case runoff --hours 6 --dt 120 --output runoff-coast
```

This demonstrates short CPU float64 coupling and bookkeeping for prescribed
fluxes. It does not validate a realistic runoff climatology, river plumes,
atmospheric bulk transfer laws, long integrations or climate response. Large
forcing can exhaust stores or leave the EOS temperature/salinity domain and
then rejects; no clipping, retry or hidden flux reduction is applied.
