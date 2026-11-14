# GM–Redi reservoir optimization

Reference: `774e2f3803878f07c91bdfe5be3b91ac38d0d493` (the first funded
Hamiltonian implementation). This pass changes evaluation and allocation order,
not the parameterization, coefficients, stage order or funding policy.

## Changes

- Build layer heights, vertical derivative indices/weights and edge metrics
  once per fixed-h/u closure call. Recompute density slopes after GM as before.
  Build the GM circulation only for GM and the rotated stencil only for Redi.
  Nothing is cached across calls or attached to the model/state.
- Evaluate Redi row gradients as batched dot products, avoiding a large
  edge × layer × eight stencil entries × tracer product temporary.
- Factor the Redi energy-flux scalar product before distributing it over the
  stencil, and release tracer-transfer storage after its transpose application.
- Reuse input concentrations during unpacking and compute scalar finite work
  without constructing unused passive/TKE/eddy conjugate columns.
- Validate the whole trial state after transport, then only the changed
  reservoir after settlement. Likewise, the resolved-step/ALE adapters reuse
  the wrapped model's validation of physical fields and separately check e.
  All finite-value, positivity, thermodynamic-domain, coast, funding and step
  restrictions remain active. New invalid-input tests cover both adapters.

The same full Hamiltonian supplies all energy work. Arithmetic regrouping
changes floating-point reduction order; it does not promise bitwise-identical
long trajectories or unchanged decisions exactly on a floating-point guard.

## Numerical equivalence and regression

`comparison.json` records 18 cases: density/cold/warm fronts, with/without
coasts, and GM-only/Redi-only/combined transport, each advanced five steps.
Reference modules are loaded from the pinned Git revision in a separate process.
487 tensors are compared, including every report field and an active backward
sensitivity; 387 are bitwise identical.

All resolved state fields and turbulent TKE are bitwise identical in these
cases. Maximum reservoir-content difference is 1.21e-13 m³/s². The backward
sensitivity differs by 6.39e-18. Differences in local energy-flux and work
reports are recorded explicitly; local/full-H closure tests still pass.

28 focused tests pass, including six invalid-input adapter cases. The outside-checkout portable
installed-wheel suite passes **1186 tests, 2 skipped**, in 201.98 s. Frozen
comparisons pass **142 corrected-Gibbs** and **93 dry-helper** tensors. Logs and
JSON summaries are recorded with this report. Production VFS sources/defaults and the existing installed wheel are
unchanged.

## Timing method

Run the two timings sequentially, without concurrent tests or simulations.
Each uses a fresh process, one warm-up per operation, three measured repetitions,
CPU float64, one thread and no compilation. The mesh is newly generated
(16,384 cells, zero Lloyd iterations, compatible rotation), with 40 exponential
layers to 4000 m, dt=120 s and a small free-surface pulse. There is one passive
tracer, zero turbulent TKE and e=0.1 J/kg. This is comparable to the wave timestep
screen but is not the user's saved mesh or compiled Mac environment.

Components are timed separately on the same initialized state. Estimated total
step time is the sum of the dynamics-with-reservoir and closure medians; it is
not a timed long integration. Source hashes, individual samples and whole-process
peak resident memory are retained in `before.json` and `after.json`. The dynamics
without a reservoir are an unchanged control for machine timing variation.

## Measured results

| Component | Before median (s) | After median (s) |
| --- | ---: | ---: |
| Dynamics without reservoir (unchanged control) | 14.077 | 13.423 |
| Dynamics with reservoir | 15.925 | 16.683 |
| Funded GM–Redi closure | 3.910 | 3.356 |
| Estimated complete step (sum of medians) | 19.835 | 20.040 |

Closure time decreases 14.2% (0.554 s); whole-process peak RSS decreases
7.4%, from 3.526 to 3.267 GB. This does **not** establish an end-to-end speedup:
the summed total increases 1.0%, while unchanged-control timing shifts 4.6%
and reservoir-dynamics timing shifts in the opposite direction. Individual
optimized closure samples range from 3.118 to 3.681 s. Three repetitions on
this host support a closure/allocation improvement, not a precise overall
throughput estimate. A smaller 2048×40 profile showed a larger closure gain;
the target-resolution measurements above take precedence. Compiled Mac
performance must be measured separately.

## Reproduce

```sh
python -I validation/temperature_coordinate/eddy_performance.py \
  --revision 774e2f3803878f07c91bdfe5be3b91ac38d0d493 --mode check --output before.pt
python -I validation/temperature_coordinate/eddy_performance.py \
  --mode check --output after.pt
python -I validation/temperature_coordinate/eddy_performance.py \
  --revision 774e2f3803878f07c91bdfe5be3b91ac38d0d493 --mode time --output before.json
python -I validation/temperature_coordinate/eddy_performance.py \
  --mode time --output after.json
```

The `check` outputs are named tensors loadable with
`torch.load(path, weights_only=True)`. Use the JSON's per-field differences to
inspect the comparison; raw .pt snapshots are reproducible intermediates, not
new frozen production baselines. GPU/MPS, compiled closure performance, float32
and long backward rollout performance remain unqualified.
