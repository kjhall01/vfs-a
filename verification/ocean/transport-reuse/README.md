# Reuse transport intermediates and omit overwritten targets

Reference: `214da28` (donor-reference limiter cleanup).

The work-constrained transport now obtains its target limiter and trace changes
from a private helper. It constructs the final authoritative amounts/fluxes
only after applying the work constraint, omitting the intermediate target
amount scatter and flux allocation. Public `local_target` still materializes
and returns its complete prescribed-carrier ledger. Its signature is unchanged.
All bound checks, positivity checks, energy pairing and numerical defaults remain.

Repeated `beta * anti` products are reused within each transport calculation.
The local target reuses its positive/negative antidiffusive flux parts for the
two incident-face budget reductions. We leave carrier-volume clamp expressions
separate: sharing those altered backward summation at roundoff. The retained
change matches the reference gradients exactly in the captured cases.

## Verification

Separately installed wheels, tested outside the source checkout, capture six
front/coast cases with tendencies, full diagnostics, three resolved steps,
donor/FCT ledgers and one-step backward derivatives. All **486 tensors match
bitwise**; see `equivalence.json`. No frozen fixture or tolerance is changed.
Wheel hashes are recorded in `wheels.json`.

## Measurement method

Fresh sequential processes using the same benchmark as the preceding pass:
16384 cells × 40 layers, float64, one CPU thread, eager execution, one warm-up
and three timed steps. No tests run concurrently with either timing process.
The benchmark includes resolved dynamics and reservoir transport, but not
GM–Redi or eddy decay. Peak RSS includes mesh/runtime/allocator memory.
Raw timings and source hashes are in `before.json` and `after.json`.

```sh
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode check --output states.pt
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode time --output timing.json
```

This does not qualify GPU/MPS, compilation, float32, long integrations or
long backward-rollout memory.

Final installed-wheel portable regression: **1203 passed, 2 skipped**, in 196.90 s.
Corrected-Gibbs and dry-helper fixture comparisons pass all 142 and 93 tensors,
respectively. Full test log and fixture summaries are retained alongside this
report.

## Measured result

Median step time falls **17.359 → 15.063 s (13.2%)**. Peak process RSS falls **2.658 → 2.574 GB (3.2%, 85 MB)**.

Before samples: [17.359356388002197, 17.454933821998566, 16.711983647001034]. After samples: [14.666377506997378, 16.261327833002724, 15.063076458005526].

The final implementation releases the original ledger before constructing
the final amounts; retaining it longer increased peak RSS in an initial trial.
Three samples provide a short CPU comparison, not a hardware-independent
speedup guarantee. GB/MB use decimal units.
