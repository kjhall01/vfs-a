# Skip discarded FCT work in the donor reference

Reference: `96f547b`, the preceding dynamics tensor-lifetime optimization.

`HorizontalTransport.advance` previously computed positive/negative
antidiffusive budgets, cell capacity ratios and an edge FCT limiter even in
donor mode, then replaced that limiter with zeros. The outer stratified
dynamics constructs its own energy-consistent corrections using this donor
reference. This pass skips only the inner calculations that were discarded.

Donor mode now creates the identical zero limiter directly from the
antidiffusive-flux shape. It still computes and returns the high/donor fluxes,
local extrema/bounds and complete transport ledger used by its callers. All
amount arithmetic, admissibility checks, positivity checks and the outer
higher-order/work correction are retained. The FCT branch uses the original
limiter expressions. No transport scheme or numerical default is changed.

## Verification method

Build a new wheel and install it outside the checkout. Run the existing
`dynamics_performance.py --mode check` before and after, capturing six
front/coast cases, three resolved steps, complete tendency/ledger outputs,
both donor and FCT transport, and backward gradients. Require exact tensor
equality; do not adjust fixtures or tolerances. Run the installed-wheel
portable suite and both frozen-fixture comparisons.

All **486 captured tensors are bitwise identical**, including backward gradients,
across the separately installed before/after wheel executions. `equivalence.json`
records this exact comparison; no fixture or tolerance was changed.

Rebuilt installed-wheel portable suite outside the checkout: **1203 passed,
2 skipped**, in 199.34 s. The corrected-Gibbs and dry-helper comparisons pass
all 142 and 93 tensors. Wheel hashes, fixture summaries and the full test log
are retained. Production changes are confined to this transport branch.

## Performance method

Fresh sequential processes, one warm-up and three timed resolved steps with
a reservoir at 16384 cells × 40 layers, CPU float64, one thread, no compilation.
No tests or other simulations run alongside timing. The benchmark includes
resolved dynamics and reservoir transport, but not GM–Redi or eddy decay.
Peak RSS is whole-process memory including mesh, runtime and allocator state;
removing one temporary does not necessarily lower the peak set elsewhere.

Raw measurements and source hashes are in `before.json` and `after.json`.
Reproduce with each version's installed wheel:

```sh
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode check --output states.pt
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode time --output timing.json
```

GPU/MPS, float32, compiled performance and long backward rollouts are not
qualified by this eager CPU timing or the short exact comparisons.

## Measured result

| Measurement | Before | After | Reduction |
|---|---:|---:|---:|
| Median step time | 18.433 s | 17.233 s | 6.5% |
| Peak process RSS | 2.654 GB | 2.600 GB | 2.0% (54 MB) |

Timing samples were 18.317–18.666 s before and 16.991–17.441 s after.
These three-sample measurements show a modest improvement on this CPU setup;
they do not establish the gain on other hardware or with compilation. Memory
uses decimal GB/MB. Most of the footprint remains in the retained dynamics,
transport ledgers and runtime allocations.
