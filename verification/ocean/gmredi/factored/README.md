# Factored Redi gradient and transpose

Reference: `be1fd29eed92d0d6f716c560f30a04062a1534e1` (the preceding allocation/validation optimization).
This published commit has the identical Git tree as local `300c19c`, the revision
recorded in the original measurement JSON. Publication changed commit metadata
only; model source hashes are unchanged.

## Operator

Redi now evaluates `R = H + S A D` in separate operations:

- `D`: the existing three-point, cell-centered vertical derivative on uneven
  layer heights, including its one-sided end rows;
- `A`: arithmetic averaging from the two adjacent columns to active edges;
- `S`: density slope relative to the layer-coordinate slope;
- `H`: across-edge concentration difference divided by edge distance.

The reverse application is `R^T = H^T + D^T A^T S`. Horizontal transpose
uses the existing ocean `divergence` ledger. Vertical derivatives and their
transposes use one stencil entry at a time at cell centers. The ordinary
vertical-diffusion routine is not substituted: its interface staggering and
boundary closure differ from this cell-centered derivative.

Tracer updates are `-dt R^T M R C`, where `M` is the original diffusivity and
edge-volume measure. The conservative Hamiltonian energy inflow uses the same
factored transpose and the original midpoint edge conjugate. Local reservoir
settlement, density refresh after GM, slope limits, weak-stratification mask,
active-edge/coast mask and positivity/domain/funding guards are unchanged.
The explicit bound factors the original absolute eight-entry stencil; it does
not coalesce duplicate center entries and thus retains the previous bound.

No edge × layer × eight entries × fields gather or transfer tensor remains.
For 16384×40 and five fields, each removed float64 tensor would be approximately
629 MB. Cell-field intermediates are approximately 26 MB and edge-field
intermediates approximately 79 MB. Several smaller intermediates still coexist;
whole-process memory also includes GM, dynamics, mesh and allocator overhead.
No hidden caches, forward autograd tendencies, clipping or compilation changes
are introduced. The private geometry helper now returns an operator and measure
instead of expanded rows and weights.

## Verification

Four independent comparisons reconstruct the original eight-entry stencil for
open/coastal domains and one/five fields. They check the forward gradient,
transpose, explicit bound, adjoint identity, conservative total and negative
variance-production identity. Existing tilted-grid nullspace tests retain their
tolerances, as do local/full Hamiltonian and failure-guard tests.

The 18 five-step reference cases compare 487 tensors (387 bitwise equal).
Velocity, thickness, temperature content, salinity content and turbulent TKE
are bitwise identical. Maximum passive-content difference is 1.42e-14;
maximum eddy-content difference is 8.53e-14 m³/s². One active backward sensitivity
differs by 1.00e-18. All state comparisons meet rtol=5e-13, atol=2e-11.

`comparison.json` records every report difference, including extensive energy
quantities in joules. Reordering cancellation changes near-zero energy reports;
they are not all bitwise identical and are not judged with a dimensionally
inappropriate common absolute tolerance. Independent local and global energy
assertions remain unchanged. No long-trajectory bitwise equivalence is claimed.

Installed-wheel testing outside the checkout: 1189 passed and 2 skipped in
201.49 s, with one exact atmospheric fixture failure (initial velocity differs
by at most 4.44e-15). Rerunning `tests/test_extraction.py --fixture-mode portable`
passes all 3 tests. Thus all 1190 distinct tests pass using the established
portable fixture policy. This is a combined result, not a second full-suite run.
The separate 142 corrected-Gibbs and 93 dry-helper portable fixture comparisons
also pass. Exact-run and portable-rerun logs are retained; no assertion or
fixture was changed to accommodate the platform difference.

## Timing

Before/after use the same `eddy_performance.py` driver in fresh sequential
processes: CPU float64, one thread, no compilation, 16384 cells, 40 layers,
one warmup and three measured repetitions per operation. Components are timed
separately, not as a long integration. Their summed medians are only an estimated
full step. Whole-process peak RSS includes dynamics; it is not closure-only
memory. The unchanged dynamics are a timing-variation control. Raw samples,
source hashes and driver hashes are in `before.json` and `after.json`.

| Component | Before median (s) | After median (s) |
| --- | ---: | ---: |
| Dynamics without reservoir (unchanged) | 14.631 | 13.606 |
| Dynamics with reservoir (unchanged) | 17.619 | 16.843 |
| Funded GM–Redi closure | 3.184 | 1.649 |
| Estimated full step (sum of medians) | 20.803 | 18.493 |

Closure median time decreases 48.2%. The estimated full-step total decreases
11.1%, but unchanged dynamics also speeds up by 4.4–7.0%, so this is not a precise
attribution of total speedup. Holding reservoir dynamics at its reference timing,
the closure saving alone corresponds to about 7.4% of that estimated full step.
Before closure samples span 2.906–3.311 s; after samples span 1.561–2.014 s.
Whole-process peak RSS is effectively unchanged: 3.265 → 3.287 GB. Removing
closure intermediates does not establish a smaller peak for the full model.

A second pair of fresh processes uses `--closure-only` (setup and GM–Redi,
no dynamics). Peak RSS decreases **26.3%, 2.396 → 1.765 GB**. Closure medians
are 3.614 → 2.587 s (**28.4% less time**) in this separate pass. The difference
from the earlier 48.2% timing reduction illustrates host variability; both
passes show a substantial closure improvement. This isolates the closure's
process footprint, not its tensor-only allocations: mesh/setup and allocator
overhead are still included. The complete-model peak is not reduced by this
change. Raw isolated samples are in `closure-before.json` and
`closure-after.json`.

The benchmark driver gained only the optional `--closure-only` filter after
the first timing pair, which explains the two driver hashes. Default operations
and model source hashes are unchanged between the timing pairs.

Reproduce from an environment with the VFS wheel installed:

```sh
python -I validation/temperature_coordinate/eddy_performance.py \
  --revision be1fd29eed92d0d6f716c560f30a04062a1534e1 --mode check --output before.pt
python -I validation/temperature_coordinate/eddy_performance.py \
  --mode check --output after.pt
python -I validation/temperature_coordinate/eddy_performance.py \
  --revision be1fd29eed92d0d6f716c560f30a04062a1534e1 --mode time --output before.json
python -I validation/temperature_coordinate/eddy_performance.py \
  --mode time --output after.json
```

Add `--closure-only` to both timing commands for the isolated-process comparison.

GPU/MPS, float32, compiled timing and long backward rollouts remain unqualified.
