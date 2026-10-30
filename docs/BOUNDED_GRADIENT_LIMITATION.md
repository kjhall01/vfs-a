# Bounded gradients: fixed validation and remaining limits

PR #6 is merged. The separate tensor-tolerance fix changes only the flux-sum
validation in `bounded_transfer`; it does not alter transport, limiting, heat
allocation, SSPRK3, defaults or the direct Euler species amount ledger.

## Fixed TypeError

On the PR #6 installed wheel, this valid transfer raised
`TypeError: allclose(): argument 'atol' must be float, not Tensor`:

```python
import torch
from vfs.atmosphere.bounded_transport import bounded_transfer

m = torch.ones(2, dtype=torch.float64)
q = torch.tensor([[.9, .1, 0., 0.], [.9, .1, 0., 0.]], dtype=torch.float64)
g = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
j = g[:, None] * q[:1]
amounts, flux, info = bounded_transfer(
    m, q, torch.tensor([0]), torch.tensor([1]), g, j)
amounts[1, 0].backward()  # Now completes, with g.grad = 0.9.
```

The old `atol=128 * eps * scale` is itself a tensor, and carries gradients when
G does. The fix expresses the same comparison elementwise:

```text
error = abs(sum_species(J) - G)
tolerance = 128*eps*scale + 128*eps*abs(G)
require every error finite and error <= tolerance
```

`eps = finfo(Q0.dtype).eps` and
`scale = abs(G).max().clamp_min(finfo(G.dtype).tiny)` are unchanged. Inputs are
already checked finite. The explicit finite-error guard retains allclose's
behavior if finite inputs overflow in summation/subtraction; infinity must not
pass an overflowing tolerance. No tolerance is relaxed or converted to a detached
Python scalar. The boolean validation branch does not participate in the physical
update; prognostic computations retain their original autograd connections.

Regression tests compare the new validation decisions with the old nongradient
allclose reference for float32/float64, both flux signs, zero flux and offsets
inside/on/outside the tolerance. Valid gradient-carrying fluxes backpropagate;
inconsistent flux sums raise ValueError. A CPU bfloat16 overflow case checks only
comparison semantics, not numerical support for that dtype.

## Complete-step directional derivatives

Tests run backward through **one complete public MoistAtmosphere call**, using
BoundedSSPRK3 with release_as_heat=True. Wind, surface pressure, entropy content
and all three water contents are differentiated in four separate input groups.
No detach or no_grad is used to make these tests pass.

The scalar output is a nonuniform sinusoidally weighted mean of all four output
fields, normalized by 10 m/s, 1e5 Pa, 1e7 Pa J/(kg K), and 100 Pa respectively.
Nonuniform weights avoid a conserved global sum masking derivative errors.
Autograd's gradient-dot-direction is compared with independently evaluated
centered differences at dimensionless h = .1, .03 and .01.

The smooth case uses 32 cells, four layers, dt=10 s, varying positive temperature
and all species, and east/north wind 10/2 m/s. Directions vary spatially and have
amplitude 1% of the corresponding normalization scale. Every stage and both
perturbation signs have no active limiter, unchanged subcycle counts and species
fraction margins above their existing bounds by more than 128 float64 epsilons.

Local CPU float64 results (Apple M3, PyTorch 2.15.0 / Accelerate):

| Input | Autograd directional derivative | Absolute error at h=.1 / .03 / .01 |
| --- | --- | --- |
| Wind | 3.227023324385182e-4 | 1.08e-16 / 2.70e-16 / 2.12e-15 |
| Surface pressure | 2.967604991301004e-3 | 8.78e-13 / 7.85e-14 / 1.18e-14 |
| Entropy content | -3.584341220929331e-5 | 4.93e-12 / 4.43e-13 / 4.77e-14 |
| Water content | 1.272792887911802e-5 | 7.20e-14 / 5.53e-15 / 2.10e-16 |

The test allowance combines four times the central-difference truncation estimate
`abs(D(.03)-D(.01))/8` with a subtraction roundoff floor
`64*eps*(1+abs(f_plus)+abs(f_minus))/.01`. This is an implementation-regression
check with a stated numerical error model, not a physical acceptance threshold,
a rigorous bound for arbitrary functions, or a full Jacobian verification.
The recorded error sequences distinguish truncation reduction from roundoff.

## Active limiter and switching

The existing 64-cell/six-layer sharp-cloud example at dt=30 s activates
105, 119 and 166 faces across its Euler stages on the local run. All four input
groups complete backward. Directions scale the corresponding initial contents
by 1%, preserving exact zero-condensate support. Active masks and donor subcycle
counts (1,1,1) stay unchanged for the tested positive/negative perturbations.
Absolute directional errors at h=.01 are, respectively, 2.09e-16, 4.73e-16,
1.57e-15 and 1.10e-15. These are **branch-local observations** along those
particular directions; zero-phase directions are not tested by introducing
negative condensates or silently clipping them.

A separate two-node test deliberately straddles a limiter switch. Initial donor
liquid amount is .05, carrier transfer .1, and the proposed liquid transfer is z.
At z=.05 the existing safe-high branch is taken. The returned donor-liquid amount
has left derivative -1 and right derivative approximately 0; autograd at equality
returns -1 from the selected branch. There is no unique two-sided derivative
there. The small right-side residual comes from the unchanged 64*eps limiter
margin. This test demonstrates the limitation rather than smoothing the limiter.

The existing donor subcycle calculation converts a gradient-carrying scalar to
an integer step count and emits PyTorch's scalar-conversion warning. It is left
unchanged: differentiation follows the selected fixed count, not transitions of
ceil/max. Minima, clamps, flux-direction choices and heat branches likewise do
not imply smoothness at switches. No additional backward exception or nonfinite
gradient appeared in these probes. This does not rule out failures elsewhere.

## Installed-wheel regression and scope

Local macOS 14.6.1 / Apple M3 / Python 3.12.13 / PyTorch 2.15.0 / Accelerate,
CPU float64, one numerical thread. A pre-fix PR #6 installed-wheel capture was
saved outside the repository; the post-fix wheel is tested from a separate
`/tmp/vfs-atol-check` directory with Python `-I`.

Local installed-wheel results:

- Full portable pytest: **595 passed, 1 skipped** (CUDA unavailable), with the
  existing donor-count scalar-conversion warning reported.
- Both portable tools: **142 moist + 193 dry tensors pass**, including the
  existing independent invariants and exact ledger checks.
- Both examples complete 10 × 30 s with finite diagnostics.
- Same-environment pre/post: **335 capture tensors exactly equal**, including
  smooth and active-cloud Euler/SSPRK3 states, ledgers and limiter results.
  Another **37 tensors exactly equal**: centered tendency/energy terms,
  diagnostic energy gradients, centered RHS VJP/fields and a facade forward step.
- Exact comparisons against frozen fixtures: **both standalone tools fail**, as
  before on this CPU/backend; portable passes are not reclassified exact passes.

Fixtures and comparison policies are unchanged. Frozen-fixture exact outcomes
are separate from same-environment forward equivalence. GitHub Actions for this
fix branch has not completed yet; these are local results, not hosted evidence.

Reproduce the derivative/validation probes with
`python -I -m pytest tests/test_bounded_gradients.py -q -s` from the copied test
workspace. This prints directional results, limiter counts and numerical test
allowances. The normal portable suite includes these tests.

What is now covered: comparator behavior, exact same-environment forward
regression, diagnostic gradient preservation and the stated one-step CPU float64
directional checks. Untested: long-rollout backward stability, arbitrary limiter
switches, all input directions, higher derivatives, IMEX backward, accelerator
backward, reduced-precision accuracy and general zero-vapor/dry equivalence.
Positive vapor remains required; no new floor or physics is introduced.
