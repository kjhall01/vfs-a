# Reading one bounded moist step

Start with the default call chain below, then follow the numbered blocks in `evaluate`.
Other explicit stepper choices are described in [the module API](MOIST_MODULE_API.md).
Source docstrings give tensor shapes, units and staggering at each boundary.
No new algorithm or scientific verification campaign accompanies this guide.

1. [MoistAtmosphere.step](../vfs/atmosphere/model.py) calls the module; `forward`
   checks geometry, dtype/device and positive finite `dt` (seconds). The default
   [BoundedSSPRK3.advance](../vfs/atmosphere/steppers.py) then delegates to
   [BoundedGibbs.step](../vfs/atmosphere/bounded_moist.py#L211).
2. [step_with_info](../vfs/atmosphere/bounded_moist.py#L194) performs three
   bounded Euler maps, each with the full `dt` budget. The second RK state is
   `3/4 initial + 1/4 Euler(Euler(initial))`; the final state is
   `1/3 initial + 2/3 Euler(second RK state)`. Combinations operate on contents,
   not fractions. The three returned ledgers describe the Euler stages.
3. [euler_with_info](../vfs/atmosphere/bounded_moist.py#L177) calls
   [evaluate](../vfs/atmosphere/bounded_moist.py#L42), advances the other fields,
   and **installs water from `info['amounts'] / cell_area`**. Keep this exact
   expression; `water + dt*rhs.water` is not a cancellation-safe replacement.

Inside each `evaluate`, in execution order:

1. [graph](../vfs/atmosphere/bounded_transport.py#L15) constructs cell-interface
   carrier amounts and oriented horizontal/vertical transfers. Horizontal faces
   precede vertical faces; positive flux moves from node `i` to node `j`.
2. [_gibbs_corrections](../vfs/atmosphere/bounded_moist.py#L272) calls
   [gibbs_face](../vfs/atmosphere/gibbs_means.py#L47). Its entropy/fraction traces
   satisfy the analytic mixture Gibbs identity. Horizontal Gibbs-minus-arithmetic
   trace changes get a paired momentum correction through
   [pullback](../vfs/atmosphere/bounded_moist.py#L217); vertical changes are zero.
3. [composition_entropy_derivatives](../vfs/atmosphere/gibbs_means.py#L90)
   supplies fixed-temperature/pressure derivatives for changes replacing dry air.
   The base tendency then supplies continuity and the actual Euler carrier endpoint.
4. [limited_traces](../vfs/atmosphere/bounded_transport.py#L187) integrates the
   proposed carrier/species fluxes over `dt`, makes dry transfer complementary to
   water, then calls [bounded_transfer](../vfs/atmosphere/bounded_transport.py#L62).
   Safe high-order amounts return immediately. Otherwise a donor reference may
   subcycle; the full RHS and SSPRK3 do not. One face factor `beta` bounds all
   species against the total outgoing demand over every incident face.
5. `limited_traces` returns the amount ledger plus limiter-minus-Gibbs trace
   changes. `evaluate` contracts those changes with jumps of area-normalized
   energy derivatives. Carrier flux times this contraction is face power.
6. With `release_as_heat=True`, negative scalar work becomes nonnegative heat;
   other work receives a mechanical correction through `pullback`. Each face
   allocates half its heat power to each endpoint, dividing by that endpoint's
   entropy-energy derivative. Equal power does not mean equal entropy increment.
   With release disabled, all work is mechanical and heat is zero.
7. The assembled RHS retains matched momentum/continuity and replaces scalar
   tendencies with limited-flux divergence plus entropy heating. The water RHS
   describes transfer, but the Euler water update still uses the ledger directly.

## Inside the centered moist base

The bounded wrapper calls into `moist_conservative.py` for its base RHS and
energy gradients. Read these functions in execution order:

1. [tendency](../vfs/atmosphere/moist_conservative.py#L469) calls
   [_adapter](../vfs/atmosphere/moist_conservative.py#L425): validate composition,
   then wrap the **same** wind, pressure and entropy tensors and geometry.
   `adapter.mtheta_i.values` is `MoistState.entropy_content`; there is no copy,
   detach or conversion to dry potential temperature. Inherited `theta` accessors
   divide this buffer by pressure thickness, so they mean specific entropy here.
   This private State must not be used for dry thermodynamic diagnostics or I/O.
2. [_assemble](../vfs/atmosphere/moist_conservative.py#L152) obtains continuity
   primitives and centered horizontal entropy transport, then calls
   [mixture_gradients](../vfs/atmosphere/moist_conservative.py#L64). This reconstructs
   two Gauss points per layer, forms species enthalpy/entropy/chemical potentials,
   and pulls temperature and water-minus-dry chemical potentials back to CP
   contents. Its pressure derivative includes quadrature motion, normalization
   at fixed contents and surface geopotential. Kinetic energy is excluded here.
3. `_assemble` pairs entropy and each water-species transport with momentum work
   through explicit transposes. It then assembles skew rotation, centered vertical
   velocity exchange with the changing mass metric, and Bernoulli work. Finally
   it solves for acceleration and completes entropy divergence/continuity.
   `tendency` solves the summed momentum covectors; `tendency_terms` solves each
   component separately and renames the private `mtheta_i` output to entropy.
   These two arithmetic paths are preserved, not merged for apparent simplicity.
4. [energy_terms](../vfs/atmosphere/moist_conservative.py#L504) reconstructs the same
   pressure quadrature, integrates geopotential upward, and sums internal,
   potential, top-reservoir and kinetic energies (J). Interface geopotential is
   a separate `(cells, interfaces)` tensor in m²/s².
5. [energy_gradients](../vfs/atmosphere/moist_conservative.py#L557) starts with
   `mixture_gradients`, adds the surface-pressure dependence of kinetic layer
   mass, then obtains the kinetic wind derivative. Outputs have state shapes,
   but units of **energy per stored variable**, not prognostic-state units.
   [boundary_energy_rate](../vfs/atmosphere/moist_conservative.py#L588) contracts
   these derivatives with an RHS to return watts. The formulas are analytic;
   their tensor results remain connected for differentiation.

The source defines retained symbols `p`, `dp`, `dm`, `S`, `T`, `q` and `pg`,
with units/staggering at the function boundaries. The private entropy adapter,
its repeated access paths, in-place pressure-gradient additions, and the two
momentum assembly paths are unchanged. Redesigning that boundary or combining
those evaluations requires a separate numerical/ownership review.

## Units and conservation landmarks

`C` = cells, `E` = horizontal edges, `L` = layers, `I=L+1` = CP interfaces.
Graph nodes are flattened `(C,I)`; faces are `(E,I)` followed by `(C,L)`.
Wind is `(E,L)` in m/s; pressure is `(C,1)` in Pa. Entropy and water contents
live at cell interfaces, in Pa J/(kg K) and Pa respectively. Four-species axes
mean dry/vapor/liquid/ice; four-trace axes mean entropy/vapor/liquid/ice.

Carrier/species **amounts are Pa m², not kg**; divide by gravity to obtain kg.
Carrier rates are Pa m²/s; timestep-integrated transfers are Pa m².
Antisymmetric face divergence conserves global species amounts. A common beta
preserves the sum-to-carrier relation. The dry ledger column and physical dry
amount can differ by the explicitly accounted carrier-endpoint residual.

The Gibbs and limiter work pairings address instantaneous total energy,
including the existing top-boundary reservoir. They do not make finite-step
SSPRK3 energy exactly conserved. Existing dry/vapor transfer lower bounds are
unchanged; they are not input clipping or a new state floor. Vapor must already
be strictly positive, and exact-zero-water dry equivalence is not established.

## Earlier bounded-wrapper readability validation

This PR adds source explanations and renames locals only. No helper extraction,
method reordering, cached/shared intermediates or algebraic simplification is
proposed. In particular, deduplicating repeated graph/gradient evaluations or
reconstructing Euler water from the RHS would need separate review for operation
order, tensor/gradient ownership and roundoff; neither is part of this change.

Executable ASTs of all four edited modules match the base after removing
docstrings and reversing the local-name map. Expressions, calls, branch order,
defaults, dictionary keys, state conventions and ledger operations are preserved.
Fixtures and the portable/exact policies are unchanged.

Local validation: Apple M3, macOS 14.6.1 arm64, Python 3.12.13, torch 2.15.0
with Accelerate, NumPy 2.5.4, SciPy 1.18.1, one thread. A freshly staged wheel
was installed into site-packages and checked outside the checkout with `-I`:
**527 portable tests passed**; standalone comparisons passed **142 moist/control
and 193 dry tensors**; both examples completed 300 simulated seconds. A separate
same-environment capture before and after this edit matched **all 335 tensors
exactly** using the original exact comparators. Those temporary captures did
not replace or modify repository fixtures. This exact pre/post result is
separate from comparison to the historical cross-backend fixtures.

AST normalization confirmed the four edited modules preserve executable syntax
relative to `9f82a0c`; wheel Python files match the current source, and
`git diff --check` passed.

Actual Linux [push run 41003311864](https://github.com/kjhall01/vfs-a/actions/runs/41003311864)
at `4046f3b` passed **527 portable tests**, both standalone portable tools and
both examples. Runner: AMD EPYC 7763 / AVX2, Python 3.12.15, torch 2.15.0+cpu
with MKL, CPU float64. Separately, both exact fixture tests and both exact tools
failed against the historical fixtures; these remain explicit informational
failures, not portable passes. Full CPU/backend details and mode-specific reports
are in the run's `cpu-checks` artifact. Subsequent PR/final-head runs are linked
in [PR #4](https://github.com/kjhall01/vfs-a/pull/4). CPU float64 checks do not
qualify GPU, long-run or full backward behavior.

## Centered moist readability validation

This follow-up formats only `moist_conservative.py`, expands packed statements,
renames ambiguous locals and explains the existing execution/adapter boundary.
No new helper file, class, cache, registry or functionality is introduced.
Executable AST comparison with `850dd21`, excluding docstrings and reversing
local renames, preserves all expressions, calls, defaults, branches, in-place
updates and evaluation order. Fixtures and comparison policies remain unchanged.

Local validation used Apple M3 / macOS 14.6.1, Python 3.12.13, torch 2.15.0 /
Accelerate, CPU float64 and one thread. A fresh staging wheel was installed into
site-packages; tests/tools/examples ran outside the checkout with `-I`:

- **527 portable tests passed**, including both fixture tests and existing
  derivative/invariant checks; standalone tools passed **142 + 193 tensors**.
- Both examples completed 300 simulated seconds.
- Same-environment captures before/after matched all **335 fixture tensors**
  exactly, plus **33 additional tensors**: component tendencies (15), energy
  terms (6), energy autograd derivatives (4), a fixed RHS vector–Jacobian product
  (4) and combined RHS (4). Additional comparisons required finite values and
  zero absolute/relative tolerance. The fixed contraction is a connectivity
  check, not a physical objective or full backward-rollout qualification.
- Adapter wind/pressure/entropy tensor identities and shared geometry were
  checked in both captures. Temporary captures did not touch repository fixtures.
- Executable AST comparison, wheel/source byte checks, single-file formatter
  check and `git diff --check` passed. Black was temporary tooling only; no
  dependency or formatting-policy change was added to the package.

Actual Linux [push run 41005300530](https://github.com/kjhall01/vfs-a/actions/runs/41005300530)
at implementation commit `4058876` passed **527 portable tests**, both standalone
portable comparisons and both examples. It used AMD EPYC 9V45 / AVX512,
Python 3.12.15, torch 2.15.0+cpu / MKL and CPU float64. **Both exact fixture tests
and both exact standalone tools also passed on this runner**; that is not a
cross-platform guarantee. Full environment and separate mode reports are in its
`cpu-checks` artifact. Subsequent PR/final-head runs are linked in
[PR #5](https://github.com/kjhall01/vfs-a/pull/5). These are regression checks,
not a scientific verification campaign.
