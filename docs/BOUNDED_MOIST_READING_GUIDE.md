# Reading one bounded moist step

Start with the call chain below, then follow the numbered blocks in `evaluate`.
Source docstrings give tensor shapes, units and staggering at each boundary.
No new algorithm or scientific verification campaign accompanies this guide.

1. [MoistAtmosphere.step](../vfs/atmosphere/model.py#L166) checks geometry and
   positive finite `dt` (seconds), then delegates to
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

## Change boundary and validation

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
`git diff --check` passed. GitHub Actions has not yet run this commit; actual
hosted results will be linked in the PR. CPU float64 checks do not qualify GPU,
long-run or full backward behavior.
