# Reading one bounded moist step

Start with the default call chain below, then follow the numbered blocks in `evaluate`.
The separate centered ARK4 choice are described in [the module API](MOIST_MODULE_API.md).
Source docstrings give tensor shapes, units and staggering at each boundary.
No new algorithm or scientific verification campaign accompanies this guide.

1. [MoistAtmosphere.step](../vfs/atmosphere/model.py) calls the module; `forward`
   checks geometry, dtype/device and positive finite `dt` (seconds). The default
   [BoundedSSPRK3.advance](../vfs/atmosphere/steppers.py) then delegates to
   [BoundedGibbs.step](../vfs/atmosphere/bounded_moist.py).
2. [step_with_info](../vfs/atmosphere/bounded_moist.py) performs three
   bounded Euler maps, each with the full `dt` budget. The second RK state is
   `3/4 initial + 1/4 Euler(Euler(initial))`; the final state is
   `1/3 initial + 2/3 Euler(second RK state)`. Combinations operate on contents,
   not fractions. The three returned ledgers describe the Euler stages.
3. [euler_with_info](../vfs/atmosphere/bounded_moist.py) calls
   [evaluate](../vfs/atmosphere/bounded_moist.py), advances the other fields,
   and **installs water from `info['amounts'] / cell_area`**. Keep this exact
   expression; `water + dt*rhs.water` is not a cancellation-safe replacement.

Inside each `evaluate`, in execution order:

1. [graph](../vfs/atmosphere/bounded_transport.py) constructs cell-interface
   carrier amounts and oriented horizontal/vertical transfers. Horizontal faces
   precede vertical faces; positive flux moves from node `i` to node `j`.
2. [_gibbs_corrections](../vfs/atmosphere/bounded_moist.py) calls
   [gibbs_face](../vfs/atmosphere/gibbs_means.py). Its entropy/fraction traces
   satisfy the analytic mixture Gibbs identity. Horizontal Gibbs-minus-arithmetic
   trace changes get a paired momentum correction through
   [pullback](../vfs/atmosphere/bounded_moist.py); vertical changes are zero.
3. [composition_entropy_derivatives](../vfs/atmosphere/gibbs_means.py)
   supplies fixed-temperature/pressure derivatives for changes replacing dry air.
   The base tendency then supplies continuity and the actual Euler carrier endpoint.
4. [limited_traces](../vfs/atmosphere/bounded_transport.py) integrates the
   proposed carrier/species fluxes over `dt`, makes dry transfer complementary to
   water, then calls [bounded_transfer](../vfs/atmosphere/bounded_transport.py).
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
   The retained configuration requires release_as_heat=True.
7. The assembled RHS retains matched momentum/continuity and replaces scalar
   tendencies with limited-flux divergence plus entropy heating. The water RHS
   describes transfer, but the Euler water update still uses the ledger directly.

## Inside the centered moist base

The bounded wrapper calls into `moist_conservative.py` for its base RHS and
energy gradients. Read these functions in execution order:

1. [tendency](../vfs/atmosphere/moist_conservative.py) calls
   [transport_state](../vfs/atmosphere/moist_conservative.py): validate composition,
   then borrow the same wind, pressure, entropy and geometry in
   [ScalarTransportState](../vfs/atmosphere/scalar_transport.py).
   `scalar_content is MoistState.entropy_content`; `specific_scalar()` divides
   by CP pressure mass. There is no mtheta/theta disguise, dry temperature API,
   copy, detach, implicit conversion or diagnostic cache.
2. [_assemble](../vfs/atmosphere/moist_conservative.py) obtains continuity
   primitives and centered horizontal entropy transport, then calls
   [mixture_gradients](../vfs/atmosphere/moist_conservative.py). This reconstructs
   two Gauss points per layer, forms species enthalpy/entropy/chemical potentials,
   and pulls temperature and water-minus-dry chemical potentials back to CP
   contents. Its pressure derivative includes quadrature motion, normalization
   at fixed contents and surface geopotential. Kinetic energy is excluded here.
3. `_assemble` pairs entropy and each water-species transport with momentum work
   through explicit transposes. It then assembles skew rotation, centered vertical
   velocity exchange with the changing mass metric, and Bernoulli work. Finally
   it solves for acceleration and completes entropy divergence/continuity.
   `tendency` solves the summed momentum covectors; `tendency_terms` solves each
   component separately. Both explicitly return entropy_content.
   These two arithmetic paths are preserved, not merged for apparent simplicity.
4. [energy_terms](../vfs/atmosphere/moist_conservative.py) reconstructs the same
   pressure quadrature, integrates geopotential upward, and sums internal,
   potential, top-reservoir and kinetic energies (J). Interface geopotential is
   a separate `(cells, interfaces)` tensor in m²/s².
5. [energy_gradients](../vfs/atmosphere/moist_conservative.py) starts with
   `mixture_gradients`, adds the surface-pressure dependence of kinetic layer
   mass, then obtains the kinetic wind derivative. Outputs have state shapes,
   but units of **energy per stored variable**, not prognostic-state units.
   [boundary_energy_rate](../vfs/atmosphere/moist_conservative.py) contracts
   these derivatives with an RHS to return watts. The formulas are analytic;
   their tensor results remain connected for differentiation.

The source defines retained symbols `p`, `dp`, `dm`, `S`, `T`, `q` and `pg`,
with units/staggering at the function boundaries. Repeated access paths,
in-place pressure-gradient additions and the two momentum assembly paths retain
their operation order. Sharing more intermediates could change roundoff or
gradients and is outside this consolidation.

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

Current [regression evidence](READABILITY_RESULTS.md) includes exact same-
environment pre/post comparisons; [earlier readability measurements](HISTORY.md)
remain historical evidence. See [all four timestep paths](CURRENT_PATHS.md) for
conservative dry stepping and the distinct ARK4 split.
