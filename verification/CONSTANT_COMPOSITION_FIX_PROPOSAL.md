# Numerical-fix proposal — not implemented

**Decision requested in a subsequent review:** replace only the condensate
partial-density construction in `gibbs_face` by
`logmean(rhoi,rhoj)[...,None] * .5*(qi[...,2:]+qj[...,2:])`, retaining gas
logarithmic means and recomputing the existing complete `(s,q,alpha)` tuple.
No implementation is activated by this investigation PR.

The [derivation and probes](CONSTANT_COMPOSITION_DERIVATION.md) establish an
algebraically viable coupled construction: symmetric, consistent, constant-q
preserving, nonnegative, and finite at both-zero/one-sided-zero condensates.
It retains the discrete Gibbs identity for unequal T,p. It is not an internal
mean of condensate partial densities and does not promise local-extremum
preservation. This tradeoff must be accepted explicitly before implementation.
A density-only mean is forced to the logarithmic mean; its zero-endpoint
extension instead has singular derivatives and zero one-sided cloud flux.

## Minimal implementation boundary

Change the condensate weights only, with a short scientific comment. The
existing normalization, entropy trace and specific-volume trace must all use
the resulting weights. Never splice new species fractions into old entropy or
alpha. Leave carrier construction, gas means, thermodynamics, lower bounds,
limiter, ledger installation, transpose work pairing, heat allocation and
integrator formulas unchanged. Their outputs can change because the traces
change. Do not add a new runtime option or silently change the frozen baseline.

No additional matched-operator formula change has been identified by the source
audit: horizontal Gibbs entropy/water differences already enter the same
energy-gradient/carrier-transpose pairing; limiter work already uses the actual
trace change. This is a hypothesis to validate at model level, not permission
to waive those checks. `face_alpha` must remain coherent even though it is
currently diagnostic in the bounded path.

## Required regression and scientific checks before merging a fix

1. Reuse the face suite and independent high-precision endpoint potentials.
   Extend coverage across admissible density ratios, temperature/pressure jumps,
   exactly empty condensates, one-sided clouds, and the near-equal series boundary.
   Check symmetry, equal-state limits, gradients and the complete Gibbs identity;
   retain the species-only negative control.
2. Use actual production spherical states with multiple CP levels. For constant
   composition, check the high carrier/species flux relation, pressure-derived
   carrier residual, physical dry fraction, authoritative water ledger and every
   Euler stage. Include divergent prescribed velocities and reversed winds;
   exercise safe-high and active donor/limiter branches without resetting mass.
3. Check independent energy derivatives, instantaneous component work exchange,
   Gibbs correction and limited mechanical/heat allocation, including boundary
   energy terms. Distinguish instantaneous closure from finite-step energy drift.
   Do not infer an energy theorem solely from the local Gibbs identity.
4. Compare one complete bounded SSPRK3 step and smooth/active-limiter directional
   derivatives for wind, pressure, entropy and water. Verify zero-condensate
   one-sided derivatives without crossing the admissible domain or hiding limiter
   branches. Retain positive vapor and direct Euler species ledgers.
5. Repeat the small #17 constant controls, 16/64/256-cell return cases and dt
   halving. Record errors, bounds, inventories, mixing, limiter activity and
   reference uncertainty. The constant-q counterexample does not explain all
   deformational errors, and repairing it does not guarantee better convergence.
   No large campaign is needed for the initial numerical-fix review.
6. Run the installed-wheel suite, both comparison tools and supported examples.
   Dry and centered ARK4 paths should retain exact same-environment behavior
   because the bounded Gibbs routine is not used there; demonstrate that.
   Report changed bounded fixture values honestly. Preserve old fixtures and
   tolerances; any new reference set/baseline identity requires explicit approval
   and provenance. Do not relabel expected numerical differences as portable
   passes, or regenerate frozen fixtures merely to get green CI.

The numerical-fix PR must state its exact source change, approved reference
policy, CPU/backend environments, old/new results, and any remaining failures.
No GPU, full-rollout, IMEX-backward or general transport qualification follows
from the present face and toy-ledger evidence. The production implementation
stays frozen until that separate numerical change is reviewed and authorized.
