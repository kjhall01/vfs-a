# Ocean energy-evaluation optimization backlog

Research candidates, not implemented changes or promised speedups. Preserve the
chosen Hamiltonian, residual tolerances, positivity/funding and entropy checks.
Measure each against the already-compiled baseline, including full-step effects.

1. Surface thermal updates: port the residual-controlled temperature-native
   increment solver into surface radiation and water/heat exchange. The current
   shared apply_heat uses six fixed iterations and repeatedly requests a full
   parcel potential/derivative tuple when it needs only the T derivative.
   Retain independent final energy acceptance. Surface mass/momentum changes
   remain part of the Hamiltonian budget.
2. Mixing work along T/S paths: integrate both buoyancy-potential and reference-
   enthalpy derivatives on a straight endpoint T/S path, reconstruct coefficients
   for the actual conservative enthalpy/salt fluxes, and keep the existing local
   allocation. No intermediate temperature inversions. Test one-point evaluation
   with residual-controlled refinement; only later consider stable analytic
   discrete differences. Avoid naive subtraction near depleted TKE reservoirs.
3. Compute only required quantities: energy-only parcel values, T-only heating
   derivatives, and T/S-only mixing conjugates (no height derivative). Prefer
   shared kernels with compiler-visible requested outputs over copied EOS
   formulas. Verify the compiler actually eliminates unused work before claiming
   savings. A value/derivative specialization is not a cheaper Hamiltonian.
4. Reuse endpoint thermal work: pass already computed psi/reference fields from
   accepted proposals through heating, budget checks and the next mixing substep.
   Geometry, T, S and velocity dependencies must be explicit. Retain final checks;
   do not cache stale fields or reuse an initial energy after state changes.
5. Hoist invariant reference-profile and quadrature geometry evaluations across
   mixing substeps, whose thickness/geometry are fixed. The current workspace
   recreates these per substep. Invalidate after geometry, device, dtype, layer
   layout or relevant model-parameter changes; no unbounded global cache.
6. Consider combining prescribed heat contributions into one solve at a clearly
   defined fixed-mass stage. Water/momentum exchange, dissipation and forcing
   order generally do not commute: this is a splitting-scheme change requiring
   new tests, not a mechanical removal of duplicated calls.

Priority: surface thermal helper integration and required-output specialization
are lower-risk implementation candidates; T/S path work is the next numerical
experiment. Endpoint reuse is incremental. Neither quadrature changes nor fewer
mixing substeps should be silently substituted as implementation speedups.

Recent profiling is noisy: one later compiled 1024x64 full-entropy step without
ALE attributed ~43% to mixing, ~34% to surface exchange/radiative heating and
~19% to dynamics. A separate temperature-mixing profile attributed ~45% of its
mixing time to energy-path work, including ~18% to intermediate inversions.
These are approximate individual profile shares, not precise speedup forecasts;
detailed instrumentation perturbed runtime and another sample had a dynamics
spike. Scalar-path inversion removal alone cannot eliminate all path-work cost.

Implementation checkpoint: the throughput-composition research branch implements
fast surface thermal helpers, fixed three-point T/S path work, an experimental
budget-triggered midpoint retry, and complete temperature-state composition with
adaptive ALE. These remain opt-in candidates. Required-output specialization is
used for the heating derivative; broader endpoint/reference reuse and a unified
supported public API are still future work. See the composition campaign for
actual timings and qualification; this list itself makes no performance claim.
