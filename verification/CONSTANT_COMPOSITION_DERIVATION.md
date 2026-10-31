# Constant composition: carrier consistency and Gibbs face freedom

Investigation only, following #17. No production source, defaults, fixtures or
baseline changed. Numerical baseline is `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`;
the approved stage-interface extraction remains `ae132773e621671f70bfa6a698eebb1a82cffa94`.
The [fix proposal](CONSTANT_COMPOSITION_FIX_PROPOSAL.md) is separate and unimplemented.

## Actual carrier and species condition

Cell full-layer pressure thickness is dp (Pa). CP interfaces use
`dm=(dp_0/2, (dp_0+dp_1)/2, ..., dp_last/2)`. At cell/interface node n,
carrier M_n=A_cell dm_n (Pa m²), water Q_na=A_cell water_content_na (Pa m²),
physical dry amount is M_n−sum_water Q_na. Divide these by g for kg.
Wind is on horizontal edges/full layers. Production layer-edge carrier flux is
`dv_edge * u_e * cell_to_edge(dp)` (Pa m²/s). Its CP horizontal flux uses the
same adjacent half weights. Vertical graph links carry `A_cell*omega` between
adjacent CP interfaces. Flux F is oriented i→j; incidence B subtracts at i and
adds at j. These are the actual `bounded_transport.graph` operations.

For a full positive Euler budget, G=dt F, carrier M'=M+BG, and species
Q'_a=Q_a+B J_a. Therefore constant fractions Q_na=q_a M_n are preserved iff

```
B(J_a - q_a G) = 0.
```

The facewise sufficient condition, necessary for arbitrary isolated-face fluxes,
is J_fa=q_a G_f. Cycle-flux cancellation can satisfy the node condition on a
particular mesh without satisfying this stronger local condition. A discretely
divergent velocity does NOT itself destroy constant composition if carrier and
species use consistent fluxes. When the actual pressure-derived target mass is
M'=M+BG+r, the composition error numerator is `B(J_a-q_a G)-q_a*r`;
carrier/target consistency must be checked as well, including rounding.

`BoundedGibbs.evaluate` constructs horizontal high species traces via
arithmetic endpoint fractions plus `gibbs_face` minus arithmetic fractions;
vertical corrections are zero. `limited_traces` multiplies these by G, and
makes dry flux complementary to the three water fluxes. Its donor reference
also preserves constant q for consistent carrier endpoints. If high and donor
fluxes both equal qG, any common limiter blending preserves that property in
exact arithmetic. Euler installs water from the returned amount ledger/area.
SSPRK3 convex combinations of carrier and species amounts then preserve it too;
combining fractions instead would not be justified. Machine precision, lower
bounds at their margins, and inexact target masses still need numerical checks.

## Where the frozen face reconstruction fails

Write partial densities d_a=rho q_a, with rho=p/(R(q)T). Let L be the positive
logarithmic mean, A the arithmetic mean. Frozen gas means are L(d_ai,d_aj),
and condensate means A(d_ci,d_cj). For identical endpoint composition:

```
d_g* = q_g L(rho_i,rho_j)
d_c* = q_c A(rho_i,rho_j)
D*   = L(rho_i,rho_j) + sum_c(q_c)*(A(rho_i,rho_j)-L(rho_i,rho_j))
q_a* = d_a*/D*
```

With nonzero condensate and unequal total density, A>L, so gas fractions are
reduced and condensate fractions increased at the face. This violates the
facewise condition before limiting. Equal density, or no condensate, removes
this particular defect. Temperature and pressure jumps both matter through rho.
No tolerance-based test for constant composition can repair the general formula.

## Discrete Gibbs identity: what is free

Use species constants cpa, Ra (zero for condensates), h_ref,a and T_ref.
Set k_a=h_ref,a−cpa*T_ref, cva=cpa−Ra, and eta_a=mu_a/T. For positive gas
densities and T, the ideal-mixture potentials implemented by VFS satisfy

```
Delta eta_a = -k_a Delta T/(Ti*Tj)
              -cva Delta T/L(Ti,Tj) + Ra Delta log(d_a).
Delta mu_a = A(T) Delta eta_a + A(eta_a) Delta T.
```

The last term is absent for condensates, including at zero condensate. Choose
any finite nonnegative condensate weights d_c*, retaining d_g*=L(d_gi,d_gj).
Let D*=sum_a d_a*>0, q_a*=d_a*/D*, alpha*=1/D*. Gas logarithmic means give
`d_g* Delta log(d_g)=Delta d_g`. Since p=T sum_g Ra d_g,

```
Delta p = A(T) sum_g Ra Delta d_g + A(sum_g Ra d_g) Delta T.
s* = A(T)*cv(q*)/L(Ti,Tj) + A(T)*k(q*)/(Ti*Tj)
     - sum_a q_a* A(eta_a) + A(sum_g Ra d_g)/D*.
```

Substitution proves `sum_a q_a* Delta mu_a + s* Delta T = alpha* Delta p`,
equivalently `Delta mu_d + sum_water q_a* Delta(mu_a-mu_d) + s* Delta T` on
the left. This is an algebraic identity for arbitrary unequal T,p,composition;
no division by Delta T or Delta p is required. The proof uses exact logarithmic
means. The production series/quotient evaluation and floating arithmetic have
roundoff/truncation errors measured by probes, not erased by this derivation.

## Simplest candidate and its limits

Retain gas means and choose, for each condensate,

```
d_c* = L(rho_i,rho_j) * A(q_ci,q_cj).
```

This is symmetric and consistent. For constant composition all species weights
are q_a L(rho_i,rho_j), so D*=L(rho_i,rho_j), q*=q. Both endpoints zero gives
exactly zero; one endpoint zero gives a finite positive weight if the other is
positive. Only strictly positive total/gas densities enter logarithms. There is
no floor, clipping, composition equality detector or division by condensate.
It is smooth in condensate fractions up to their admissible zero boundary,
provided dry/vapor and T,p stay positive. Numerical face derivative probes below
cover compositions, not timestep backward or arbitrary machine extremes.

**This is a coupled endpoint construction, not a mean of condensate density
alone.** It need not lie between endpoint condensate partial densities. The
anti-correlated density/cloud probe exceeds that interval by 0.05114 kg/m³.
Normalized species fractions stay nonnegative and sum to one, but local
endpoint extrema are not guaranteed. Production FCT constrains amount bounds,
not local extrema; shape/mixing consequences need separate testing.

If d_c* must instead depend ONLY on its two partial densities, constant
composition for arbitrary rho_i,rho_j and positive q_c forces
`d_c*(a,b)=L(a,b)` by logarithmic-mean homogeneity. Its continuous extension has
L(0,b)=0 for b>0 and L(0,0)=0. The derivative as a→0+ with fixed b diverges,
and the high condensate flux into an exactly empty endpoint vanishes. An
initial interface with zero high flux need not activate the donor limiter to
move that cloud front. Thus that density-only choice cannot also supply a smooth
nonzero one-sided-cloud trace. The coupled candidate avoids this incompatibility;
it does not assert uniqueness among coupled constructions.

## Matched operators: never change q alone

Recompute the FULL existing `(s*, q*, alpha*)` formula from the new weights.
Changing q while retaining old s and alpha breaks the identity (negative-control
residual up to 6.24e4 J/kg). Endpoint chemical potentials and thermodynamic
constants stay unchanged. `BoundedGibbs._gibbs_corrections` forms both entropy
and water trace differences and applies their energy-covector jumps through the
negative carrier transpose. The limiter's entropy correction and mechanical/
heat allocation then use the actual new traces/limited transfer. Their formulas
need no identified algebraic change, but their numerical values will change.
`face_alpha` is currently returned as diagnostics; it must still be recomputed.
This audit does not replace a full atmospheric work/energy/gradient regression.

## Executable evidence (CPU float64 only)

From outside the checkout, with the unchanged installed #16/#17 wheel:

```sh
python -I /path/to/vfs-a/tools/investigate_condensate_means.py \
  --repo /path/to/vfs-a --output new-condensate-probe
```

114 faces: equal states, pressure/temperature/both jumps, equal density,
varying composition, both/one-sided zero condensates, anti-correlated density,
nearly equal states (relative perturbations 1e-3 to 1e-12), and 100 seeded
varying cases. Endpoint chemical potentials are independently computed with
60-digit Decimal arithmetic, not the production chemical helper. Trace values
remain float64. A toy two-cell × three-CP-interface graph invokes production
`limited_traces`/`bounded_transfer`; independent face sums audit its ledger.
This is not a candidate inserted into a spherical atmospheric step.

Measured candidate maximum identity residual: 1.8345e-10 J/kg, maximum residual
relative to summed absolute work terms 1.5037e-14. Frozen means: 2.1146e-10 J/kg
and 1.4074e-14. Candidate constant trace and post-Euler fraction discrepancies
are at most 1.1103e-16; symmetry differences zero. Zero-condensate faces remain
exactly zero; the one-sided-cloud case activates three limited faces without
negative amounts. Five focused tests pass, including equal-state thermodynamic
consistency, a species-only negative control and finite one-sided composition
AD/FD probes at zero condensate. These are algebra/implementation checks, not
scientific qualification thresholds or evidence about all deformational errors.

[Recorded summary and artifact hashes](results/condensate_means_investigation.json)
identify the unchanged runtime and exact probe source. No coupled candidate run,
GPU run, full-rollout derivative or large experiment has been performed.

Local installed-wheel validation outside the checkout: **568 passed, one CUDA
skip**; both standalone portable fixture tools passed. PyTorch 2.15.0, macOS
CPU float64. These checks exercise the unchanged production wheel; they do not
activate the candidate. Hosted outcomes are recorded separately in the PR.
