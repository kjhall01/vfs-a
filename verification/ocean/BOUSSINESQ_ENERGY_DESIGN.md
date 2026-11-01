# Active stratification: energy/content pairing design, not runtime adoption

Dependent on drafts #28 (balance limitations) and #29 (transport admissibility).
Production baseline remains `e959b7005832a5041e082b6e300838e15bfbaa9e`.
Probe/derivation commit: `48336c2425c73b0333511740d7df0c481cd7c30e`.
No selectable model, production tendency, timestep, default or fixture changes.
The proposed force is evaluated only in an independent design probe; no active
simulation or wave/rollout qualification is claimed.

## Minimal physical scope

Use hydrostatic Boussinesq inertia with constant rho0, fixed bathymetry H>0,
a free surface eta, and material layers ordered **top to bottom**. Each layer
has piecewise-constant idealized temperature T and salinity S. They are materially
advected in the inviscid limit; T is a conserved temperature-like scalar measured
in Celsius, not a new seawater enthalpy or in-situ/potential-temperature conversion.
Density anomalies enter hydrostatic pressure, not continuity or inertial mass:

```text
rho' = aT T + aS S + a0
     = rho0 [-alpha (T-T0) + beta (S-S0)]
aT = -rho0 alpha; aS = rho0 beta; a0 = -aT T0-aS S0.
```

The initial design uses existing constants rho0=1025 kg/m³, g=9.80665 m/s²,
alpha=2e-4 K^-1, beta=7.6e-4 per model salinity unit, T0=10 degC and S0=35.
This linear EOS is pressure-independent and restricted to small density anomalies.
It is not JMD95, nonlinear thermodynamics, cabbeling, thermobaricity, convection,
vertical mixing, heat-flux coupling, or a thermodynamic total-energy claim.

Prognostic arrays remain u_e [edge,layer] in m/s and h, Theta=hT, Sigma=hS
[cell,layer] in m, m degC and m salinity-unit. Passive hq is unchanged. Geometry
and fixed column configuration remain model-owned. No cached density or physical
fields, additional Field/state abstractions, or atmospheric column dependency
are needed. Layers have no interlayer mass transfer in this first design.
Closed global geometry, no surface pressure work, and impermeable fixed bottom
are assumed. Coast, remapping, mixing and forcing require separate budget work.

The continuum assumptions are standard hydrostatic/Boussinesq ones; see the
[FESOM equations](https://fesom2.readthedocs.io/en/latest/main_equations.html).
The need to pair buoyancy and pressure work is also explicit in
[MITgcm's hydrostatic discretization](https://mitgcm.org/public/r2_manual/latest/online_documents/node51.html).
The discrete expressions below are the proposed VFS derivation, not a claim to
reproduce either model's discretization.

## Potential energy and fixed-content derivatives

Choose z upward from a fixed datum. For layer k,

```text
eta = sum_k h_k-H
zbar_k = -H + sum_{j>k} h_j + h_k/2
m'_k = rho'_k h_k = a0 h_k + aT Theta_k + aS Sigma_k       [kg/m²]
E_b = g sum_cells A_c sum_k m'_ck zbar_ck                [J]
H_active = H_constant_density + E_b.
```

`E_b` is the exact vertical integral of g rho' z for the assumed piecewise-constant
layers. The existing rho0 free-surface energy omits the fixed bottom constant
`-rho0 g A_c H²/2`; adding it does not affect tendencies. Anomaly energy, and
therefore the shifted total, may be negative without indicating invalid density.
It is mechanical/gravitational energy, not heat content `rho0 cp sum A hT`.

Let `e_b=E_b/A_c` within a column. At fixed **stored Theta and Sigma**:

```text
d e_b/d Theta_k = g aT zbar_k
d e_b/d Sigma_k = g aS zbar_k
d e_b/d h_k = g [a0 zbar_k + m'_k/2 + sum_{j<k} m'_j].
```

Increasing h_k raises its own midpoint by dh_k/2 and all overlying midpoints
by dh_k; deeper midpoints stay fixed. Holding T/S instead of their contents
would incorrectly replace a0 by rho'_k in the first thickness term. The probe
checks these derivatives against an independently written vertical integral,
autograd, and central directional differences. Production forces use none of
those differentiated calculations.

Define b_h, b_T, b_S as the Euclidean derivatives of E_b divided by rho0 A_c.
Their units are m²/s², m²/(s² degC), and m²/(s² salinity-unit). They live at cells,
not edges. Applying G produces the corresponding per-length gradient. The
existing Bernoulli kinetic/free-surface contributions remain in the base model.

## Matching the actual content transport

With unchanged `F=I(h)u`, continuity is `h_dot=-D F`. The production donor
traces give `Theta_dot=-D(F T_up)` and `Sigma_dot=-D(F S_up)`. Grad/div
adjointness implies the extra scalar energy work

```text
W_b = rho0 sum_edges A_e F [G b_h + T_up G b_T + S_up G b_S].
```

The existing velocity derivative is `H_u=rho0 A_e F`, so the proposed **extra**
acceleration is

```text
u_dot_extra = -G b_h - T_up G b_T - S_up G b_S.
```

This uses the same stencil locations and **actual donor concentrations** as the
content fluxes. Using a centered thermodynamic trace while retaining donor
content transport does not satisfy this cancellation argument. Adding the
archived baroclinic pressure also has no such derivation. Rotation retains the
base metric pairing; its accuracy limitations from #28 remain fully applicable.
No Hamiltonian bracket, PV consistency, enstrophy property, or Jacobi identity
is established by this scalar-work identity.

A continuum one-layer check gives extra pressure acceleration
`-(g/rho0)[rho' grad(eta) + h grad(rho')/2]`. Together with `-g grad eta`, this
is the depth-mean hydrostatic pressure force for a layer of vertically uniform
density anomaly with Boussinesq inertia. This check supports the interpretation;
it does not prove that the proposed discrete pressure stencil is accurate.

When alpha=beta=0, aT=aS=a0=0 and E_b, all extra derivatives and the extra force
are exactly zero. The existing constant-density energy, fluxes, rotation and
momentum are recovered, without importing the archived active-EOS operator.
This is explicitly tested at the array level. Donor transport may mix density;
forcing mechanical cancellation can route that numerical mixing work into
momentum. Its accuracy and physical interpretation require investigation before
adoption, rather than being assumed from the algebra.

## Small probe evidence and limitations

Use the fixed 32-cell ocean fixture geometry, three positive layers, archived
nonzero velocities/thicknesses and explicit smooth active T/S perturbations.
No time integration is performed. The independent AD work contraction gives:

| Measurement | Result |
|---|---:|
| Maximum analytic/AD derivative relative error | 3.8841e-16 |
| Matched proposed work | -0.3359375 W |
| Sum of absolute local work terms | 2.00108e16 W |
| Cancellation-normalized matched work | -1.67878e-17 |
| Work from grafting archived pressure onto proposed energy | -1.25025e12 W |
| Normalized graft work | -6.24744e-5 |

The graft result is a controlled incompatibility example for this functional,
not a general claim about the archived solver's intended energy. The original
reference solver is untouched. Best relative central-difference errors over the
reported epsilon sweep are 2.01e-12 (h), 2.23e-11 (Theta), 4.73e-13 (Sigma).
The entire sweep is retained, including subtraction cancellation at smaller
increments. Independent tests also deliberately omit an active pairing term
and detect the resulting work defect.

Two rest constructions have eta=0, u=0, horizontal upper interfaces at depths
500 and 1200 m, T=[15,10,5] degC and S=35 per layer. The flat bottom is 3000 m;
the second bottom varies as 3000+200 x m within the lowest homogeneous-density
layer. Proposed acceleration RMS is **8.40e-21** and **1.43e-20 m/s²**. These are
simple discontinuous-layer rest constructions, **not** a qualification of a
smooth stratified profile crossing sloping coordinates, outcropping, arbitrary
bathymetry, or general hydrostatic pressure accuracy.

[Small results and provenance](boussinesq_design_results) retain the derivative
sweeps, rest metrics, incompatible graft and constant-density limit. Every
uploaded artifact is below 128 KiB. Detailed environment manifests/source
snapshots are omitted with hashes, without splitting or re-encoding large files.
The original scientific evidence and fixtures are unchanged.

Local installed unchanged wheel: **633 passed, one CUDA skip**. Six new design
tests cover the independent column integral, all three fixed-content derivatives,
central differences, matched work, deliberate corruption and exact zero-anomaly
limit. These are algebraic/design checks. No active-model finite-step energy,
internal-wave, rotating/baroclinic, remap, mixing, GPU, reduced-precision or
backward-rollout claim follows. Existing dry/moist atmospheric checks remain
untouched and pass through the full portable suite.

```sh
python -I tools/probe_ocean_boussinesq_energy.py --repo /path/to/repo \
  --output /new/nonexistent/boussinesq-design-run
python -I -m pytest tests/ocean/test_boussinesq_energy_design.py -q
python -I -m pytest tests --fixture-mode=portable -q
```

## Review gates before an active runtime model

1. Review the linear EOS, fixed-content energy and actual donor-force pairing;
   add independent manufactured hydrostatic pressure accuracy tests. Decide how
   numerical density mixing should exchange mechanical energy. Do not append
   the archived baroclinic pressure or silently switch content traces.
2. Implement a separately selected candidate only after that review, preserving
   the archived default. Add the per-Euler admissibility/acceptance policy from
   #29, with a separate gravity-wave stability assessment. Do not add floors.
3. Establish the constant-density limit, then flat and topographic rest for
   smooth stratification and layer-interface geometries. Measure pressure-gradient
   errors independently of energy cancellation.
4. Derive the continuous linear two-layer free-surface wave matrix and compare
   external/internal modes, resolving time, mesh and vertical errors separately.
   No internal-wave reference is asserted by the present work contraction.
5. Address the shared rotating reconstruction limitation before interpreting
   balanced/baroclinic accuracy. Any shared change needs atmospheric regression
   evidence in a separate PR; it must not enter through this design artifact.
6. Add remap, mixing or forcing only with explicit before/after H, momentum,
   volume and content budgets. Leaving velocity unchanged under h remap is not
   an established energetic algorithm. Long runs follow these checks, not precede them.
