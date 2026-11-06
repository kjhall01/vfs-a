# Ocean thermal work: first-law closure and second-law audit

Dependent research follow-up to bounded transport #33, based on
`ae8d1a1348fd154bc5a9c0fcda8b296eeac95664`. All installed runtime files,
defaults, fixtures and tolerances are unchanged. New implementation lives in
`tools/probe_ocean_thermal_work.py`, with tests and reproducible measurements.
It does not import atmosphere, replace the EOS, or integrate ALE #32.

**Result:** signed thermal conversion can close the bounded scheme's
instantaneous total-energy work without a zero-velocity force jump. However,
equal-endpoint thermal conversion fails an entropy test on tilted layers.
A separate thermodynamic exchange reference conserves energy and salt and has
nonnegative entropy production at the RHS level. These are distinct experiments;
the latter is not silently added to conceal the former's entropy deficit.

## 1. What JMD95 specifies, and what is missing

The retained density is JMD95(S, potential temperature, fixed-datum pressure).
Its use of **potential**, rather than in-situ, temperature agrees with the
[MITgcm JMD95 documentation](https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html).
The stored `temp_h` remains h times potential temperature in Celsius.

The existing energy is kinetic plus free-surface energy plus a quadrature-defined
dynamic-enthalpy anomaly Psi. Its vertical derivative determines the model's
buoyancy. That relationship leaves a reference caloric function undetermined.
A complete thermodynamic potential needs reference heat/entropy/composition
information as well as density. The general Boussinesq enthalpy construction is
discussed in [Tailleux, section 2.3](https://arxiv.org/abs/1011.5071).
[Young (2010)](https://doi.org/10.1175/2009JPO4294.1) explains the separation of
potential and dynamic enthalpy, using Conservative Temperature. We do **not**
rename the existing potential-temperature variable Conservative Temperature.

The equations and experiments below are a declared reduced-fluid completion,
not a fit to seawater calorimetry and not a claim that JMD95 determines cp.
The existing density-anomaly/quadrature potential is retained, rather than
silently substituting another paper's buoyancy or pressure convention.

## 2. Explicit caloric completion

Let theta be potential temperature in Celsius, tau=theta+273.15 its absolute
value, and S practical salinity. For the experiment choose

\[
H_c=H_{JMD95}+\rho_0 c_p\sum_i A_i(h\theta)_i,
\qquad
\eta(\theta,S)=c_p\ln(\tau/283.15)-k S\ln(S/35).
\]

The measured examples use cp=4000 J/(kg K) and k=0.3 J/(kg K salinity-unit).
Both are **required constructor arguments**, not calibrated defaults hidden
behind the EOS. The salt entropy term is idealized; S log(S/35) has value zero
at S=0. Its derivative is singular there. Finite entropy can be measured at
zero salinity, but the instantaneous chemical/entropy-gradient audit requires
strictly positive S. No salinity floor is introduced.

This defines a thermodynamic potential through the invertible relation

\[
\tau=283.15\exp\{[\eta+k S\ln(S/35)]/c_p\},\qquad
e_B(\eta,S,z)=c_p\theta(\eta,S)+\Psi(\theta(\eta,S),S,z).
\]

Here e_B denotes the reduced energetic potential; it is not asserted to equal
empirical seawater internal energy. Its conjugates at fixed geometry are

\[
T_B=\tau(1+\Psi_\theta/c_p),\qquad
\mu_B=\Psi_S+T_B k[\ln(S/35)+1].
\]

Thus T_B is the thermodynamic temperature **of the declared reduced fluid**;
it is not simply theta+273.15, nor an independently validated seawater in-situ
temperature. Layer means use the same quadrature-defined energy derivatives.
Independent tests differentiate this potential to check temperature, chemical
potential, the vertical derivative and Maxwell symmetry.

The added energy is linear in stored temperature content. Its thermal covector
is spatially constant on wet nodes, and its face jump is exactly zero. Therefore
the existing centered Hamiltonian pressure force remains the matching adiabatic
force. This cancellation is evaluated analytically, avoiding differences of
large cp terms. A general nonlinear empirical reference caloric function would
require corresponding discrete thermodynamic face identities; it is not a
drop-in constant-cp substitution.

## 3. Closing limiter work through the thermal state

The #33 ledger supplies the actual limited scalar transfers. With the existing
centered pressure, face work is

\[
W_e=\sum_{a\in\{\theta,S\}}[C_a]_e
\frac{J_{a,limited}-J_{a,centered}}{\Delta t},
\qquad C_a=\frac{1}{A}\frac{\partial H_{JMD95}}{\partial(ha)}.
\]

Let a_i=partial H_c/partial(h theta)_i. Allocate power -W_e/2 to each face
endpoint, sum all incident powers P_i, and set

\[
\dot{(h\theta)}_{thermal,i}=P_i/a_i.
\]

Then sum_i a_i dot(h theta)_thermal,i = -sum_e W_e. The full instantaneous
energy contraction closes to roundoff. Equal endpoint power is an explicit
allocation convention, not a uniquely derived physical law. Only positive
finite thermal-energy derivatives are accepted.

No extra energy reservoir is invented: the correction changes the existing
active thermal state, which feeds JMD95 density and subsequent forces. No
division by velocity or mechanical force correction is used. Since W is
proportional to the carrier flux, the source vanishes as velocity vanishes.

The sign matters. Some salinity mixing cases have **positive** limiter work,
requiring thermal cooling under this construction. Taking only the positive
part of a heating source would leave energy unclosed. Calling every correction
"dissipative heating" would be incorrect.

The transport-only ledger still conserves T/S/passive contents and bounds their
advective endpoint. The thermal source deliberately changes total potential-
temperature content; its separate source budget is reported for each Euler
stage and for SSPRK3 weights (1/6, 1/6, 2/3). Full temperature bounds need not
equal advective bounds once this source is included. Volume, salinity and
passive inventories remain conservative. The three full Euler endpoints retain
the authoritative salinity/passive ledgers; nonpositive absolute temperature
or invalid JMD95 states are rejected.

## 4. Why that construction is insufficient

Energy closure is a first-law statement. A closed dissipative system also needs
nondecreasing entropy. We audit both the nodal entropy-gradient/RHS contraction
and a separate face expression, with series forms near equal concentrations
to avoid cancellation of large global advective terms.

Two counterexamples are retained as tests:

1. **Salt-only mixing at uniform potential temperature.** Limiter work is
   positive, so local compensation cools the water. Without a composition
   entropy term, total entropy decreases. Including the declared k=0.3 term
   makes this example's total entropy production positive. Consequently the
   caloric/chemical closure cannot be inferred from density alone.
2. **Tilted layers with almost uniform potential temperature.** The layer
   thickness varies by 1%, theta by only 0.001 K times the horizontal coordinate,
   and S is uniform at 35. Converting limiter work to equal-endpoint thermal
   exchange decreases total entropy even with k=0.3. Potential-temperature
   gradients are not interchangeable with thermodynamic-temperature gradients
   across different parcel heights. The stable face audit resolves this deficit
   without cancellation against the large background entropy transport.

For the global tilted case, the measured entropy rates are -4.76249e4 W/K
(donor) and -2.22587e2 W/K (FCT). The corresponding absolute sums in the stable
audit are 1.42980e5 and 4.20095e2 W/K. These deficits are well resolved; they
are not tiny residuals of the 3.4e15 W/K gross nodal transport contraction.
Autograd of entropy independently agrees with the face audit to roundoff.

This falsifies **this thermal allocation**, not every possible completion using
JMD95. It is why the research step is not promoted to an Ocean runtime stepper.

## 5. A separate exchange operator with both spatial laws

At fixed layer geometry, use actual energetic and chemical conjugates to
construct an irreversible reference. On each oriented active face i to j,

\[
J_E=K(T_{B,i}-T_{B,j}),\qquad
J_S=-M[\mu_B/T_B],\qquad K,M\ge0.
\]

J_E is energy transfer in W; J_S is reference-mass-weighted salt-content transfer
in kg salinity-units/s. K is W/K and M is kg^2 salinity-units^2 K/(J s), **per
edge/layer**; geometry factors would be included in calibrated conductances.
This is a diagonal Onsager reference in the energy/salt flux basis. Onsager
here means that transfers follow thermodynamic forces with nonnegative
coefficients, so their entropy-production products have a known sign.

Antisymmetric transfer conserves energy and salt. Its face entropy production is

\[
\sigma_e=J_E[1/T_B]-J_S[\mu_B/T_B]
=K\frac{(T_{B,i}-T_{B,j})^2}{T_{B,i}T_{B,j}}
+M[\mu_B/T_B]^2\ge0.
\]

The salt flux uses an all-incident-face outflow budget reserving 10% of each
donor's positive salt amount. Multiplying each flux by a nonnegative donor
budget factor preserves the entropy sign. This provides a positive **salt
Euler endpoint** without claiming a maximum principle for cross-coupled
thermodynamic diffusion.

Convert the energy and salt divergences to stored potential temperature using

\[
\dot{(h\theta)}_i=
\frac{\dot E_i-H_{hS,i}\dot{(hS)}_i}{H_{h\theta,i}}.
\]

Both derivatives come from H_c. Omitting the salt contribution would break
energy closure. Tests independently differentiate energy and entropy, check
their contractions against face transfers, and stress the salt budget on global
and closed-coast meshes. Momentum and layer geometry remain unchanged.

This operator is an isolated reference, **not a replacement for the advective
limiter and not a repair obtained by tuning diffusion until its positive entropy
hides another operator's deficit**. It assumes strictly positive salinity.
The empirical transport matrix, heat/salt cross terms, energy-reference
transformations and seawater caloric calibration remain unspecified. A finite
Euler step is not an exact nonlinear energy/entropy map: the deliberately large
salt-budget stress runs can decrease finite-step entropy despite their positive
instantaneous production. Those changes are reported. No claim about finite-step
entropy follows from the spatial proof.
The largest artificial stress endpoint reaches S=147.37; this exercises the
salt-outflow algebra and is not a physically valid seawater trajectory or an
EOS accuracy test. The reference is evaluated at the original admissible state.

## 6. Measurements and verification

The compact [report](thermal-work-results.json) records explicit assumptions,
installed-runtime/driver hashes, 16 active thermal-conversion cases, entropy
counterexamples, velocity reversal, an amplified-velocity timestep sweep, and
eight isolated exchange cases. CPU float64, torch 2.15.0+cpu. Scientific results
are measurements, without accuracy acceptance cutoffs.

Across the active cases, maximum absolute energy contraction divided by the
sum of absolute local work is 5.50e-17. At velocities +/-1e-12 m/s, the centered
force jump is 2.41e-16 m/s2 and the FCT thermal source magnitude is 3.69e-22 K/s.
Both correction source and force jump are exactly zero in the zero-speed probe.
These are spatial work and continuity results, not proof of global smoothness.
Volume/salinity/passive inventory relative errors are at most 3.06e-16 in the
16 one-step active cases. The separate exchange reference has maximum normalized
energy contraction 1.84e-17 across the eight reported cases and nonnegative
face entropy production, including cases where the salt budget is active.

For 640 seconds with initial velocities amplified 100 times to resolve temporal
error above the thermal-background roundoff, measured total energy changes are:

| dt (s) | Dynamic change (J/m2) | Reference thermal change (J/m2) | Sum (J/m2) |
| ---: | ---: | ---: | ---: |
| 160 | -213.54247 | 213.46594 | -7.654e-2 |
| 80 | -212.56259 | 212.55301 | -9.580e-3 |
| 40 | -212.06469 | 212.06341 | -1.279e-3 |
| 20 | -211.80847 | 211.80814 | -3.277e-4 |

The first refinements are approximately third order in this energy residual;
the final refinement loses that order. Tiny thermal increments and accumulated
inventory roundoff on the large stored background become relevant. The summed
change is computed from separate energy differences, not hidden by normalization
with the enormous reference thermal energy. This short synthetic sweep is not a
balanced-jet, wave-growth or climate-drift qualification.

Installed-wheel portable suite: **680 passed, one CUDA skip**, including 21 new
research tests. All 164 reviewed legacy-ocean capture tensors are exactly equal;
corrected Gibbs and dry portable comparisons pass. Historical exact discrepancies
remain separately reported by existing CI. No fixture or tolerance changes.

Reproduce after building/installing the wheel and copying tests/tools outside
the checkout:

```sh
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/probe_ocean_thermal_work.py --output thermal-work.json
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

The research equations use analytic energy derivatives, never forward autograd.
Tests independently differentiate the potential, entropy and one thermal-source
directional derivative. Long backward runs, GPU, compilation, performance,
long active trajectories and real seawater caloric accuracy are unqualified.
No field binaries, wheels or full logs are uploaded; every new/changed file is
below 128 KiB.

## 7. Next implementation boundary

Keep this dependent PR and #33 unmerged. The next candidate should have:

1. A separately inspectable reference caloric/entropy object validated against
   seawater thermodynamic data, retaining JMD95 density explicitly. Choose the
   stored thermal variable deliberately; entropy, potential enthalpy and potential
   temperature are not interchangeable under irreversible mixing.
2. Matched thermodynamic face identities for the adiabatic part and a coupled
   heat/salt exchange driven by T_B and mu_B, with a nonnegative mobility and
   explicit stage admissibility. Treat tracer limiting as part of that design,
   not as a scalar update followed by an unconstrained energy patch.
3. Independent first-law, entropy, positivity and zero-speed tests on flat and
   tilted geometry, plus timestep convergence, before active runtime adoption.
   Do not claim finite-step entropy from a semidiscrete proof.

After those checks, compose with ALE using separate remap energy/entropy
accounts. The rotation/kinetic reconstruction issue remains a separate open
problem. This work narrows the physical design; it does not establish a complete
energy/entropy-consistent bounded ocean-atmosphere model.
