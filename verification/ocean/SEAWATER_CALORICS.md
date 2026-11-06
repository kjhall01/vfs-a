# Surface seawater calorics and an enthalpy-coordinate research ocean

Parent: thermal-work draft #34, `81481b0b81dcd193666dde86db265b82b7c70d61`,
which depends on bounded-transport draft #33. ALE #32 remains separate.
This change is research code, tests and evidence. Installed `vfs/` files,
defaults, fixtures and tolerances are unchanged. Keep the drafts unmerged.

The constant-cp/ideal-mixture calorics in #34 were explicit assumptions. Here
one surface Gibbs potential is calibrated to GSW function evaluations, with
analytic tensor derivatives and an explicit salinity/temperature convention.
A separate research state transports **potential enthalpy**, recovering
potential temperature for the existing JMD95 density calculation. Its matched
energy gradients and momentum force close spatial work. This advances the
thermodynamic foundation; it does **not** solve bounded active advection's
second-law problem. The tilted-layer entropy counterexample survives.

## Units, reference and scope

* JMD95 still supplies density as a function of practical salinity SP,
  potential temperature theta and fixed-datum pressure. Its numerical
  polynomial, reference profile, density-anomaly integral and quadrature are
  unchanged. No atmosphere module is imported.
* This research adapter explicitly interprets theta on IPTS-68. The surface
  reference evaluates ITS-90 temperature `theta90=theta68/1.00024`. The surface
  class itself requires either scale explicitly; the JMD95 adapter requires
  IPTS-68. No existing runtime state is silently converted or relabelled.
* `SA=SR=(35.16504/35)*SP` in g/kg assumes reference-composition seawater. It is
  **not** the geographic Absolute Salinity conversion. A constant conversion
  factor preserves the conservative SP inventory; no salinity anomaly is
  diagnosed. Chemical derivatives below are per unit SP, including this factor.
* The numerical reference rectangle is `SP in [0,42]`,
  `theta90 in [-2.5,40] degC`, at zero sea pressure. Values outside it are
  rejected. This rectangle is not a claim about liquid-phase stability:
  freezing, sea ice and thermobaric validity limits are not implemented.
* Gibbs/enthalpy are J/kg; entropy and heat capacity are J/(kg K). Absolute
  temperature is theta90+273.15 K. Reference energy and entropy offsets follow
  GSW's convention, rather than independently choosing an offset in each field.

Primary references:

* [TEOS-10](https://www.teos-10.org/) defines a Gibbs-based seawater framework
  and distinguishes Absolute from Practical Salinity.
* [GSW Gibbs function](https://www.teos-10.org/pubs/gsw/v3_04/html/gsw_gibbs.html),
  [GSW-Python](https://teos-10.github.io/GSW-Python/conversions.html),
  [reference salinity](https://www.teos-10.org/pubs/gsw/html/gsw_SR_from_SP.html),
  and [temperature conversion](https://www.teos-10.org/pubs/gsw/html/gsw_t90_from_t68.html).
* [MITgcm JMD95 convention](https://wwwcvs.mitgcm.org/viewvc/MITgcm/utils/python/MITgcmutils/MITgcmutils/jmd95.py?view=markup)
  describes potential temperature on IPTS-68.
* [Young (2010), Seawater and the Boussinesq Approximation](https://www-pord.ucsd.edu/wryoung/reprintPDFs/SeawaterBoussinesq.pdf)
  and [Tailleux, thermodynamics of Boussinesq fluids](https://arxiv.org/abs/1011.5071)
  motivate declaring the caloric completion and distinguishing reduced-fluid
  thermodynamics from a full compressible seawater equation of state.

This is a **hybrid JMD95-density caloric completion**, not a full TEOS-10 model.
Near-exact agreement with surface GSW evaluations validates the implementation
against that software reference, not against independent ocean observations.

## One potential, then derivatives

`tools/fit_surface_gibbs.py` fits independently written basis functions to
1,600 `gsw.gibbs(0,0,0,SA,t,0)` evaluations using GSW-Python 3.6.23 and
NumPy 2.3.5. No GSW implementation is copied or modified. The committed
coefficient file records the fitter hash, versions, domain and fit condition.

The basis comprises a temperature polynomial of degree 8, SA times a tensor
polynomial of degree 6 in sqrt(SA/SAmax) and degree 8 in temperature, and one
`(t+273.15)*SA*log(SA/35.16504)` term. Chebyshev fitting is converted to powers
for analytic Horner evaluation. The column-scaled design condition number is
9443.73; the maximum training Gibbs residual is 4.82e-11 J/kg.

For the fitted potential G, define at the surface

\[
\eta=-G_t,\qquad a=h^0=G-T_0G_t,\qquad
c_p=-T_0G_{tt},\qquad T_0=t+273.15.
\]

These properties are **not** independently fitted. Temperature-scale and
salinity conversion derivatives are included analytically. In particular,
`a_theta/eta_theta=T0`. The logarithmic term cancels analytically from enthalpy
and its first salt derivative. Values and the first enthalpy derivatives are
finite at SP=0. The entropy/chemical salt derivative is singular there and is
explicitly rejected; no salt floor is introduced. Higher salinity derivatives
at zero are not claimed.

`SurfaceGibbs.temperature_from_enthalpy(SP,a)` uses twelve fixed, bracketed
analytic Newton iterations on the declared temperature interval. It rejects
out-of-range enthalpy or a failed numerical residual check. It does not clip
the stored content, detach the tensor graph or invoke forward autograd.

The independent validation set has 4,536 points: a held-out rectangular grid,
seeded uniform samples, and small positive salinities. Maximum absolute errors
against direct GSW evaluations are:

| Quantity | Maximum error | Units |
| --- | ---: | --- |
| Gibbs potential | 5.00e-11 | J/kg |
| Potential enthalpy | 2.06e-8 | J/kg |
| Entropy | 6.73e-11 | J/(kg K) |
| Heat capacity | 1.84e-8 | J/(kg K) |
| Enthalpy SP derivative | 1.68e-8 | J/(kg PSU) |
| Entropy SP derivative | 6.20e-11 | J/(kg K PSU) |
| Inverted theta68 | 2.13e-14 | degC |

The minimum sampled cp is 3944.5164 J/(kg K). These are measurements, not
accuracy gates or a proof over all possible states.

## Why transport potential enthalpy

With realistic calorics, `a(theta,SP)` is nonlinear. Adding `rho0*h*a` to the
old theta-content Hamiltonian while retaining its old centered pressure and
fluxes would lose the discrete work identity. It also would not make a
temperature inventory into a physical heat inventory.

The research `EnthalpyState` stores `u_e`, `h`, `enthalpy_h=h*a`,
`salt_h=h*SP` and passive contents. Temperature recovery is an explicit method.
The existing ocean-owned carrier/FCT primitive is reused through a named
`transport_view()`: only that adapter temporarily places enthalpy in the
primitive's first generic scalar slot. It is never sent to JMD95 or interpreted
as temperature. The installed OceanState contract remains unchanged.

Write Q=h*a, Sigma=h*S, Theta=h*theta, and m=rho0*A (masked on land). With
the previous dynamic Hamiltonian H_d, the new energy is

\[
H=H_d(u,h,\Theta(Q,\Sigma,h),\Sigma)+\sum mQ.
\]

If the old fixed-content covectors are g, the analytic transformed covectors are

\[
H_u=g_u,\quad H_Q=g_\Theta/a_\theta+m,\quad
H_\Sigma=g_\Sigma-g_\Theta a_S/a_\theta,
\]
\[
H_h=g_h+g_\Theta\left[\theta+(Sa_S-a)/a_\theta\right].
\]

The paired pressure uses centered traces of **a and S**:

\[
\dot u_{pressure}=-G(H_h/m)-\bar a\,G((H_Q-m)/m)
                         -\bar S\,G(H_\Sigma/m).
\]

The constant reference-enthalpy covector has zero face jump and is removed
analytically before taking differences. This avoids subtracting large caloric
terms. Rotation retains the existing kinetic-metric skew operator. There is
no division by velocity; the force remains continuous as velocity crosses zero.
Reference-composition rest over topography is independently checked.

Entropy uses the same potential: `E=sum(m*h*eta(a,S))`. Its covectors are

\[
E_Q=m/T_0,\quad E_\Sigma=m(\eta_S-a_S/T_0),\quad
E_h=m[\eta-a/T_0-S(\eta_S-a_S/T_0)].
\]

The reduced layer thermodynamic temperature and chemical potential satisfy
`H_Q=T_B*E_Q` and `H_Sigma=T_B*E_Sigma+m*mu_B`. These layer values use the
Hamiltonian's vertical quadrature. A parcel at height z instead has

\[
T_B=T_0(1+\Psi_\theta/a_\theta),\qquad
\mu_B=a_S+\Psi_S-T_B\eta_S.
\]

## Bounded transport and the remaining entropy failure

The authoritative Euler ledger conserves volume, transported Q, salt and
passives before the explicit thermal conversion. Q conservation does not
imply conservation of h*theta: nonlinear mixing changes temperature even
without an external heat source. Each full Euler endpoint recovers temperature
and checks density, layer thickness and the caloric domain before SSPRK3
convex combinations. There is no clipping, automatic retry or energy repair.

Relative to centered fluxes, the limiter's dynamic energy work W is computed
from face covector jumps and flux changes. As in #34, half of -W is allocated
to each endpoint, with `Q_source=power/H_Q`. This closes instantaneous total
energy work and explicitly changes the potential-enthalpy inventory. The
conversion is a signed thermal source, not guaranteed entropy production or
a claim that all numerical mixing is physical.

Across 16 front/salt-front/tilted/freshwater cases, with and without coasts,
donor and FCT, maximum absolute normalized spatial work is 4.88e-17. Salt,
volume and passive inventories close at floating-point roundoff, including
zero-salinity/passive states. Analytic energy and entropy covectors are also
compared with independent autograd derivatives of their scalar definitions.

For the closed-sphere tilted-column case, face-form entropy rates are:

| Transport | Advective contribution | Thermal conversion | Total, W/K |
| --- | ---: | ---: | ---: |
| Donor | +47,537.03 | -95,304.42 | **-47,767.39** |
| FCT | +98.51 | -321.35 | **-222.84** |

The coastal variants also have negative totals. Thus replacing the provisional
calorics does not remove the failure. The face calculation includes centered
transport entropy error; it does not attribute every contribution to the
limiter. Independent nodal entropy contractions agree to roundoff relative to
their much larger local terms. No additional diffusion is tuned to mask this.

At fixed duration 1,000 s, the FCT research composition has energy changes
-0.0487235, -0.00611630 and -0.000785642 J/m2 for dt=1000, 500 and 250 s,
respectively (successive ratios 7.97 and 7.79). This short example is consistent
with third-order temporal error; it is not exact finite-step conservation or
a long stratified-wave/balanced-jet qualification. Much smaller steps can reach
subtraction roundoff in the extensive caloric inventory.

## Separate thermodynamic exchange

The fixed-geometry exchange from #34 is rederived in Q coordinates, using the
same H and E as advection. With nonnegative declared K and M,
`J_energy=K*(Ti-Tj)`, `J_salt=-M*jump(mu/T)`. The complete donor salt outflow
budget supplies a positive face rescaling alpha. Its entropy production is

\[
K(T_i-T_j)^2/(T_iT_j)+\alpha M[\mu/T]^2\ge0.
\]

The Q tendency is `(energy_rate-H_Sigma*Sigma_rate)/H_Q`. This accounts for
salt/thermal coupling and closes energy and salt at the RHS level. The
independent entropy derivative agrees with the nonnegative face production.
K=1e9 W/K and M=1e14 or 1e24 kg2 PSU2 K/(J s) are diagnostic parameters,
not calibrated diffusivities. The extreme M case leaves salinity positive
(minimum 3.6) but reaches SP=89.92, outside the reference rectangle; its full
Euler state is rejected. Positive salt alone is not joint thermodynamic
admissibility. Neither this exchange nor its spatial entropy proof repairs
advective entropy or establishes a finite-step law.

## Comparison with GSW below the surface

For theta90={0,10,25}, SP={5,35,40}, compare the parcel formulas with GSW
in-situ temperature and Gibbs SP derivative at the same datum pressure.
`gsw.pt_from_t(SA,theta90,0,p)` follows the same isentropic parcel to p.

| Depth, m | Max temperature difference, K | Max chemical difference, J/(kg PSU) |
| ---: | ---: | ---: |
| 0 | 0 | 1.20e-11 |
| 500 | 0.00437 | 0.164 |
| 2000 | 0.01541 | 0.557 |
| 4000 | 0.02476 | 0.997 |

These differences are reported rather than hidden by changing JMD95. They
include the retained density EOS, density-anomaly Boussinesq convention,
temperature convention and quadrature. This experiment does not separate
those contributions or claim the same error throughout the ocean.

## Reproduction and next boundary

Build/install the wheel and copy `tests`, `tools`, and `examples` to a directory
outside the checkout. Use Python >=3.11 for the independent GSW 3.6.23 checks;
VFS's existing runtime Python requirement and dependencies are unchanged.

```sh
python -m pip install gsw==3.6.23
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/probe_ocean_seawater.py --output seawater-reference.json
python -I tools/fit_surface_gibbs.py --output regenerated-reference.json
```

The fitter is optional for ordinary use; the committed coefficients are used
by forward tests. The independent GSW measurement is a required hosted-CI step,
not an optional skipped reference test. Evidence is in `seawater-results.json`.

Local installed-wheel validation: **697 passed, one CUDA skip**, including
17 new research tests. All 164 reviewed legacy-ocean capture tensors are exactly
equal. Corrected Gibbs and dry atmospheric portable comparisons pass. All 100
runtime Python files are unchanged; historical fixture/tolerance policy is
unchanged. The new reference driver completed outside the source checkout.

The next numerical step is a joint face construction in enthalpy/salinity and
layer geometry that satisfies the entropy inequality **and** the Hamiltonian
work identity while retaining donor admissibility and a continuous rest force.
That needs a derived entropy-compatible reference flux and a constrained
antidiffusive budget, including full Euler thermodynamic bounds. Do not infer
that independent scalar FCT or arbitrary endpoint heating supplies those
properties. Only after that result should this become an active runtime
stepper and be composed with ALE and long stratified/balanced-flow tests.

GPU, performance, long rollouts, full backward qualification, freezing,
geographic salinity anomalies, a Poisson/Jacobi structure and atmosphere-ocean
coupling remain unqualified. One smooth-branch directional derivative and
analytic/AD identity tests are evidence, not a global differentiability claim.
