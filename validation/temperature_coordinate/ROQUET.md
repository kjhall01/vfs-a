# Roquet freezing ocean configuration

`RoquetOcean` is an explicit research configuration with the four published
Freezing coefficients from Roquet et al. (2015), Table 3. It has active
Conservative Temperature (CT, degrees Celsius), Absolute Salinity (SA, g/kg),
layer thickness, edge velocity, optional passive contents and prognostic TKE.
The JMD95 and linear configurations/defaults are unchanged.

```python
# With this research directory and validation/ocean_column on sys.path:
from roquet import RoquetOcean, RoquetMixing, RoquetHeating, TemperatureBoundary
model = RoquetOcean(mesh, column,
    closure=RoquetMixing(2., substeps=8),
    boundary=TemperatureBoundary(.0025),
    heating=RoquetHeating(.6, (.5, 20.)))
state = model.state(u_e=u, h=h, conservative_temperature=CT,
                    absolute_salinity=SA, tke=1e-4)
state, report = model.step_with_report(state, 60., surface_stress=stress,
                                      shortwave=shortwave, surface_heat=heat)
```

`temp_h` stores h*CT; `salt_h` stores h*SA. Generic `temperature` and `salinity`
initialization keywords are deliberately rejected. Internal legacy tensor
accessors `.temperature()`/`.salinity()` consequently return CT/SA in this
configuration. States must not be interchanged with JMD95 states without explicit
conversion. Existing JMD95 restart/output metadata is not qualified for these CT/SA fields;
label diagnostic outputs explicitly. There is no new installed public API/export yet.

`convert_jmd95_initialization` optionally uses GSW to convert IPTS-68 potential
temperature/Practical Salinity to CT/SA, given pressure in dbar and geographic
longitude/latitude. IPTS-68 potential temperature is first divided by 1.00024.
This is an explicit non-differentiable NumPy initialization boundary; gradient
inputs are rejected. Native CT/SA initialization and all runtime operators have
no GSW dependency. Conversion alone does not match the two models' buoyancy.

## EOS and datum

With z positive upward (negative below sea level), define

    P(CT,SA,z) = s*SA + t*CT + tt*CT² - tz*CT*z
    s  =  0.7718       kg/m³/(g/kg)
    t  = -0.0491       kg/m³/K
    tt = -0.005027     kg/m³/K²
    tz = -2.5681e-5    kg/m⁴/K

The paper uses positive depth; SeawaterPolynomials' evaluator takes geopotential
height and includes the minus sign before tz. This sign is independently tested.
The model's density is rho0 + P(CT,SA,z)-P(CT0,SA0,z), with default CT0=10 and
SA0=35. Thus the density *derivatives* and anomalies match the published form;
the absolute density offset is a declared Boussinesq datum, not a TEOS density
prediction. Depth-only compression is not inserted into active buoyancy.

Dynamic enthalpy relative to the fixed reference column is

    Psi = (g/rho0) * [z*(s*(SA-SA0) + t*(CT-CT0) + tt*(CT²-CT0²))
                     - (tz/2)*z²*(CT-CT0)].

The exact layer mean replaces z² by z_mid²+h²/12. It is a quadratic function
A*CT²+B*CT+C*SA+D at fixed geometry. Thickness derivatives also include moving
layer endpoints and all lifted overlying layers; they are independently checked
against autograd. The full Hamiltonian includes mechanical energy, this layer
integral, reference thermal energy and TKE.

Shared ALE diagnostics use a two-node Gauss rule that is **exact** for this
quadratic polynomial. Dynamics and mixing use direct coefficient expressions;
there is no nested nonlinear EOS quadrature or iterative temperature inversion.

## Caloric choice and analytic mixing

The reference potential enthalpy is h0=cp*CT with cp=3991.86795711963 J/kg/K.
The accompanying **idealized** entropy coordinate is

    eta_model = cp*log((CT+273.15)/273.15).

This supplies an explicit mathematical Gibbs reference and matching face means.
It is **not physical TEOS-10 entropy**, and CT+273.15 is not in-situ absolute
thermodynamic temperature. Entropy nondecrease checks apply to this model entropy.
The density polynomial alone does not specify a complete caloric equation of
state. This declared approximation must remain visible in any climate claims.
Internal inherited field names `to_90` and reference domain keys are compatibility
names only: no CT-to-potential-temperature conversion occurs during stepping.

For fixed h, the exact finite buoyancy-energy change is

    delta_Psi = (2*A*CT+B)*delta_CT + A*delta_CT² + C*delta_SA.

Conservative interface enthalpy fluxes give delta_CT algebraically. Evaluating
the conjugates at the midpoint of the **total** layer increment yields an exact
quadratic discrete gradient and interface work. This includes the cross term
from simultaneous upper/lower interface exchanges. The existing neighboring-
layer allocation, tridiagonal mixing, TKE funding retries and full-H acceptance
remain unchanged. This is not an unconditional one-shot mixing solver.

Heating q J/kg solves A*dCT² + capacity*dCT = q using the cancellation-safe root

    dCT = 2*q / (capacity + sqrt(capacity²+4*A*q)),
    capacity = cp+2*A*CT+B.

Positive capacities/discriminant and temperature/salinity bounds are checked.
Existing full energy/entropy checks remain. The inherited mixing candidate
currently requires fully wet columns; coastal derivative tests do not qualify
coastal mixing. Prescribed heating, surface exchange,
stress and adaptive ALE use the same energy. ALE still has feasibility limits
and an iterative thermal allocation; analytic mixing does not eliminate that.

## Run and scope

```sh
python validation/temperature_coordinate/roquet_campaign.py --ale --output roquet.json
python validation/temperature_coordinate/roquet_campaign.py --cells 32 --layers 16 \
  --profile cold --dt 600 --steps 144 --mixing-substeps 8 --ale --output cold.json
python validation/temperature_coordinate/roquet_campaign.py --wave --cells 128 --output wave.json
python -m pytest tests/ocean/test_roquet_configuration.py -q
```

Cold and strong-contrast cases, analytic derivative/work/heating checks, runtime
inversion-disabled checks, conversions and backward smoke tests are included.
Runtime bounds are CT in [-2.5,40] and SA in [0,42], not a global accuracy guarantee.
There is no ice formation/freezing-point enforcement, true TEOS entropy, steric
sea-level correction or climate/AMOC qualification. No GPU/float32 qualification
or additional compilation speedup is claimed. See
[measurements](../../verification/ocean/roquet/README.md).

Sources: [Roquet et al. 2015](https://doi.org/10.1175/JPO-D-15-0080.1);
[SeawaterPolynomials source](https://github.com/CliMA/SeawaterPolynomials.jl/blob/main/src/SecondOrderSeawaterPolynomials.jl),
retrieved blob SHA b7c0336f8c350dcf0debdde4451099ff75fcce77. Coefficients and
polynomial signs were cross-checked against this source; no Julia runtime is
used or Julia execution equivalence claimed. The implementation is derived
from the published equations, not a vendored dependency.

For timestep selection at a specified horizontal and vertical resolution, use
[the timestep campaign](../timestep/README.md). Its deep-ocean tests differ from
the shallow throughput example, and distinguish completion from temporal accuracy.
