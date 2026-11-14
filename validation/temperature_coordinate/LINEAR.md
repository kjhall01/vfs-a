# Linear-density ocean configuration

`LinearOcean` is an opt-in research configuration with active temperature and
salinity, prognostic layer TKE, prescribed surface fluxes, bottom stress and
energy-corrected ALE. The supported JMD95 configuration and defaults are unchanged.
It uses the shared C-grid transport, rotation, stage checks and splitting.
It is imported from this research directory, not from the installed public API.

```sh
# Run from the repository with VFS installed. No external data are required.
python validation/temperature_coordinate/linear_campaign.py --ale --output linear.json
python validation/temperature_coordinate/linear_campaign.py --cells 1024 --layers 64 \
  --compare-jmd95 --ale --steps 2 --output linear-comparison.json
python -m pytest tests/ocean/test_linear_configuration.py -q
```

The campaign is also a complete runnable construction example. `LinearOcean`
accepts `LinearEOS`, `LinearMixing`, `TemperatureBoundary`, `LinearHeating` and
`LinearExchange` instances. `state(...)` accepts physical temperature, salinity,
velocity, thickness, passive tracers and specific `tke`; its prognostic fields
are `u_e`, `h`, `h*T68`, `h*S`, tracer contents and `h*k`. Call
`step_with_report(state, dt, ...)`, then optionally `remap_with_report(state,
target_h)`. ALE scheduling remains explicit. Use `total_energy(state)` for the
matched energy including TKE. Nonlinear mixing/heating/exchange adapters are
rejected by the constructor.

## Energy and dynamics

With height z measured upward from fixed sea level, define

    r = -alpha*(T68-T0) + beta*(S-S0)
    rho = rho0*(1+r)
    H = K + free-surface energy + sum(rho0*A*h*(g*z*r + cp*T90))
        + sum(rho0*A*h*k)
    T90 = T68/1.00024

The layer midpoint is the exact vertical integral for this potential: there is
no nonlinear density quadrature. Derivatives include midpoint motion and the
lifting of all overlying layers when thickness changes. They are checked against
autograd, including coastal masks. Pressure/buoyancy forces use these derivatives;
the actual spatial RHS is checked for zero Hamiltonian work. This establishes
spatial energy closure, not exact SSPRK3 finite-step conservation or a Jacobi proof.

The caloric reference is h0=cp*T90 and eta=cp*log((T90+273.15)/273.15).
Its face entropy is the analytic temperature-path mean, satisfying
Delta G + eta_face*Delta Kelvin = 0. Salinity has no caloric chemical potential;
it remains active through the buoyancy energy. Temperature is prognostic, but
its tendency retains the entropy-coordinate chain rule. It is **not** a passive
conservative temperature tracer. Mass, salt and passive inventories use the
existing conservative transport.

Defaults: cp=3990 J/kg/K, T0=10 degrees T68, S0=35,
alpha=0.0001908821582383638 per degree T68 and
beta=0.0007568878444130999 per practical salinity unit. These frozen expansion
coefficients came from local JMD95 slopes at T68=12.5, S=34.75 and zero pressure,
divided by rho0=1025. They are not recomputed as the state evolves. The density
intercept is deliberately rho0 at T0,S0; this is not a full tangent fit to absolute
JMD95 density. Changing rho0 does not recalibrate alpha or beta.

## Mixing and heating

At fixed thickness, let Fh and Fs be the time-integrated interface fluxes of
reference enthalpy content and salt content, positive into the upper layer.
The exact buoyancy-energy cost per interface is

    W = rho0*A*((a_upper-a_lower)*Fh + (b_upper-b_lower)*Fs)
    a = -g*alpha*z/(cp/1.00024)
    b = g*beta*z

No state-dependent path integral or iterative enthalpy inversion is required.
The existing adjacent-layer allocation, lagged tridiagonal mixing, TKE funding
retries, shear production, dissipation, positivity and full-H/entropy acceptance
checks remain in place. This is not a new one-shot guaranteed-admissible solver.

A prescribed specific heat increment q at fixed geometry uses

    delta_T68 = q/(cp/1.00024 - g*alpha*z).

Thus forcing, dissipative heating, mixing and ALE all use the same energy.
ALE still uses the existing adaptive thermal redistribution and can reject an
infeasible target. Linear thermodynamics does not eliminate remap geometry work
or the funding/acceptance constraints.

## Scope

This is an idealized Boussinesq thermodynamic model, not realistic seawater
thermodynamics. It omits pressure-dependent expansion, nonlinear cabbeling and
thermobaricity, and salinity-dependent caloric properties. Runtime guards retain
T90 in [-2.5,40] degrees C and salinity in [0,42]; those are numerical bounds, not
an accuracy guarantee for a fixed linearization. Positive density and full-H
thermal derivative are checked.

Float64 CPU checks include independent derivatives, finite mixing work, stable
and unstable profiles with zero/tiny/finite TKE, the full forcing/remap path with
nonlinear evaluators disabled, and a composed backward smoke test. Passing these
is not a global wave/jet, climate, GPU, float32 or full differentiability
qualification. The inherited local funding and entropy constraints can still
reject states or large steps. See the recorded campaign for its actual duration,
errors, timing samples and any rejection. Comparisons with JMD95 have different
physics and do not establish equal physical accuracy.
