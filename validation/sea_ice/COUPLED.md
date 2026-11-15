# Coupled ocean and category ice

The optional [snow wrapper](SNOW.md) adds accumulation/melt, transport, insulation
and albedo while retaining ice-equivalent mechanical geometry.

`OceanIce` is an opt-in research wrapper around `RoquetOcean`. It now advances
actual ocean dynamics and moving two-category ice together. Packaged ocean and
atmospheric defaults remain unchanged. `run_coupled.py` supplies reproducible
small global, coastal and moving-ice-margin experiments.

## Ownership and use

- `ocean_ice.py`: shared-H transport/forces, ocean loading, wind, compression,
  composition and timestep validation.
- `ice_drag.py`: reciprocal implicit drag and positive ocean TKE production.
- `category_phase.py`: freezing/melting, category geometry, conservative phase
  momentum exchange and conductive forcing.
- `ridging.py`: state, common Hamiltonian and thin-to-thick redistribution.

State remains `RidgingState(ocean, CategoryIce(area, volume, velocity))`.
Area and volume have shape `[cell,2]`, ice velocity `[edge,1]`. The ocean keeps
its existing TKE and optional mesoscale-energy reservoirs.

```python
coupled = OceanIce(ocean_model)  # unforced RoquetOcean
state, external_energy_J = coupled.step(
    state, dt,
    heat_flux=Q,                # [cell,1], W/m², zero on land
    wind_velocity=wind_xyz,    # [cell,3], tangent Cartesian m/s
)
```

Alternatively supply `air_temperature` in degrees Celsius instead of heat flux.
The wrapper owns surface forcing: configured ocean boundary/heating/exchange
objects are rejected to avoid counting them twice. Existing vertical mixing and
instantaneous mesoscale backscatter are composed explicitly. Other externally
scheduled operators such as GM–Redi must still be composed by the caller; this
wrapper is not a complete climate driver.

## One Hamiltonian

With cell area A, total ice volume V, ice mass m=rho_i V and water elevation eta:

```
H = H_ocean
  + sum A [-Lf m + g m eta + g m²/(2 rho_i)]
  + 1/2 sum_edges (dc dv) I(m) ui²
  + sum A C [sum_categories V_k²/a_k - V²]
C = rho_i g (1-rho_i/rho0)/2
```

`H_ocean` includes the existing full Roquet thermal/mechanical energy, ocean TKE
and mesoscale energy. The subcell term assumes hydrostatic reference-density
water displacement, as in the isolated ridging model.

**Transport/loading.** Donor fluxes carry category area and volume. Ice pressure
forces use the analytic derivatives of this same H and the adjoint gradient;
ocean loading is `-grad(g*m)/rho0`. At an empty category, the derivative of V²/a
uses the thickness of incoming material. Simultaneous ridge and advective
births release their unresolved thickness variance as ocean thermal energy.

**Rotation.** A metric-antisymmetric stencil gives zero kinetic work. Harmonic
pair masses keep accelerations bounded at vanishing ice margins. Uniform mass
recovers the metric-skew mesh rotation; this margin extension is an energy
construction, not a discrete potential-vorticity conservation theorem.

**Drag.** Reconstruct both velocities to cell tangent vectors with R. The
stress is `a*rho0*Cd*|R(uo-ui)|*R(uo-ui)`. The exact transpose R^T applies equal
and opposite generalized edge forces. The cell power `stress·R(uo-ui)` goes
into surface ocean TKE. An implicit midpoint Newton/diagonally preconditioned
CG solve uses the same relative velocity for stress and TKE production. It
solves the constitutive exchange; it does not repair a total-energy residual.
Midpoint drag conserves KE+TKE to solver precision, but is not L-stable.

**Wind.** Bulk stresses act separately over covered and open-water fractions,
relative to the corresponding velocity. Their discrete work is external energy,
not internal heat/TKE production. There is no prognostic atmosphere here.

**Compression.** Thin-to-thick ridging remains, supplemented by compaction of
all categories so resistance does not disappear when thin ice is exhausted.
Stress work pays for category gravitational energy and the specified frictional
heating. This is compressive resistance only, without shear strength or elastic
memory.

**Phase change.** A per-cell scalar solve balances the phase-varying terms of H
against supplied heat, with CT=-0.054 SA for fresh zero-layer ice. Only the top
liquid layer changes; salt and passive-tracer contents remain in the liquid.
Existing categories grow/melt proportionally: footprint varies at fixed thickness
until coverage reaches one, then growth thickens ice. New ice enters category
zero at 0.1 m. Donor-velocity mass transfer conserves ocean–ice edge momentum;
recipient mixing loses KE into ocean TKE. Complete melting sets ice velocity
to zero only where its mass is zero. No mass floor or clipping is applied.

## Time integration and scope

The step is phase half / drag half / SSPRK3 dynamics / optional vertical mixing /
drag half / phase half. Splitting is second order for smooth phase/drag branches
without the separately applied mixing; the dynamics integrator is third order.
Instantaneous closure does not imply exact finite-step energy conservation.
Monitor H minus accumulated external wind work and heat, and refine dt. Wind,
transport and compression remain explicit. Invalid stages or nonlinear-solver
failures raise rather than silently clipping or retrying.

The phase solve rejects exhausted liquid/EOS brackets (SA<=42 g/kg and at least
1 m top liquid thickness). This is a first two-category closure, not a calibrated
ice-thickness distribution, VP/EVP rheology, snow/brine/ice-enthalpy model, ice
thermodynamic Gibbs equilibrium, or complete freshwater/atmosphere coupling.
Static coasts are supported; dynamic wetting/drying has not been qualified.
A common energy functional and matched work do not establish a Poisson/Jacobi
structure or entropy theorem.

## Run and check

From an environment with the current VFS wheel installed:

```sh
python -I validation/sea_ice/run_coupled.py --cells 128 --hours 24 --dt 120 \
  --coast --patch --wind 8 --heat -30 --verbose --output coupled-ice-128
python -I validation/sea_ice/run_coupled.py --cells 32 --hours 6 --dt 60 \
  --heat 5000 --output coupled-ice-melt
```

Outputs are `series.csv` and `summary.json`, including conservation errors,
forcing, velocities, coverage, ice mass, timing and source hashes. These small
runs are numerical coupling checks, not climate realism or timestep limits for
higher resolutions. See [recorded evidence](../../verification/ocean/sea-ice/coupled/README.md).
Tests independently differentiate H, check drag transpose/work and phase
momentum, exercise coasts/empty categories/complete melt, and compare selected
smooth-branch backward sensitivities. CPU float64 is tested; GPU, compile,
float32 and long-horizon differentiability are not qualified.
