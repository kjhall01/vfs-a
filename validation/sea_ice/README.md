# Hamiltonian ocean–ice columns

The subsequent [coupled ocean–ice wrapper](COUPLED.md) composes transport,
loading, rotation, reciprocal drag, wind and category-aware phase change.
The isolated API described below retains its original scope.

An opt-in research component, **stationary columns only**, based on the useful
thermodynamic ideas in the October 2026 `VFS-main.zip` prototype. No packaged
model defaults change. This is not yet a moving-ocean sea-ice configuration.

## Components and common energy

- `SeaIceState`: fresh ice mass per horizontal cell area, kg/m².
- `IceOceanState`: existing Roquet ocean/reservoir state plus ice.
- `IceHamiltonian`: existing ocean Hamiltonian plus latent and ice gravitational
  energy. Uses the existing exact ocean column decomposition, not another EOS.
- `SeaIceThermodynamics`: local freezing/melting, freshwater transfer and brine
  concentration, constrained by that same energy functional.
- `ConductiveIceBoundary`: prescribed air temperature, finite air resistance
  and ice conduction; reports external heat into the combined system.

For cell area A, ice mass per area m, liquid surface height eta and ice density
rho_i, the total energy is

```
H = H_ocean + sum A * [-Lf*m + g*m*eta + g*m*m/(2*rho_i)]
```

`H_ocean` includes resolved kinetic, free-surface and Roquet dynamic-enthalpy
energy, caloric energy, turbulent TKE and mesoscale eddy energy. Ice occupies a
fully covered slab from eta to eta+m/rho_i. Its mass decreases liquid thickness
by dm/rho0. This is the ocean's reference-density mass convention, not a claim
that Boussinesq volume tracks in-situ seawater mass exactly.

The phase endpoint obeys both local full-H energy balance and, when ice remains,
`CT_top = -0.054 * SA_top`. Salt amounts stay in the liquid. Passive tracer, TKE
and mesoscale reservoir contents also remain there: this is an explicit
retention assumption. Complete melting uses the existing ocean full-H heat
solve for surplus energy. No prognostic store is clipped, and no extra reservoir
is adjusted afterward to hide an energy discrepancy.

The scalar phase solve is an equilibrium constitutive update, not an ocean
momentum integrator. On the ice-bearing equilibrium branch, its tangent obeys
`(dH/dm)_phase * dm/dt = A*Q`, with the derivative taken along the mass-conserving,
salt-conserving liquidus constraint. Both the tangent and finite update use the
same H. This does **not** establish a Hamiltonian/Poisson structure for irreversible
freezing, or thermodynamic entropy consistency of the empirical liquidus.

Bisection brackets the root without recording a backward graph. Three analytic
Newton steps restore fixed-branch implicit sensitivities. The forward path
never calls autograd. Phase onset is nonsmooth; tests qualify only a smooth
ice-bearing branch, not arbitrary long backward rollouts.

## Run it

With VFS installed, from the checkout:

```bash
python -I validation/sea_ice/run.py --cells 32 --days 365 --dt 86400 --output ice-columns
```

The driver explicitly loads the research modules, runs independent columns with
seasonal prescribed air temperature, and writes `series.csv` and `summary.json`.
Larger `--cells` tests batched throughput, not horizontal physical resolution.
No ocean dynamics or lateral heat/salt transport runs in this example.

For manual use, create `IceOceanState(ocean_state, SeaIceState(mass))`, construct
`SeaIceThermodynamics(roquet_ocean)`, and call
`end, report = thermo.apply(state, dt, heat_flux=Q)`. Q is a tensor shaped
`[cells,1]`, W/m², positive into the system. Ice mass and Q must be zero on dry
cells. A `ConductiveIceBoundary().flux(state, air_temperature)` supplies Q;
explicitly zero its dry-cell entries when using a coastal domain.

## What remains

Ice has zero sensible heat capacity and no snow, partial concentration, trapped
salt, radiative/albedo feedback, transport, momentum or rheology. The air boundary
is an idealized conductance, not an atmosphere or full surface-flux scheme. The
liquidus is empirical in Conservative Temperature, not a chemical-potential
calculation. Liquid remains within the existing EOS range and a configurable
minimum top-layer thickness; forcing beyond this bracket raises an error.

The ice term adds `dH_ice/deta = A*g*m`. Moving-ocean pressure forces must include
that load, and moving ice needs kinetic energy, reciprocal stresses and consistent
transport/phase momentum exchange. The API rejects nonzero ocean velocity so
these missing terms cannot silently invalidate the advertised column closure.
Calling the standalone ocean step between ice updates is not supported.

Next integration order: derive loading and phase momentum transfer; add paired
ocean/ice drag whose KE loss heats the common thermal store; add conservative
ice transport; then a rheology with a provable discrete stress-work identity.
The archive's EVP and global force correction have not been ported or qualified.

See [verification](../../verification/ocean/sea-ice/README.md).

## Separate compression mechanics prototype

[Two-category ridging](RIDGING.md) now implements an isolated instantaneous
stress/redistribution/heating stage with its own extended common Hamiltonian.
It includes ice KE and subcell hydrostatic ridge energy. This does not remove
the stationary restriction of the thermodynamic API above: category phase
change, transport, ice loading and drag still require a combined integration.
