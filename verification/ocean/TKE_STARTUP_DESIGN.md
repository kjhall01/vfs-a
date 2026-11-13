# Starting ocean turbulence from zero

## Recommendation

Add an optional **prescribed surface TKE energy flux** F_k [W/m²], positive
into the water. Default zero. It represents mechanical energy delivered directly
to unresolved motions, for example by breaking waves. This is a proposed next
operator, not implemented by the sustained-column campaign.

Craig and Banner (1994) model breaking-wave input as a surface TKE flux.
The official GOTM wave-breaking case uses this boundary condition and provides
an independent benchmark. Their results also show why prescribing a constant
mixing length does not establish a realistic near-surface turbulence profile.

- [Craig and Banner (1994), Modeling Wave-Enhanced Turbulence in the Ocean Surface Layer](https://journals.ametsoc.org/abstract/journals/phoc/24/12/1520-0485_1994_024_2546_mwetit_2_0_co_2.xml)
- [GOTM surface-wave-breaking case](https://gotm.net/cases/wave_breaking/)

## Discrete energy accounting

For cell area A, reference density rho0 and top-layer thickness h_0, add

    delta(h*k)_0 = dt * F_k / rho0
    delta E_TKE  = A * dt * F_k

before evaluating turbulence diffusivities. Retain a separate external energy
ledger entry. Do not add the same amount to thermal energy: subsequent modeled
TKE dissipation supplies heat through the existing energy-closing operator.
No momentum is implied by this energy-only boundary input. The incoming water's
carried energy and sensible/latent/radiative heating remain separate terms.

This creates finite top-layer TKE even if the initial column has k=0. Its downward
transport and shear production can then proceed. Wind stress itself supplies
resolved momentum and finite-step resolved kinetic work; its presence does not
uniquely determine a wave-breaking energy flux. Do not silently debit resolved
stress work again or assume a fraction of it is wave power. In a coupled model,
the supplied TKE energy must have a matching debit in the atmosphere/wave energy
budget, with the same interface exchange convention.

First expose a prescribed flux; leave any stress/wave-state formula outside the
boundary operator. A later optional parameterization could use
F_k = rho0 * alpha * u_*w^3, u_*w = sqrt(|tau|/rho0), following the Craig–Banner
scaling. The coefficient and wave-state assumptions require qualification; the
formula is not an exact conversion of resolved ocean stress work.

## Fit to the existing API (proposed, not available yet)

Extend `OceanBoundaryStress.apply` and `TKEOcean.boundary_with_report` with an
optional `surface_tke_flux` tensor on **cells**, in W/m². Surface stress remains
on edges. The boundary operator already receives `tke_h` to deposit bottom-drag
loss, so this does not require a new state object or a new column model. Route
the same keyword through `step_with_report` before vertical mixing.

Keep `BoundaryStressReport.surface_work` as resolved wind work. Add separate
cell and total surface TKE input fields, and an explicit total external-energy
quantity equal to wind work plus TKE input. Update caller ledgers to use that
quantity when the new forcing is enabled. Reject nonfinite input and nonzero
input on dry cells; a first version can restrict the prescribed input to
nonnegative values. With omitted/zero input, retain existing numerical behavior.

## Tests before physical claims

1. Zero TKE plus prescribed positive F_k produces exactly the supplied energy;
   zero flux leaves the zero state unchanged. Preserve salt, mass and momentum.
2. With production/decay disabled, cumulative TKE equals integrated input;
   with decay enabled, TKE plus heat equals input. Test variable thickness,
   wet/dry masks and timestep refinement.
3. Run the wind-column campaign from exactly zero TKE, then vary initial seeds
   over several orders of magnitude under the same prescribed F_k. Compare
   profile/entrainment sensitivity and limiter activity, not just survival.
4. Use the GOTM idealized wave-breaking case as a physical reference, explicitly
   matching length-scale assumptions or documenting their differences.

## Separate limitations

The present K_m proportional to sqrt(k) closure has zero mixing and zero shear
production at k=0. Changing variables to sqrt(k), choosing a growing solution at
zero, or adding an undocumented floor changes the startup assumption; none by
itself establishes a physical energy source.

Surface injection does not solve wind-free convective onset: with zero
conductance, an unstable column also cannot initiate buoyancy mixing. That needs
a separately justified convective exchange that converts the measured reduction
in resolved gravitational/thermodynamic energy into TKE, or a closure with a
well-defined instability-driven onset. Molecular viscosity is another separate
process; its loss normally becomes heat, not an arbitrary turbulent energy seed.

A source fixes startup, not a poor mixing-length or stability formulation.
Use the sustained limiter/refinement evidence to decide whether those need
revision before claiming a calibrated mixed-layer model.
