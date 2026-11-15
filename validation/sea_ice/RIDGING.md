# Compression stress and two-category ridging

The subsequent [coupled ocean–ice wrapper](COUPLED.md) composes transport,
loading, rotation, reciprocal drag, wind and category-aware phase change.
The isolated API described below retains its original scope.

`ridging.py` is an opt-in **isolated mechanics operator** on the existing Voronoi
mesh. It advances ice velocity, category area/volume and ocean surface thermal
content. It does not run ocean dynamics, transport, drag, Coriolis, freezing,
melting, wind forcing or shear rheology. Do not combine it with an unmodified
ocean timestep and claim a closed coupled ocean–ice system.

The implementation separates `CategoryIce`, `RidgingState`,
`RidgingHamiltonian`, `CompressionRidging` and its tendency/report. The existing
stationary thermodynamic API and all packaged defaults are unchanged.

## State and energy

Two columns of `area` and `volume` represent thin and thick ice; each has shape
`[cells,2]`. Volume is m³ ice per m² horizontal cell. Thickness is `V_k/a_k` in
present categories; empty categories have exactly zero area and volume. The
shared ice normal velocity has shape `[edges,1]`. The ocean is its existing
`TKEState`/`EddyState`; no extra untracked energy reservoir is introduced.

The common functional extends the previous full ocean/slab-ice Hamiltonian:

```
H = H_slab + 1/2 sum_e (dc*dv)_e I(rho_i*V)_e u_i,e^2
           + sum_c A_c C [sum_k V_k^2/a_k - V^2]
V = sum_k V_k; C = rho_i*g*(1-rho_i/rho0)/2
```

The last term represents subcell ice/water gravitational energy from a thickness
distribution. For hydrostatic reference-density water at mean liquid height eta,
the local ice underside is `eta+(rho_i/rho0)*(V-h_k)`. Summing the ice PE and
water free-surface variance yields the original slab PE plus this correction.
Subcell seawater stratification/EOS anomalies are neglected: this is an explicit
mechanical subgrid energy assumption, not an exact subcell Roquet integration.
A single fully covered category reduces exactly to the previous slab energy.
The ordinary resolved Roquet ocean, TKE, eddy, latent and gravitational terms
remain part of H. Ice KE uses the mesh's existing grad/div metric and interpolation.

## Constitutive law and instantaneous work

For coverage `a=sum a_k`, ice divergence D, compression `c=max(-D,0)`, define

```
chi = a^n * c/(c+c_star)
F = chi*c*V_thin
Vdot_thin = -F;             Vdot_thick = F
adot_thin = -F/h_thin;      adot_thick = F/h_thick
P_ridge = C*chi*V_thin*(h_thick-h_thin)
P = (1+f)*P_ridge + zeta*chi*a*c
```

P is vertically integrated compressive pressure (N/m), zero in separating cells.
The positive `c_star` regularizes the resistance at zero compression; it avoids
a finite pressure jump at vanishing deformation. Defaults `n=4`, `c_star=1e-6/s`,
`f=1`, `zeta=0` are uncalibrated experiment choices, not tuned sea-ice parameters.
`zeta` has units N s/m. A new thick category is born at `ridge_ratio*h_thin`
(default ratio 3). Existing receiver thickness must be >= donor thickness.
Transfer preserves the two representative thicknesses and changes their weights.
A zero-thickness/zero-area category is never populated by dividing by zero.

The acceleration is `-grad(P)/I(rho_i V)`, projected onto active, massive edges.
The shared incidence/metric pairing gives

```
dK_ice/dt = sum_c A_c P D = -sum_c A_c P*c
ridge_PE_power = A_c P_ridge*c
heat_power = A_c [f*P_ridge + zeta*chi*a*c]*c >= 0
```

The mass-preserving transfer leaves slab latent energy and ice kinetic mass
unchanged. Its derivative of the category PE is exactly the indicated ridge
power, including the one-sided derivative at category birth. Thus the total
instantaneous H derivative vanishes on a closed domain. Forward code uses no AD.
Independent tests differentiate the actual full functional.

The report also retains conservative spatial stress-work redistribution:
individual cells can gain KE through stress work even though the global KE
change is nonpositive. `energy_redistribution` sums to zero; it is not missing
local heating or a negative reservoir withdrawal.

All crushing dissipation goes to ocean surface **thermal energy**, an explicit
allocation choice for zero-heat-capacity ice. This is distinct from the proposed
ice–water drag-to-ocean-TKE allocation. The temperature-content tendency divides
by the full-H caloric conjugate, including Roquet potential-energy changes;
it does not simply divide by cp. Existing TKE and eddy stores remain unchanged.

## Running and testing

From the checkout with VFS installed:

```bash
python -I validation/sea_ice/run_ridging.py --cells 32 --seconds 1000 \
  --dt 500 250 125 20 10 5 --output ridge-open
python -I validation/sea_ice/run_ridging.py --coast --dt 20 10 --output ridge-coast
```

The driver creates a converging/diverging ice velocity field on a 100-km-radius
sphere and advances only this mechanics stage with RK4. It saves CSV time series
and JSON metrics/provenance. Ocean CT is initialized at zero for sensitive
mechanical-budget diagnostics; this is not a phase-equilibrated ice climate.
No automatic retries, timestep changes, energy fixes or state clipping occur.
Every stage is checked; invalid coverage, thickness ordering or negative stores
reject the step. Birth is nonsmooth, and backward evidence covers only a smooth
positive-category branch.

`stable_change` is an independent factored **diagnostic** of the same endpoint
energy difference; it avoids subtracting the large common ocean/latent datum.
Both stable and direct full-H differences are reported. Neither feeds back into
the state. Finite RK error is measured and decreases with timestep refinement.

## Deliberate limits and next integration work

This is not VP/EVP, an ice concentration limiter, or a complete ice dynamics model.
Only donor thin ice is redistributed. Once it is depleted, this law cannot ridge
the thick category further; with zero bulk viscosity its compressive resistance
also vanishes. Shear, tensile failure, snow, rubble porosity and evolving ridge
thickness spectra are absent. No angular-momentum qualification is claimed.

Without advection, redistribution alone decreases coverage: it is the mechanical
part that would accompany transport-driven concentration increase. The example
therefore demonstrates stress relaxation/redistribution, not a complete moving
ice patch or a physical prediction of gap evolution. Coverage checks do not
constitute a coupled transport/ridging admissibility algorithm.

Before global integration: implement conservative area/volume/momentum transport,
load the ocean consistently with the new H, add reciprocal drag with its TKE
source and phase momentum transfer, then coordinate concentration admissibility
with redistribution. The old slab phase solver cannot be applied directly to
the new category state. No method here bypasses that limitation.

Design context: [Icepack mechanical redistribution](https://cice-consortium-icepack.readthedocs.io/en/icepack1.5.0/science_guide/sg_mechanical.html).
This prototype is much simpler and is not an implementation of that scheme.
See [measured results](../../verification/ocean/sea-ice/ridging/README.md).
