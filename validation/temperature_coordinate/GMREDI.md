# GM–Redi with a Hamiltonian eddy-energy reservoir

`eddy.GMRedi` supplies prescribed-coefficient eddy transport for the Roquet CT/SA
research ocean. `EddyState` adds an explicit cell-layer content `eddy_h = h*e`
[m³/s²], where e is mesoscale eddy energy [J/kg]. This is separate from turbulent
TKE. The model owns one full functional through `model.hamiltonian`:

    H = H_Roquet(resolved ocean) + sum(rho0*A*h*k) + sum(rho0*A*h*e).

`model.total_energy(state)` evaluates this functional. Its resolved part is the
existing mechanical energy, exact Roquet layer-integrated dynamic enthalpy and
reference thermal energy. The reservoir represents unresolved mesoscale energy;
no kinetic/potential partition of that energy is specified yet.

GM and Redi both exchange their actual finite Hamiltonian work with this
reservoir. They do not modify temperature to repair the budget. Accepted steps
close the local first law and global energy budget to floating-point roundoff.
The existing RK dynamics retain their finite-timestep integration error; that
error is not charged to the reservoir.

This remains a potential-density, small-slope research parameterization, with
prescribed coefficients and explicit positivity/funding rejection. It is not
an exact neutral closure or a calibrated climate parameterization. Installed
production defaults remain unchanged.

## Use

As with `roquet.py`, put `validation/temperature_coordinate` and
`validation/ocean_column` on the import path. Initialize reservoir energy
explicitly when constructing a Roquet state:

```python
state = model.state(u_e=u, h=h, conservative_temperature=CT,
                    absolute_salinity=SA, tke=1e-4, eddy_energy=.1)
```

Or add it explicitly to an existing `TKEState`:

```python
from eddy import GMRedi

state = model.with_eddy_reservoir(state, specific_energy=.1)
eddy = GMRedi(kappa_gm=1000., kappa_redi=1000.,
              max_slope=.01, min_n2=1e-8)

state, physics_report = model.step_with_report(state, dt)
state, eddy_report = eddy.apply(model, state, dt)
# Optional: state, remap_report = model.remap_with_report(state, target_h)
H = model.total_energy(state)
terms = model.hamiltonian.terms(state)
```

Here 0.1 J/kg and 1000 m²/s are **test choices**, not recommended climate values.
No seed energy is invented when the reservoir is empty. GM/Redi now require an
`EddyState`; the old unfunded `TKEState` call raises. Initializing a reservoir
changes the initial total energy and must be included in the initial budget.

The reservoir is carried as a conservative scalar by existing resolved
transport, vertical mixing and ALE, then restored to its explicit state field.
The implementation assumes zero mesoscale-energy content in external freshwater
fluxes. Surface and mixing energy exchanges otherwise retain their existing
reports. `model.energy` remains the resolved-ocean component; use
`model.hamiltonian` or `model.total_energy` for the complete state.

At fixed prognostic `eddy_h`, the reservoir term has no h or velocity derivative;
its conjugate is rho0*A. Existing resolved pressure and momentum derivatives
therefore remain those of the full H. Independent autograd checks test all
resolved derivatives and both energy-reservoir conjugates.

## Slopes and transport

Layer indices increase downward, physical z upward. From the same Roquet EOS,
use surface-referenced buoyancy

    b = -g/rho0 * anomaly(CT, SA, z_reference=0).

With dn the edge center distance, ell its width and brackets an adjacent-cell
mean, the physical normal density-surface slope is

    S = (z_j-z_i)/dn - [(b_j-b_i)/dn] / <Dz b>.

Dz averages neighboring secants internally and is one-sided at endpoints. The
height correction cancels spurious slopes for a vertically linear, horizontally
flat density field on tilted layers. At least two layers are required.
For `<Dz b> <= min_n2`, the slope and entire Redi row are disabled. Other slopes
are capped at ±max_slope. These declared geometry regularizations do not clip
prognostic tracers. Potential-density directions omit thermobaricity even though
the energy work uses the full thermobaric Roquet Hamiltonian.

GM averages slopes to internal interfaces, sets Psi=kappa_gm*S and sets Psi=0
at the top and bottom. Layer volume transports are

    Q_h = ell * (Psi_lower - Psi_upper).

Downward vertical transport is the cumulative horizontal inflow above each
interface. Combined divergence vanishes to roundoff, with closed top, bottom
and coast boundaries. Donor advection transports CT, SA, passives, TKE and e.
The outflow fraction must not exceed `cfl` (default 0.8).

Redi **recomputes slopes after GM**, then uses a conservative adjoint operator:

    G q = (q_j-q_i)/dn + [S - (z_j-z_i)/dn] * <Dz q>,
    W = kappa_redi * ell * dn * <h>,
    M = diag(A*h),
    dq/dt = -M^-1 G^T W G q.

G annihilates constants and W is nonnegative. Thus tracer inventories are
conserved and the semidiscrete volume-weighted quadratic norm decreases.
The explicit Euler step requires `dt * Gershgorin_bound <= cfl`. Slopes are
frozen within each substep, not across the GM–Redi pair. The composition is
first order in time, including when coupled to SSPRK3 resolved dynamics.

**Variance stability is not positivity preservation or energy neutrality.** A
sharp passive patch can go negative under rotated Redi even at small dt. Such
updates raise before returning and leave the input unchanged. There is no
silent clipping, timestep change or transfer limiting.

## Local energy exchange from the same H

`RoquetHamiltonian.midpoint_conjugates` and `.finite_change` use the exact
coefficients of `RoquetEnergy`, including reference cp, nonlinear CT² and
thermobaric terms. At unchanged h and u, H is quadratic in CT and linear in
SA/TKE/e, so the midpoint discrete gradient gives exact finite work. Forward
transport uses no autograd-generated tendencies.

On the experimental parcel-energy branch, each GM face carries the donor's
caloric energy, Roquet potential, turbulent energy and mesoscale energy per unit
mass. The potential uses the mean of the two adjacent layers' polynomial
coefficients evaluated at the donor CT and SA. The same bolus volume flux
determines the donor for tracer and energy transport. For each Redi stencil
row, use the midpoint conjugates averaged at
its two horizontal endpoints for every transfer in that row. Row transfers sum
to zero, so this energy transport is conservative too.

After a trial transport, for each cell-layer i compute

    conversion_i = [H_i(after transport)-H_i(before)] - energy_inflow_i,
    (rho0*A*eddy_h)_new = (rho0*A*eddy_h)_transported - conversion_i.

The specific conservative energy flux above is an explicit spatial allocation
choice; a global Hamiltonian alone does not uniquely determine local eddy
production. The resulting local equation is

    H_i(after exchange)-H_i(before) = energy_inflow_i.

Closed-domain energy inflows sum to zero. Redi may release energy or require
energy, depending on the actual EOS/state; either sign uses this same equation.
An unaffordable local withdrawal raises. Positive energy somewhere else cannot
silently fund the deficit. At an empty reservoir, reducing dt may not resolve a
costly flux: its coefficient/initial condition needs a physical energy source.

This scheme does not yet parameterize reservoir dissipation, backscatter,
eddy stresses, an EKE/EAPE partition or energy-dependent diffusivities.

## Diagnostics and verification

All report fields retain tensor graphs. Horizontal fields use active wet–wet
edges in mesh order; vertical transports are positive downward.

| Field | Meaning / units |
|---|---|
| `slope`, `redi_slope` | GM and refreshed Redi physical slopes, dimensionless |
| `streamfunction` | GM Psi, active edge × interface, m²/s |
| `horizontal_transport`, `vertical_transport` | Signed GM volume transports, m³/s |
| `volume_residual` | Combined GM volume inflow, cell × layer, m³/s |
| `gm_outflow_fraction`, `redi_step_number` | Explicit step restrictions |
| `inventory_change` | Actual content integrals in CT, SA, passives, TKE, e order; concentration units × m³. The e entry includes exchange and need not vanish. |
| `redi_variance_change` | Redi transport-only change of half volume-weighted squared concentrations, before reservoir exchange |
| `energy_change` | Full H change over GM+Redi, including reservoir, J |
| `gm_energy_change`, `redi_energy_change` | Resolved+TKE work, excluding reservoir, J |
| `reservoir_change` | Total change in mesoscale reservoir, J |
| `gm_reservoir_exchange`, `redi_reservoir_exchange` | Exchange with reservoir, excluding its transport, J |
| `gm_energy_flux_divergence`, `redi_energy_flux_divergence` | Integrated conservative energy **inflow**, cell × layer, J |
| `gm_local_energy_residual`, `redi_local_energy_residual` | Local full-H change minus energy inflow, J |

Run `eddy_campaign.py --front density|cold|warm --eddy-energy .1 --output FILE`.
See [reservoir verification](../../verification/ocean/gmredi/ENERGY_RESERVOIR.md)
for funded GM, energy-consuming and energy-releasing Redi, local budgets,
whole-model composition and timestep refinement.

No long-run climate, GPU, float32, compiled-kernel or 16384×40 performance claim
is made. Positivity-preserving Redi, exact neutral geometry, boundary tapering,
spatial convergence and energy-dependent coefficients remain open work.

## Evaluation cost

The optimization pass shares geometry only within a fixed-h/u call, rebuilds
Redi slopes after GM, and avoids large redundant stencil/conjugate tensors.
Validation of unchanged fields is reused; reservoir and funding checks remain
explicit after each exchange. See [equivalence and timing](../../verification/ocean/gmredi/optimization/README.md).
Algebraic regrouping can change roundoff, but no physical coefficient or limiter
policy changes. No model/state cache survives into a later call.

The [factored Redi implementation](../../verification/ocean/gmredi/factored/README.md)
now computes vertical derivatives per column, averages them onto active edges,
and applies the matching transpose for both tracer and Hamiltonian energy flux.
It avoids the eight-entry edge/tracer expansion while retaining the same
mathematical stencil, explicit bound and local funding checks.

## Optional energy feedback and dissipation

`GMRedi(kappa_redi=..., energy_closure=EddyMixingLength(length=..., efficiency=...,
kinetic_fraction=...))` replaces prescribed GM with `c L sqrt(2 f E_edge)`.
Do not also specify nonzero `kappa_gm`. All parameters are explicit; Redi keeps
its separate coefficient and the original funding/CFL guards remain active.
`EddyDissipation(timescale=...)` is a separate exponential decay-to-heat stage.
It reuses the exact Roquet full-H heat solve, not heat/cp, and leaves turbulent
TKE unchanged. See [feedback equations, usage, tests and timing](../../verification/ocean/gmredi/feedback/README.md).
These closures are uncalibrated and opt-in; see the optional backscatter stage below.

## Optional deterministic backscatter

A first [reservoir-funded rotational negative-viscosity stage](BACKSCATTER.md)
is now available as `EddyBackscatter(viscosity=..., timescale=..., fraction=...)`.
Configure `RoquetOcean(..., backscatter=...)` to pair its momentum and reservoir
tendencies within each resolved RK stage. GM/Redi and decay remain separately
composed; their paths and defaults are unchanged.
