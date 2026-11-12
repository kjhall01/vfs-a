# Prescribed ocean surface exchange

`OceanSurfaceExchange` applies supplied freshwater, momentum and energy fluxes
to a `StratifiedOcean` state. It calculates no atmospheric transfer coefficients,
evaporation rate, precipitation rate or radiation. `OceanSurfaceFluxes` is a
small tensor-only container for interval-mean inputs.

| Field | Layout | Units and sign |
|---|---|---|
| `precipitation` | `[cell]` | kg/m²/s, nonnegative freshwater addition |
| `evaporation` | `[cell]` | kg/m²/s, nonnegative freshwater removal |
| `sensible_heat` | `[cell]` | W/m², positive into water |
| `latent_heat` | `[cell]` | W/m², positive into water; evaporative cooling is negative |
| `water_energy_flux` | `[cell]` | W/m², signed net energy transported with water |
| `water_momentum_flux` | `[edge]` | Pa, signed native edge-normal advected momentum flux |

Tensors must match state dtype/device. Dry cells and closed coast edges receive
no flux. Omitted heat/rate fields mean zero. With nonzero precipitation or
evaporation, **both water-carried fluxes must be supplied**, even if the two
rates balance. Explicit zero tensors mean zero transfer; they are not inferred
rain properties. Water-carried fluxes without local water exchange reject.

```python
from vfs.ocean.stratified import OceanSurfaceExchange, OceanSurfaceFluxes

exchange = OceanSurfaceExchange()
fluxes = OceanSurfaceFluxes(
    precipitation=P, evaporation=E,
    sensible_heat=Q_sensible, latent_heat=Q_latent,
    water_energy_flux=F_water_energy,
    water_momentum_flux=F_water_momentum)
state, report = exchange.apply(ocean, state, dt, fluxes)
```

## Energy convention

`water_energy_flux` is the advective/boundary-work energy input expressed in the
**same reference as the ocean's resolved Hamiltonian**, including the relevant
thermal and mechanical energy carried by the transferred water. It excludes
heat already supplied through sensible, latent or radiative terms. In particular,
if vapor energy already includes the vaporization latent contribution, that
contribution must not also be supplied as latent cooling. A future atmosphere
adapter must translate its enthalpy/reference convention to this contract.
The operator does not guess this conversion from a rain temperature or silently
multiply evaporation by a latent-heat constant. There is no implied thermodynamic
consistency between independently supplied rates and heat; the caller owns it.

Wind stress remains a separate force accepted by `OceanBoundaryStress`.
`water_momentum_flux` is momentum transported by water itself. For example,
evaporation carrying the old native current would use -I(E)*u_top, whereas
stationary precipitation adds zero momentum. These are possible prescribed
choices, not defaults calculated by this operator.

## Conserved update

For cell rates P and E, reference density rho0 and duration dt,

\[
h_t'=h_t+\Delta t(P-E)/\rho_0.
\]

This changes actual Boussinesq water volume and free-surface height. Salt and
all existing passive tracer **contents** remain fixed: they are treated as
nonvolatile with zero boundary flux. Rain dilutes concentrations; evaporation
concentrates them. Lower-layer thickness and scalar contents stay unchanged.
Condensation may be represented as freshwater addition with a consistent signed
latent heat input. There is no ice, runoff composition or volatile-tracer model.

Native top-edge momentum follows

\[
M_t'u_t'=M_tu_t+\Delta t A_e F_{p,e},\qquad
M_t=\rho_0 A_e I(h_t),\quad A_e=dc_e\,dv_e.
\]

Zero prescribed impulse therefore preserves momentum, not velocity, when mass
changes. The original bottom topography remains fixed. A provisional state
retains the original temperature after changing mass, salinity and velocity.
A signed top-layer thermal solve then imposes the cell energy target

\[
\Delta H_c=A_c\Delta t(Q_{\rm sensible}+Q_{\rm latent}+F_{\rm water\ energy}).
\]

The correction includes the actual changes in free-surface energy, kinetic
energy, JMD95 dynamic enthalpy and reference thermal energy. Native shared-edge
kinetic energy uses the existing Hamiltonian cell allocation. Consequently,
changed edge momentum can change the thermal correction in adjacent cells;
it is not an assertion of mechanically isolated columns. The correction and
each provisional energy component are reported, never hidden as an extra flux.
There is no global redistribution or forward automatic differentiation.

Inadmissible thickness, salinity or temperature rejects without clipping or
input mutation. Strong evaporation may require a shorter step; wetting/drying
and complete layer removal are absent. Cooling/exchange can decrease ocean
entropy, since this is an open boundary, not an isolated mixing step.

## Reports and composition

`SurfaceExchangeReport` contains added/removed water mass (kg), mass residuals,
native edge impulse and momentum residuals (N s), separate sensible/latent/water
energy inputs (J), provisional column Hamiltonian changes, thermal adjustment,
inversion residual and independent column/global energy residuals. The extensive
salt/passive arrays are retained directly. Energy checks use thermal-background
roundoff scales, not a promise of relative accuracy for tiny flux increments.

```python
from vfs.ocean.stratified import TKEOcean
model = TKEOcean(ocean, exchange=exchange)  # optional closure/heating/boundary
state = model.state(resolved_state, tke=k)
state, report = model.step_with_report(state, dt, surface_fluxes=fluxes)
```

`exchange_with_report` runs only this operation. The full step order is resolved
dynamics, requested ALE, water/heat exchange, wind/bottom stress, radiation,
then TKE mixing/dissipation. Processes remain first-order split. The TKE wrapper
retains **extensive h*k** through water exchange: specific k changes with h, but
no unresolved TKE crosses this boundary. Supplied water energy refers to the
resolved Hamiltonian; a TKE-carrying boundary is not implemented.

For radiation use `shortwave` and `surface_heat` on `OceanSurfaceHeating`.
Do not supply the same sensible/latent flux there and in `OceanSurfaceFluxes`.
Defaults remain unchanged; configuring exchange alone applies no forcing.

Run [the prescribed-flux example](../examples/ocean_surface_fluxes.py). It uses
synthetic supplied fluxes, not an atmospheric bulk law. See
[qualification](../verification/ocean/SURFACE_EXCHANGE.md) for evidence and scope.
