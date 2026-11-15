# Freshwater boundary for moving ocean, ice and snow

`FreshwaterBoundary` and the optional `freshwater=` argument of `SnowOceanIce.step`
compose prescribed rain, snowfall, evaporation, sublimation and river runoff
with the existing shared Hamiltonian. Packaged ocean kernels/defaults are unchanged.
The existing `snowfall=` argument retains its previous covered-area-only behavior.

## Inputs and routing

All cell fields are tensors shaped `[cell,1]`; incoming velocity is `[edge,1]`.
They must match the state's dtype/device. None means the process is absent.
Nonzero liquid inputs require explicit enthalpy and velocity, including explicit
zeros for stationary water at the zero thermal datum.

| Field | Units and interpretation | Recipient |
| --- | --- | --- |
| `rain: LiquidWater` | rate kg/m²/s of horizontal wet-cell area | Liquid ocean, including immediate drainage through ice |
| `snowfall` | kg/m²/s over the entire wet cell | Frozen cover receives its area fraction; the rest enters liquid with solid enthalpy |
| `evaporation` | kg/m²/s **per open-water area** | Removes liquid water at its native donor velocity |
| `sublimation` | kg/m²/s **per covered area** | Removes snow first, then underlying ice, at donor ice velocity |
| `runoff: LiquidWater` | kg/m²/s of receiving cell area | Liquid ocean, even beneath ice; no ice-fraction multiplier |

Atmospheric forcing over dry cells is outside this component and contributes no
ocean flux. Runoff into dry cells rejects rather than silently discarding river
water. Rates must be nonnegative; the direction is determined by the process.
Do not supply a grid-mean evaporation/sublimation field as a per-fraction rate.

```python
from freshwater import FreshwaterBoundary, FreshwaterFluxes

boundary = FreshwaterBoundary(snow_model)
rain = boundary.liquid(rain_rate, rain_temperature, rain_edge_velocity)
river = boundary.route_runoff(
    state,
    discharge,          # [river], m³/s
    river_temperature,  # [river], model temperature, °C
    receiving_cells,   # int64 [river,slot], explicitly selected wet cells
    weights,           # [river,slot], nonnegative, each river sums to one
    river_edge_velocity,
)
fluxes = FreshwaterFluxes(
    rain=rain, snowfall=snowfall, evaporation=evaporation,
    sublimation=sublimation, runoff=river,
)
state, report = snow_model.step(state, dt, freshwater=fluxes, heat_flux=Q)
# Q excludes latent cooling already carried by evaporation and sublimation.
```

`liquid` converts temperature to the model's idealized thermal datum `cp*CT`.
Alternatively construct `LiquidWater(rate, enthalpy, velocity)` with thermal
specific enthalpy in J/kg in that same reference. This field excludes the
mechanical incorporation energy, which the boundary computes separately.
This is a declared model convention, not an exact TEOS-10 freshwater enthalpy.

`route_runoff` is a sparse conservative discharge adapter, not a land model or
automatic river-mouth locator. The caller supplies receiving cells; coastal
mapping can be prepared outside the timestep. Multiple rivers may share cells.
The adapter conserves both mass and incoming thermal energy:

```
sum_cells A * R       = rho0 * sum_rivers Q
sum_cells A * R * hin = rho0 * cp * sum_rivers Q * Tin
```

Weights are checked, never silently normalized. Invalid or dry recipients reject.
The helper allows any explicit wet receiver, including a cell covered by ice.

## Water and salt

For coverage a and a wet cell, the external mass changes per horizontal area are

```
delta liquid = dt * [rain + runoff + (1-a)*snowfall - (1-a)*evaporation]
delta frozen = dt * [a*snowfall - a*sublimation]
```

Snow remains a labelled subset of frozen ice-equivalent mass, never a second
mass store in the Hamiltonian. Its melting/freezing transfers are internal and
cancel from the combined water budget. The freshwater stage retains salt and
passive-tracer contents exactly; salinity changes through actual top-layer
thickness. Extensive TKE and mesoscale-energy contents remain in the ocean.
There is no salt, passive tracer, TKE or mesoscale-energy boundary flux in this pass.

Rain on ice drains immediately without pond storage or a separate contact-freezing
model. Open-water snow supplies mass with enthalpy `-Lf`; its melting requirement
therefore draws heat from the ocean. The existing phase stage subsequently handles
freezing/melting. Sublimation consumes snow before ice, preserving footprint during
snow-only loss; underlying ice loss shrinks each category's footprint at fixed
underlying-ice thickness. It never supplies meltwater to the ocean.

## Momentum and energy

Rain/runoff carry their supplied native velocity. Snow arrives co-moving with
its recipient, preserving the earlier snow-on-ice convention; snow landing in
water carries the local ocean velocity. Vapor leaves at donor velocity. These
are explicit boundary assumptions, not inferred falling-water trajectories.
Native edge impulses update momentum using the changed mass metric. Incoming
kinetic energy is also recorded; inelastic incorporation contributes to the
reported local thermal adjustment, not an unreported TKE source.

The exact hydrostatic work of liquid incorporation is, per cell,

```
A*g*[dm*(eta + dh/2) + m_frozen*dh],   dh = dm/rho0
```

The last term lifts or lowers overlying ice and snow. It must not be omitted
from a coupled calculation. Solid precipitation/removal also has explicit latent,
slab gravitational, category-variance and kinetic energy increments.

Evaporation removes donor thermal enthalpy plus constant `Lv=2.5e6 J/kg`.
Sublimation applies cooling `(Lv+Lf)*removed_mass` alongside removal of the solid
store at datum `-Lf`; the resulting vapor thermal datum is `Lv` for zero-layer
solid. Snow/ice sensible heat and temperature-dependent vapor enthalpy are absent.
These are prescribed rates with an explicit constant latent-heat approximation,
not humidity-dependent bulk transfer laws. Do not also supply their latent cooling
through `heat_flux`, a radiation field, or an external surface-exchange object.

After the declared mass/momentum changes, a cell-local analytic Roquet thermal
solve enforces the supplied change of **full ocean+ice+snow H**. It includes the
salinity/dynamic-enthalpy change, free surface, ice loading, ice/category energy,
and both kinetic energies. The resulting thermal adjustment and independently
evaluated residual are reported. There is no global rescaling, forward autograd,
state clipping, hidden retry, or altered dynamical tendency.

The finite boundary update is separate from the liquidus equilibrium stage and
may leave a small off-liquidus state at a split-step endpoint. Its donor properties
are evaluated at the start of the boundary stage. Exact boundary energy accounting
does not establish a higher-order coupled trajectory or exact finite-step dynamics.
Refine the timestep when judging both budgets and physical trajectories.

## Reports, rejection and verification

`FreshwaterReport.inputs/outputs` contain extensive cell masses; `energy_inputs`
contains signed cell energies for each thermal/mechanical contribution. It also
reports global external energy, ocean/ice impulses, water/momentum residuals,
thermal adjustment and the independently checked full-H cell energy residual.
`SnowReport.freshwater` contains the two boundary half-step reports, and
`SnowReport.external_energy` includes them along with the usual heat/wind work.
The old snow-specific diagnostics still refer only to snowfall intercepted by ice.

Use `boundary.apply(state, dt, fluxes)` to isolate the boundary operation. The full
step uses boundary / phase / drag / dynamics / mixing / drag / phase / boundary.
Supplying both legacy `snowfall=` and `freshwater=` rejects to prevent double counting.
Top-layer exhaustion, out-of-domain salinity/temperature, excessive sublimation,
invalid forcing and inadmissible roots reject without mutating the input state.

With the VFS wheel installed, run:

```bash
python -I validation/sea_ice/run_freshwater.py --coast --hours 6 --dt 120 --output freshwater-coast
```

See the
[verification record](../../verification/ocean/sea-ice/freshwater/README.md).
This pass does not add land hydrology, automatic river routing, river plumes,
ice-shelf discharge, trapped salt/brine, melt ponds, or full atmospheric bulk fluxes.
