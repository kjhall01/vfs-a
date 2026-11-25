# Coupled campaigns and surface exchange

Use [the Earth runner](EARTH_SHELF.md) for an atmosphere, ocean, sea ice,
snow and land run with topography and partial ocean columns. Use the
[aquaplanet campaign](CAMPAIGN.md) for an all-ocean planet. Both use
[campaign.py](campaign.py) for time integration, diagnostics and NetCDF
restarts. The Earth runner supplies the Earth preset and locates its inputs.
The [device guide](GPU.md) covers CPU and accelerator configurations.

At each physics interval, [exchange_step](driver.py) applies one
[exchange record](../exchange.py) to the atmosphere and the surface. On
land, the [coastal receiver](coast.py) supplies the land skin temperature,
advances the land stores and routes runoff to wet cells. The campaign also
schedules ocean mixing, GM–Redi, eddy-energy decay, backscatter and ALE
according to its YAML configuration.

## Small all-ocean demonstration

The standalone `run.py` example uses a prescribed 270 K ice skin. The
exchange contract below describes that example. The campaign adds prognostic
ice temperature and cold content.

With the separately installed VFS wheel and pinned PyAM21 dependency:

```bash
python -I workflows/coupled/run.py --source /path/to/PyAM21 \
  --output coupled-check --hours 6 --ice --dynamics --dt 60 --dynamics-dt 15 --verbose
```

Without `--dynamics`, the test advances column physics and the shared boundary;
it does not advance horizontal dynamics or the internal ice phase operator.
With `--dynamics`, the driver first advances bounded atmospheric SSPRK3 and the
existing ocean–ice–snow integration, then performs one common exchange. This is
first-order sequential splitting. No second prescribed heat/wind/freshwater
forcing is added to the surface integrator. Radiation remains off unless
`--radiation-data /path/to/sea_esf` is supplied.

The example uses 32 cells, 24 atmospheric levels, three ocean layers and idealized
polar ice. Ocean SST is approximated as `CT + 273.15 K`; this is not a TEOS in-situ
surface-temperature conversion. Initial conditions are analytic, not observations.
The output reports combined full-H, water and salt, upstream AM2 residual, local
transfer residuals, SST, snow and ice mass and velocities at every step.

## Contract

| Quantity | Units | Positive direction |
| --- | --- | --- |
| `vapor[C,3]` | kg/m² per interval | surface tile → atmosphere; signed |
| `rain[C]`, `snow[C]` | kg/m² per interval | atmosphere → surface; nonnegative |
| `sensible[C,3]` | J/m² per interval | tile → atmosphere |
| `stress[C,3,3]` | N s/m² | impulse on atmosphere; last axis Cartesian |
| `sw_surface`, `lw_surface` | J/m² | radiation into surface |
| `toa` | J/m² | radiation into combined system |
| `terms` | J/m² | named atmospheric energy contributions |

Every amount is per **whole cell area**, including tile fluxes. Tiles are open
water, thin ice and ridged ice, with fractions summing to one. Surface vectors
are tangent. Inputs must match the mesh, dtype and device; exhausted water stores
reject rather than reducing the requested flux. The first receiver requires an
identical, all-wet mesh and equal gravity (configured to 9.8 before initialization).
There is no remapping between component grids, land model or runoff in this driver.

The public upstream surface routine is called for each tile with its own skin
T and moving velocity. Ice uses the upstream saturation lookup without seawater's
salinity reduction. Roughness is explicitly shared with the current AM2 ocean
surface state in this first mosaic. Surface fluxes and derivatives are area
weighted for one implicit bottom-air solve. Each tile then receives that **same**
implicit air increment; its post-implicit fluxes sum to the actual applied AM2
fluxes. Turbulence receives weighted bulk diagnostics. This is a declared mosaic
approximation, not a reproduction of the full GFDL sea-ice surface package.

Radiation sees area-mean albedo and an emission-equivalent skin temperature
`(sum(f*T**4))**0.25`. Snow depth changes ice albedo. The combined surface receives
the net radiation; spectral ocean penetration and category-specific radiative
skin heating are not resolved by this boundary. Crucially, both components use
identical cached bottom interface fluxes. Re-emitting at a new SST between
radiation calls would create a mismatch, so net LW is `-flxnet_bottom` until the
next call. Downward-LW diagnostics remain those of the upstream radiation call.

## Water, momentum and energy

Vapor carries donor temperature and velocity: the tile for evaporation or
sublimation, the old lowest air layer for condensation/deposition. Rain and snow
use the old lowest air temperature/velocity as explicit assumptions; AM2 does
not diagnose precipitation temperatures here. All water enthalpies use the VFS
atmospheric mixture datum. Liquid has `cp_l*(T-Tref)`, ice additionally has `-Lf`,
and vapor has `Lv + cp_v*(T-Tref)`. Transported potential and kinetic energy are
included once. This contract reconciles the components' caloric references; it
does not make their distinct approximate equations of state identical.

Rain passes into liquid even beneath ice. Snow is intercepted by old ice cover;
open-water snowfall enters liquid with its frozen enthalpy. Sublimation consumes
snow first, then underlying ice; deposition on ice joins snow. Category volume
already includes snow mass. Liquid thickness changes, while salt, passive tracer,
TKE and EKE **contents** stay in the liquid. No salt dilution approximation is used.

Cell water/stress impulses use the exact transpose of the shared kinetic-weighted
cell-wind reconstruction on both sides. This external map weights observations
by native edge kinetic-energy ownership, avoiding short-edge impulse amplification.
Reciprocal impulses therefore cancel in the native
generalized edge momentum. This is a discrete work/impulse identity, not a new
claim of global Cartesian or angular-momentum conservation on the spherical mesh.
Water first changes mass and momentum. The subsequent surface stress kick uses
its exact native midpoint work; the atmosphere receives its negative. AM2's
stationary-surface dissipative heating is adjusted by this declared transferred
work, preserving its own residual. Finite water-incorporation/impact energy is
accounted for in the local thermal first law; ocean–ice internal drag still
returns its loss to ocean TKE through the existing operator.

The atmospheric layer solve transfers AM2's energy increment plus the difference
between AM2's boundary convention and this shared contract. The ocean receiver
solves its analytic local thermal energy at the new mass, salt and momentum for
exactly the **negative shared surface transfer plus absorbed radiation**. It does
not inspect the atmospheric final energy or cancel its residual. Thus:

`Δ(H_air + H_ocean+ice) = TOA + AM2_residual + transfer/time-discretization errors`.

A deliberately injected upstream heat defect verifies that it remains visible.
There is no global energy rescaling, package monkeypatch or forward autograd.
The surface thermal increment can be negative; inadmissible thermodynamic roots
reject. Heat arriving through ice is represented in the common ocean–ice energy,
then redistributed by the existing phase operator on the next split step. For the prognostic ice reservoir and combined restart format,
see the [campaign guide](CAMPAIGN.md).

See [verification](../../verification/atmosphere/pyam21/coupled/README.md).
