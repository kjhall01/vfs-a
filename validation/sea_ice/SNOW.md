# Zero-layer snow on moving sea ice

`SnowOceanIce` adds accumulation, transport, redistribution, melt, insulation and
albedo to the existing coupled ocean/category-ice model. It is opt-in research
code. Existing model defaults and numerical kernels are unchanged.

## State and energy ownership

```python
snow = SnowOceanIce(coupled)  # coupled is an OceanIce instance
state = snow.bare(existing_ridging_state)
state, report = snow.step(
    state, dt,
    snowfall=snowfall,          # [cell,1], kg/m²/s per ice-covered area
    air_temperature=Tair,      # [cell,1], degrees Celsius
    shortwave=SWdown,           # [cell,1], W/m², nonnegative
    wind_velocity=wind_xyz,     # [cell,3], tangent m/s; optional
)
```

`SnowState` contains `coupled: RidgingState` and `snow_mass[cell,category]`.
**The coupled state's `ice.volume` now means total frozen water in ice-equivalent
m³/m², including snow.** `snow_mass` labels a subset of that mass; it is never
added a second time to mass, latent energy, loading or kinetic energy. The
underlying ice volume is `ice.volume - snow_mass/rho_i`.

The same existing `RidgingHamiltonian` supplies energy:

```python
H = snow.hamiltonian.total_energy(state.coupled)
```

Snow and ice have the same zero-layer latent-energy datum. Neither stores
internal sensible heat. Mechanically, both use the existing ice-equivalent
hydrostatic geometry. Snow density controls its diagnostic porous depth and
thermal resistance, **not a separate porous-snow gravitational energy**. Thus the
budget closes for the declared approximate H, not for a resolved layered snow
column. This explicit approximation retains the existing energy-paired forces.

Never initialize a snow label on top of unchanged ice stores and interpret it
as added mass: that relabels some existing solid water. Use `accumulate` to add
snow or initialize total frozen volume and its snow subset consistently.

## Accumulation and its boundary ledger

Snowfall intercepted by category k adds

```
delta snow_mass_k = dt * snowfall * area_k
 delta volume_k   = delta snow_mass_k / rho_i
```

The forcing is per ice-covered area: snowfall on open water is **not** handled
by this component. An atmospheric/ocean forcing driver must account for that
remaining fraction separately. Nothing is silently redirected or accumulated
on dry cells. This is a snow-on-ice interface, not complete precipitation forcing.

Snow arrives co-moving with the frozen cover at the solid enthalpy datum.
Its external energy comprises `-Lf * added_mass` plus gravitational incorporation
work and the incoming kinetic energy. `accumulate` analytically integrates that
mass conjugate along the addition path; it does not diagnose and repair a global
energy residual. Incoming horizontal momentum is the added kinetic metric times
the existing ice velocity. Cold-snow sensible heat, falling-speed impact energy
and atmospheric removal of this boundary mass/energy are not modeled here.

`SnowReport.external_energy` includes precipitation incorporation, atmospheric
heat and wind work. `snowfall_mass`, `snowfall_energy` and `melted_snow` provide
separate integrated diagnostics. All are tensors. A negative precipitation
energy is expected because the solid-water latent-energy datum is negative.

## Conduction, radiation and melt

For category area a, solid-equivalent volume V and snow mass M_s:

```
h_ice  = (V - M_s/rho_i) / a
h_snow = M_s / (rho_s a)
R = h_ice/k_ice + h_snow/k_snow
f_snow = h_snow/(h_snow + h_patch)
alpha = alpha_ice + f_snow*(alpha_snow - alpha_ice)
```

The baseline snow density is 330 kg/m³ and conductivity 0.30 W/(m K), following
common [Icepack constant-property values](https://cice-consortium-icepack.readthedocs.io/en/main/science_guide/sg_thermo.html).
Albedos (snow 0.85, bare ice 0.60, water 0.06) and the 0.02 m patch-depth scale
are explicit illustrative parameters, not a calibrated spectral radiation model.
Snow cover smoothly increases reflectivity as depth grows. Absorbed and reflected
shortwave sum to incident shortwave over the wet area. All absorbed shortwave
enters the zero-layer surface budget; no penetrating radiation is represented.

For atmospheric conductance c and basal freezing temperature T_f, the surface
balance is solved algebraically:

```
T_surface = min(0, [T_f + R*(c*T_air + (1-alpha)*SW)] / [1+c*R])
Q = c*(T_air-T_surface) + (1-alpha)*SW
```

Covered-category Q and open-water exchange are area weighted. At subfreezing
surface temperatures this balances conduction through the snow/ice resistance;
at the melting constraint, the remaining heat is available for phase change.
The 0°C cap is a physical surface condition, not clipping of prognostic stores.
Alternatively, prescribe `heat_flux` directly; it is mutually exclusive with
air temperature/shortwave and bypasses this surface-feedback calculation.

The same full-H phase solve changes total frozen mass and liquid thickness.
Snow is removed first on a melting branch. While only snow melts, underlying
ice volume and footprint stay fixed; once snow is exhausted, the existing ice
melt geometry resumes. Freeze growth creates ice, never snow. Meltwater goes
immediately to the liquid ocean; liquid salt content is retained, so dilution
is represented. The existing donor-momentum transfer routes inelastic KE loss
to ocean TKE. No additional meltwater is added by the snow bookkeeping.

This is one zero-layer column equilibrium, not separate basal and surface melt
fronts. Preferential snow removal applies to net melting regardless of heating
origin. Melt ponds, refreezing in snow, snow-to-ice flooding, density evolution,
age/wetness-dependent albedo, brine, snow enthalpy and blowing-snow loss remain
outside scope. Those need additional state and energy terms.

## Movement and stepping

Snow mass uses the same donor velocity and category transfer fractions as its
frozen carrier. Ridging transfers snow with the donor mass; compaction changes
area while retaining snow. SSPRK3 stages advance both together and reject
negative snow or snow exceeding total solid mass. No positivity clipping occurs.
Phase/drag half steps surround dynamics and existing optional vertical mixing;
snowfall is deposited in half steps at the two endpoints. Existing backscatter
remains in the coupled tendency. Step error is visible in H minus boundary work.

## Reproduction and evidence

```sh
python -I validation/sea_ice/run_snow.py --cells 32 --hours 12 --dt 120 \
  --verbose --output snow-cycle
python -I validation/sea_ice/run_snow.py --cells 128 --hours 12 --dt 120 \
  --coast --patch --output snow-coast
python -I validation/sea_ice/run_snow.py --no-snow-feedback --output snow-control
```

The first half supplies cold air and snowfall; the second supplies warm air and
sunlight. The control retains snow mass and latent energy but gives it compact
ice's conductivity, density and albedo, isolating the surface feedback.
Outputs include per-step energy/water/salt errors, snow mass and depth, melt,
albedo and shortwave absorption, with source hashes and timing in JSON.
See [verification](../../verification/ocean/sea-ice/snow/README.md).
CPU float64 and selected smooth-branch backward sensitivities are tested;
GPU, compilation, float32 and long-run climate behavior are not qualified.
