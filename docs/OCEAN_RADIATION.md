# Ocean radiation and prescribed heating

`OceanSurfaceHeating` applies prescribed non-solar surface heat and penetrating
sunlight to the active JMD95/entropy ocean. It can operate directly on a
`StratifiedOcean` state or compose with `TKEOcean`. No atmospheric radiation
solver, bulk air–sea exchange or freshwater flux is implied.

```python
from vfs.ocean.stratified import OceanSurfaceHeating

heating = OceanSurfaceHeating(
    first_band_fraction=0.6, attenuation_lengths=(0.5, 20.), bottom='export')
state, report = heating.apply(ocean, state, dt,
    shortwave=net_solar, surface_heat=net_longwave)
```

Fluxes are tensors `[nCells]`, W/m², matching the state's dtype/device. Positive
means energy entering the water. `shortwave` must be nonnegative and already
net of surface reflection; no albedo is applied again. `surface_heat` is signed
non-solar heating deposited in the top layer. It can contain prescribed net
longwave or a supplied sum of non-solar heat fluxes, but must exclude shortwave.
Omitting either input gives zero. Inputs are held fixed for the substep; use
interval-mean fluxes when available. Dry-cell inputs are masked to zero.

The illustrative optical parameters above are explicit choices, not fitted
VFS coefficients. Both attenuation lengths must be positive and the first-band
fraction must lie in [0,1]. This implementation neither diagnoses water type nor
uses chlorophyll, solar angle, scattering or spectral atmospheric transfer.

## Absorption and bottom transmission

For depth d measured downward from the instantaneous sea surface,

\[
F(d)=F(0)[a e^{-d/\ell_1}+(1-a)e^{-d/\ell_2}].
\]

Interface depths are cumulative **current layer thicknesses**, including after
an ALE remap. Layer k receives F(d_top)-F(d_bottom). The implementation evaluates
this difference with `expm1` for thin layers, rather than subtracting nearly
equal exponentials. No vertically uniform heating or layer-centre approximation
is used. Integrated absorption is independent of how a fixed water depth is
partitioned into layers.

- `bottom='export'` (default) reports transmitted sunlight as leaving the water
  energy budget. It is not silently discarded from the accounting.
- `bottom='absorb'` deposits that remainder in the lowest water layer. This is an
  explicit instantaneous-return approximation; it does not model sediment heat
  storage, reflection or a separate seabed temperature.

## Energy update

Let A be cell area, Q_k the layer heat input in W/m², and dt the forcing duration.
At fixed layer mass, salinity, passives and velocity, the entropy update solves

\[
\Delta H_k=A\,\Delta t\,Q_k.
\]

H includes the existing JMD95 dynamic enthalpy and reference thermal energy.
The solver uses their analytic temperature derivative with temperature-path
quadrature, then independently checks the actual layer Hamiltonian change.
It shares the TKE heat inversion machinery, with a signed branch for external
heating/cooling. A zero-increment entropy baseline preserves both exact zero
forcing and the linear response there. Existing TKE dissipation retains its
original nonnegative-heating numerical path.

Cooling can decrease ocean entropy: the ocean is exchanging energy with an
external reservoir. There is no imposed entropy-nondecrease check on this
external flux. Out-of-domain temperatures or failed energy checks reject without
clipping or input mutation. No latent mass loss, precipitation or salinity
change accompanies a supplied heat flux; those require a future freshwater
exchange operator.

`SurfaceHeatingReport` energies are joules: surface input, incident shortwave,
bottom transmission/export/absorption, deposited layer heat, and global
`external_heat`. It also reports optical flux closure, inversion residual,
independent layer Hamiltonian residual and their global sum. Actual energy
checks scale with the thermal background, so they do not guarantee relative
accuracy for vanishingly small heat increments.

## Composition

```python
from vfs.ocean.stratified import TKEOcean, TKEClosure

model = TKEOcean(ocean, closure=TKEClosure(2.), heating=heating)
state = model.state(resolved_state, tke=k)
state, report = model.step_with_report(state, dt,
    shortwave=net_solar, surface_heat=net_longwave)
print(report.heating.external_heat)
```

`heat_with_report` applies only heating to a TKE state and leaves TKE unchanged.
The full step order is dynamics, optional ALE, boundary stress, surface heating,
then vertical TKE physics. Splitting remains first order. Existing unforced
constructors/defaults are unchanged; passing heat fluxes without configuring
heating raises. Heating changes stratification, which can affect subsequent
mixing, but does not directly inject TKE or remove the closure's need for a
seed/source when initialized at zero.

Run [the example](../examples/ocean_radiation.py) for prescribed sunlight and
net longwave cooling with varying depth and per-layer TKE. See
[qualification](../verification/ocean/RADIATION.md) for CPU float64 scope,
analytic optical checks and remaining limitations.
