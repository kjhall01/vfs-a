# Stratified ocean API

`StratifiedOcean` combines active JMD95 density, entropy/salinity transport,
checked SSPRK3 and an optional adaptive energy-closing vertical remap.

## Public API and ownership

```python
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean, AdaptiveThermalRemapper

model = StratifiedOcean(mesh, column, domain=domain,
                        remapper=AdaptiveThermalRemapper(method='plm'))
state = model.state(u_e=u, h=h, temperature=theta, salinity=sp, tracer=q)
state, report = model.step_with_report(state, dt)
target_h = state.h.sum(-1, keepdim=True) * fractions
state, remap_report = model.remap_with_report(state, target_h)
fields = model.physical_fields(state)
budgets = model.budgets(state)
```

`mesh` and `column` are explicitly constructed and borrowed. State tensors are
unbatched: velocities [edge, layer], thickness/entropy/salt contents [cell, layer],
optional passives [cell, layer, tracer]. Temperature input is potential temperature
in degrees Celsius on IPTS-68; salinity is practical salinity. Conversion from
`OceanState` is explicit through `from_temperature`. The native state stores
`entropy_h`, not temperature content. `physical_fields` explicitly inverts that
coordinate. No xarray members, inferred model type or hidden clipping are used.

The facade owns an existing geometry-checked Ocean/JMD95 composition, the bundled
surface reference and optional remapper. It uses the sharp-front checked SSPRK3
path, not the legacy centered tendency. `step_with_report` returns three stage
ledgers and an optional post-step remap ledger. An explicit `target_h` passed to
it must preserve the depth of the post-step columns; constructing the target
after stepping, as above, makes that requirement clear. No automatic grid schedule
is selected. Salt, entropy, volume and passive budgets use wet-cell mass weights;
energy terms are in joules, entropy in J/K, volume in m³, salt in kg times practical
salinity, and tracer inventories in kg times tracer concentration.

Run `python -I examples/ocean_stratified.py` after installing a wheel. The example
uses only installed public package imports, advances three steps and performs a
real remap. It is a usage demonstration, not a physical-accuracy benchmark.

## Thermodynamics and remapping

JMD95 supplies density. A bundled surface Gibbs fit supplies the caloric reference,
using reference-composition salinity; this is not full TEOS-10. GSW is not required
at runtime. The fit's domain, calibration and accuracy are documented in
[seawater calorics](../verification/ocean/SEAWATER_CALORICS.md).

The adaptive remapper first projects potential enthalpy, salt, passives and edge
momentum. It then applies layer-dependent temperature adjustments to close the
column Hamiltonian while checking entropy nondecrease within roundoff. Rank,
thermal-domain or nonlinear failures reject explicitly. The target grid is a
caller choice. See [the formulation and measurements](../verification/ocean/ADAPTIVE_THERMAL_ALE.md).

## Supported scope and unresolved limitations

The API supports unforced, unbatched CPU float64 configurations. GPU, compile,
long-backward, performance and long climate integrations are unqualified.

- Strictly positive thickness and 0 < SP < 42; temperature strictly inside
  -2.5 to 40 C ITS-90 after scale conversion. Invalid stages reject, without floors
  or automatic timestep reduction. Zero passives and absent passives are allowed.
- No surface forcing, vertical physical mixing, convection, open boundaries,
  wetting/drying, TKE/EKE or atmospheric coupling in this facade.
- Semidiscrete work closure does not imply exact SSPRK3 energy conservation.
  Remapping corrects only its own energy change, not timestep drift.
- Sharp-front active local bounds are conditional on the kinetic-work budget;
  temperature extrema are not guaranteed. Numerical entropy production is not a
  calibrated physical mixing closure. Smooth observed order was about 1.6–1.7.
- Adaptive remapping is feasibility-seeking, not globally optimal. Rank deficiency,
  EOS limits or nonlinear failure reject explicitly. No fallback silently changes
  salt, momentum or the target grid. It does not preserve potential enthalpy
  separately after full-H correction. Entropy nondecrease is within roundoff.
- Latest adaptive wave evidence is only .05 period. Longer runs of other configurations do
  not establish the adaptive remapper’s behavior. Repeated near-neutral resets/front cycles remain
  needed. Perturbation-energy diagnostics must avoid subtracting huge backgrounds.
- Rotation/kinetic reconstruction limitations and fine-mesh balance measurements
  remain unchanged.
- Derivatives are piecewise through active-set/iteration changes; existing smooth
  directional checks do not establish gradients across all switching surfaces.


See [ocean verification](../verification/ocean/README.md) for measurements and reproduction.
