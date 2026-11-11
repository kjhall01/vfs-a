# Supported stratified-ocean integration

This is an opt-in, unforced integration of selected research candidates, based on
PR #42 at `1b31978cd6c447e9090973ff36628a6c44af3b15`. Reviewed GitHub main is
`5841cda8b4585418b18f0ba0486315870b1a59b2`. PRs #28–31 are merged; #32–42 remain
unmerged. This integration does not merge or approve that entire stack.

## Selection review

| PR | Decision | Reason |
| --- | --- | --- |
| #28–30 | Retain evidence | Rotation consistency and transport restrictions inform scope; no unsubstantiated rotation fix |
| #31 | Reuse existing runtime | JMD95 Hamiltonian and fixed rotation/kinetic operators |
| #32 | Reuse overlap primitives | Conservative scalar and edge-momentum remapping; original temperature-coordinate facade remains legacy |
| #33 | Reuse transport ledger | Carrier-outflow checks and ocean-owned conservative budgets |
| #34 | Evidence only | Idealized caloric completion is superseded; do not expose it as a competing solver |
| #35 | Adopt fitted surface reference only | Retain GSW calibration/provenance; enthalpy-coordinate dynamics failed an entropy case |
| #36 | Adopt entropy state and work pairing | Checked stages, continuous zero-edge behavior and matched analytic derivatives |
| #37 | Retain qualification evidence | Defines the limits of current wave, balance and transport claims |
| #38 | Adopt selected transport | Local passive bounds and work-limited active targets; document conditional active bounds |
| #39 | Retain native-entropy remap as evidence | Measured thermal-energy loss makes it unsuitable as the selected remapper |
| #40 | Adopt provisional enthalpy remap internally | Removes dominant thermal defect, but alone does not close full Hamiltonian |
| #41 | Adopt column-energy ownership only | Uniform correction rejected weak-stable cases; keep that solver research-only |
| #42 | Expose adaptive remap explicitly | Resolves demonstrated failures; retains mandatory feasibility checks and known limits |

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

## Implementation map and provenance

`thermodynamics.py` contains the bounded-domain surface Gibbs fit; `faces.py`
contains ocean Gibbs means; `equations.py` owns the entropy state/Hamiltonian and
parent work pairing; `transport.py` applies the selected front correction.
`enthalpy_remap.py`, `energy_columns.py` and `remap.py` implement the selected ALE
composition. `model.py` supplies the small public facade. Research scripts remain
frozen as independent migration references. Their duplication is intentional
historical evidence; package runtime never imports tools, drivers or atmosphere.
The coefficient JSON is packaged in the wheel, unchanged from its calibration
artifact. No GSW runtime dependency is introduced. This hybrid reference is not
full TEOS-10: JMD95 supplies density, with a fitted surface caloric reference and
reference-composition salinity mapping. See SEAWATER_CALORICS.md in verification.

## Supported scope and unresolved limitations

“Supported” here means an installed API with regression coverage, not a globally
qualified climate model. Evidence is CPU float64 and unbatched. GPU, compile,
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
- Latest adaptive wave evidence is only .05 period. Earlier longer wave tests do
  not qualify the newer remapper. Repeated near-neutral resets/front cycles remain
  needed. Perturbation-energy diagnostics must avoid subtracting huge backgrounds.
- Rotation/kinetic reconstruction limitations and fine-mesh balance measurements
  remain unchanged. Better packaging is not a new balance or accuracy result.
- Derivatives are piecewise through active-set/iteration changes; existing smooth
  directional checks do not establish gradients across all switching surfaces.

## Verification and next campaign

Migration checks compare native states, all three stage ledgers, Hamiltonian terms
and adaptive remap reports exactly against frozen research execution across four
profiles and wet/coastal configurations. Run the full portable suite and legacy
164-tensor ocean comparison outside the checkout. Historical exact atmosphere
fixture disagreements remain informational and separate from portable success.
All research evidence and source hashes remain untouched.

Next: repeated-remap and multi-period wave qualification, then matched stress/work
integration and Ekman/pumping/basin tests. Atmospheric validation remains separate;
no new turbulence closure is bundled into this integration.
