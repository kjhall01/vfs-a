# Temperature-native complete research ocean

`TemperatureSystem` composes checked temperature-content SSPRK3, prescribed
water/heat exchange, wind and bottom stress, penetrating radiation/surface heat,
vertical TKE mixing, and explicit post-physics adaptive thermal ALE. It uses the
same JMD95/quadrature Hamiltonian and surface caloric fit. Defaults in the
supported entropy model are unchanged; no runtime GSW dependency is added.

## Three optimization passes

1. `FastSurfaceHeating` and `FastSurfaceExchange` replace fixed six-iteration
   full-parcel heating with the residual-controlled, T-derivative-only increment
   solver. Both entropy and temperature storage are supported. The original
   forcing laws, external work ledgers and final energy tolerances remain in the
   shared parent operators. Thin protected hooks leave default behavior intact.
2. `TemperaturePathMixing` integrates derivatives along a straight endpoint T/S
   path, eliminating intermediate enthalpy-to-temperature inversions. Define
   PT=integral Psi_T, PS=integral Psi_S, HT=integral h0_T, HS=integral h0_S along
   that path. Then a=PT/HT and b=PS-a*HS give dPsi=a*dh0+b*dS. These coefficients
   are used with the actual conservative enthalpy/salt fluxes and the previous
   midpoint energy-flux allocation. Finite quadrature is checked by the original
   full-H, local-TKE and entropy acceptance tests. Three-point evaluation is the
   reference research configuration. One-point evaluation is an experiment.
3. `TemperatureSystem` carries h*T throughout the composed step and remap. TKE is
   packed with passive contents for horizontal transport/remap, then evolved by
   the vertical closure. The native ALE is an explicit research translation of
   the existing enthalpy remap plus mass-weighted adaptive thermal projection;
   it uses the same projection solver, energy/entropy constraints and tolerances.
   Salt/passives and column-integrated momentum are conservatively remapped;
   remap thermal correction can change potential enthalpy and temperature content.

The integral identity is exact for exact integration; numerical quadrature and
finite inversions are not exact identities. No claim of exact finite-timestep
energy conservation or Poisson/Jacobi structure is added. Nonlinear coordinate
changes do not commute with finite RK updates. Temperature is not transported as
an independently conserved passive tracer.

`GuardedTemperaturePathMixing` is an optional midpoint proposal with an order-3
retry on specific energy/entropy-budget failures. It retains the original input
for retry, and records refinements. It does NOT compare two quadrature orders on
accepted proposals, so it cannot certify local work-allocation accuracy. It is
not enabled implicitly. Failed domain checks are not masked or clipped.

## Example

Run from the repo with VFS installed, explicitly exposing the research modules:

```python
import sys
sys.path[:0] = ['validation/temperature_coordinate', 'validation/ocean_column']
from system import TemperatureSystem
from surface import FastSurfaceHeating, FastSurfaceExchange, TemperatureBoundary
from path import TemperaturePathMixing
from vfs.ocean.stratified import TKEState

# ocean is the existing StratifiedOcean, optionally compile_thermodynamics().
physical = ocean.check_state(entropy_state)  # initial conversion only
state = TKEState(physical, tke_h)
model = TemperatureSystem(
    ocean,
    closure=TemperaturePathMixing(2., substeps=4),
    boundary=TemperatureBoundary(.0025),
    heating=FastSurfaceHeating(.6, (.5, 20.)),
    exchange=FastSurfaceExchange(),
)
state, report = model.step_with_report(state, 60., **prescribed_fluxes)
target = state.ocean.h.sum(-1, keepdim=True).expand_as(state.ocean.h) / state.ocean.h.shape[-1]
state, remap = model.remap_with_report(state, target)
```

Complete-step examples and timing: `complete_campaign.py --compile --ale`.
`paired_timing.py --output paired.json` alternates the compiled entropy baseline
and three-point native composition on 1024×64, including ALE and all reports.
Qualification examples (installed VFS, from repository root):

```sh
python validation/temperature_coordinate/qualification.py wave --cells 32 --output wave.json
python validation/temperature_coordinate/qualification.py rest --cells 32 --output rest.json
python validation/temperature_coordinate/qualification.py jet --cells 128 --output jet.json
python validation/temperature_coordinate/complete_campaign.py --cells 32 --layers 8 --compile --ale --steps 60 --variants entropy temperature_ts3 --output hour.json
```

The wave/steady-flow script advances unforced dynamics. The complete campaign
includes forcing, mixing and ALE. Results include the reference and scope.
All published comparisons keep order-4 Hamiltonian quadrature. Heating/path
quadrature is a separate integration, not that EOS quadrature setting.

## Scope

This is research composition, not promotion to the supported package facade.
The ALE translation intentionally stays alongside the original for comparison;
consolidating the implementations requires further review. Fully wet mixing is
still required by the lagged closure; coast tests cover dynamics/remap/forcing
without implying coastal mixing support. Forcing is prescribed, not calculated
from an atmosphere. Optical parameters, drag and mixing length are explicit.
First-order splitting is retained. No GPU/Mac or long climate-run qualification.
No thermal closure, limiter, energy guard or diagnostic report is silently removed.
