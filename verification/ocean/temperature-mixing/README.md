# Temperature-native heating and mixing

Research extension of the temperature-coordinate dynamics candidate. The
production package/defaults are unchanged. `physics.py` in the temperature
campaign contains `heat` and `TemperatureMixing`; the latter reuses the existing
lagged closure's momentum/scalar solves, funding retries, substeps, local TKE
budget, column full-H acceptance, entropy nondecrease and admissibility checks.
Small storage hooks were extracted in the research lagged closure rather than
duplicating the algorithm. Existing entropy behavior is regression checked.

Temperature mixing transports reference enthalpy and salinity, solves for the
new temperature, and stores h*T directly. It eliminates the entropy roundtrip
baseline and candidate entropy inversions. The finite thermal heating solve is
shared with the existing fast path, but its temperature is returned directly.
No nonlinear energy solve is replaced with constant heat capacity. The same
JMD95 quadrature and caloric reference define H. Energy checks and tolerances
are not relaxed. Zero forcing returns the input state; finite nonzero heating
and cooling are checked against the full layer energy change.

The model still needs enthalpy-to-T solves: scalar mixing proposals and the
quadrature states used to allocate buoyancy work. This change removes entropy
storage overhead, not all nonlinear thermodynamic work. Baseline subtraction
in the old representation preserved stored entropy against roundtrip noise.
Direct temperature storage instead accepts the enthalpy solve result, with the
unchanged full-H check bounding numerical conversion error. Zero mixing is not
promised bitwise identical in temperature after enthalpy reconstruction.

## Use

From a checkout with VFS installed (research directories explicitly on path):

```python
import sys
sys.path[:0] = ['validation/temperature_coordinate', 'validation/ocean_column']
from model import TemperatureOcean
from physics import heat, TemperatureMixing

candidate = TemperatureOcean(ocean)  # existing StratifiedOcean
physical = ocean.check_state(entropy_state)  # one initial conversion
physical, heat_residual = heat(ocean, physical, layer_heat_J_kg)
closure = TemperatureMixing(2., substeps=4)
physical, tke_h, report = closure.apply(ocean, physical, tke_h, dt=60.)
```

`heat` takes prescribed energy per layer, not W/m². It is not yet a radiative
penetration, surface-water exchange or wind/bottom-stress adapter. Those operators
and ALE still need temperature-native composition before this replaces a full
supported ocean timestep. No new supported public API is exported.

## Reproduce

```sh
python validation/temperature_coordinate/physics_campaign.py --compile --output small.json
python validation/temperature_coordinate/physics_campaign.py --compile \
  --cells 1024 --layers 64 --repeats 1 --output large.json
python -m pytest tests/ocean/test_temperature_physics.py -q
```

The benchmark is layer heating (+0.1 J/kg) followed by 60 s mixing in four
substeps, not a full ocean step. CPU float64, one thread, q4; both variants use
the same optional compiled kernels. Warmup is excluded; call counts are collected
in separate untimed executions. No test jobs run alongside final timing jobs.
Large samples are exploratory, not confidence intervals or Mac predictions.

## Results

At 1024×64 the single post-warmup sample was 10.0738 s (entropy) versus 10.0343 s
(temperature). This is effectively no demonstrated speedup; do not multiply it
into the earlier dynamics gain. Both use the same kernel compilation. The
remaining 32 enthalpy inversions per update include scalar proposals and
energy-path quadrature evaluations; this campaign counts calls but does not
attribute wall time to individual kernels.

| Inversions per heating + mixing update | Entropy storage | Temperature storage |
|---|---:|---:|
| State entropy inversions | 2 | 0 |
| Caloric entropy inversions | 13 | 0 |
| Caloric enthalpy inversions | 36 | 32 |

Both large runs had energy-minus-prescribed-heat residual -5.93e-12 J/kg,
maximum local TKE budget ratio 3.47e-7 (acceptance threshold 1), eight trials,
no reduced/deferred interfaces, and positive final TKE/passives. Endpoint
between-coordinate temperature RMS difference was 4.59e-15°C; maximum h*k
difference 2.36e-18, edge velocity 8.67e-19 m/s, salt content 7.11e-15.
These are shallow 100 m short cases, not deep-ocean, wave, long-run or full-ocean
qualification. Tolerances and fixtures are unchanged. GPU/Mac performance is
not established. Raw small/large samples are retained alongside this document.

At 128×16 the final three-sample medians were 0.4334 s (entropy) and 0.3686 s
(temperature), about 1.18×. This modest small-case improvement did not persist in
the larger sample. The candidate's standalone layer-heat helper also performs an
explicit full-H check; timings include that check. No safety check was omitted
to obtain a speedup.

Fourteen new physics checks cover heating/cooling/zero heat, independent energy
change, salt/passive inventories, stable/unstable columns with zero/low/ordinary
TKE, explicit deferral, entropy-inverse prohibition, reference agreement and a
nonzero-heating backward smoke. Refactoring the old entropy path matched 24
state/report tensors bitwise in both ordinary and fast modes on a 16×4 case.
These comparisons are not blanket bitwise guarantees for all states.

Final broad regression: **1102 passed, one CUDA skip, one existing warning**, in
256.51 s. Tests, tools, verification fixtures and candidate research helpers were
copied outside the checkout; VFS was imported from the unchanged installed wheel.
The run used `--fixture-mode portable`, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1.
No fixture/tolerance changes. See suite.txt and provenance.json.
