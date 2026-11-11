# Ocean verification

The active stratified model is `vfs.ocean.stratified.StratifiedOcean`. It uses
JMD95 density, entropy/salt transport, checked SSPRK3 and an explicitly selected
adaptive thermal ALE remapper. See [the API](../../docs/OCEAN_INTEGRATION.md) and
[example](../../examples/ocean_stratified.py).

## Evidence by property

| Property | Measurements and limitations |
| --- | --- |
| Hamiltonian pressure and content work | [JMD95 formulation](JMD95_HAMILTONIAN.md), [entropy pairing](ENTROPY_TRANSPORT.md) |
| Thermodynamic reference | [Surface caloric fit and comparison](SEAWATER_CALORICS.md); hybrid JMD95/reference composition, not full TEOS-10 |
| Carrier and tracer admissibility | [Outflow restriction](TRANSPORT_ADMISSIBILITY.md), [bounded transport](BOUNDED_TRANSPORT.md) |
| Sharp fronts | [Front measurements](SHARP_FRONT.md); active local bounds remain conditional on work budget |
| Waves and balance | [Qualification campaign](QUALIFICATION_CAMPAIGN.md), [rotation diagnosis](ROTATING_BALANCE.md) |
| Adaptive vertical remap | [Energy/entropy constraints and short waves](ADAPTIVE_THERMAL_ALE.md) |
| Remap alternatives and resolution sensitivity | [Native entropy](ENTROPY_ALE.md), [potential enthalpy](ENTHALPY_ALE.md), [uniform thermal correction](HAMILTONIAN_ALE.md) |
| Installed API regression | [Source provenance](integration-provenance.json), [migration tests](../../tests/ocean/test_stratified_integration.py) |

Each report identifies its configuration. The three-period wave campaign and
fine-mesh initial-balance results do not qualify every later transport/remap
combination. Adaptive-remap wave measurements cover only .05 period. Energy
closure alone does not establish accurate wave phase, low mixing or balanced jets.

## Configurations

`Ocean` with default layered dynamics retains a reference mechanical-energy
diagnostic that omits stratification; it is not the active model's Hamiltonian.
[Constant-density dynamics](CONSTANT_DENSITY.md) and centered JMD95 dynamics are
explicit equation choices. Source/mixing objects in `vfs.ocean.physics` belong
to the configurable `Ocean` path; they are not integrated into `StratifiedOcean`.

The stratified API is unforced, unbatched and verified on CPU float64. It rejects
invalid stages and infeasible remaps without clipping or hidden timestep changes.
Rotation/kinetic consistency, repeated-remap behavior and sustained front/wave
accuracy remain limitations. There is no TKE/EKE closure or coupled driver.

## Running checks

After installing a wheel, copy `tests`, `tools` and `examples` outside the checkout:

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I examples/ocean_stratified.py
```

The tests compare the package with frozen research calculations, including coast
masks, stage ledgers and remap reports. Legacy reference comparisons are separate
from active-model qualification. See [test setup](../../CONTRIBUTING.md) and
[verification conventions](../README.md). Numerical artifacts and dated reports
retain their original source hashes and measurements.
