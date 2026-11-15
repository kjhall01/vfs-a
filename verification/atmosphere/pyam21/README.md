# External PyAM21 workflow: first integration checks

VFS base `c392bc4302369014ebb4833623f554ca2448680a`, PyAM21
`e1bdfcc3b23c05adea80b35af880702bb85718eb`; CPU float64, one thread,
Python 3.12.14 and PyTorch 2.15.0+cpu. Both packages were installed separately;
the wheel was tested from an external harness. See [environment.json](environment.json)
and the [workflow conventions](../../../workflows/pyam21/README.md).
No VFS package source or default was changed, and the workflow is excluded from
the wheel. Each run's summary records SHA-256 hashes of all workflow Python files.

## Required checks

- Installed-wheel portable suite: **1,351 passed, 2 skipped**, including all 21
  adapter tests. [Full log](installed-wheel-pytest.log).
- Corrected Gibbs v1 frozen portable comparison: **142 tensor comparisons passed**.
  [Report](baseline-portable.json).
- Dry helpers frozen portable comparison: **93 tensor comparisons passed**.
  [Report](dry-portable.json).

These checks cover amount transfer, independent work identities, zero-forcing
identity, staged equivalence to actual PyAM21 `forward`, source signs, invalid
input rejection, restart round trips, cloud transport, bitwise preservation of
the native atmospheric step, and one smooth SST derivative. The radiation test
uses the real upstream driver with a synthetic radiation package and exercises
its scheduling, flux exposure, restart cache and failure rollback. It does not
exercise SeaEsfRad or the GFDL input-data bundle.

## Short runs

All cases have 24 levels, a 1000 Pa lid, prescribed SST, an analytic initial
temperature profile and 70% relative humidity. The PyAM21 BOMEX namelist selects
the physics, but these are not BOMEX reference-case reproductions. Radiation is
off. The aquaplanet adds VFS dynamics and a meridional temperature gradient.

| Case | Cells; duration | Physics / dynamics dt (s) | Peak relative dry-air change | Peak water ledger residual (kg/m²) | Mean VFS minus AM2 heat ledger (W/m²) |
| --- | --- | --- | --- | --- | --- |
| [Columns](column6h/summary.json) | 32; 6 h | 60 / off | 1.20e-15 | 3.38e-14 | +0.594 |
| [Aquaplanet](aqua6h/summary.json) | 32; 6 h | 60 / 15 | 1.06e-13 | 2.37e-12 | +0.424 |
| [Aquaplanet, half timestep](aqua6h-dt30/summary.json) | 32; 6 h | 30 / 7.5 | 2.12e-13 | 4.77e-12 | +0.443 |
| [Larger mesh smoke](aqua128/summary.json) | 128; 1 h | 60 / 15 | 1.83e-14 | 4.04e-13 | +0.907 |

This is 1,500 physics intervals over 19 aggregate model hours. All four cases
completed. Peak per-step local species mapping residual is below 2.45e-15 kg/m².
After subtracting the measured physics energy changes, peak accumulated native
dynamics energy residual is 7.76e-4 J/m² in the 60 s six-hour aquaplanet and
4.28e-4 J/m² at half timestep. The `series.csv` next to each summary retains
every physics interval; the top-level case logs retain console output.

**The physics is not closed against VFS's full Hamiltonian.** The last table
column is total VFS energy change minus the accumulated conventional AM2
thermal/latent forcing ledger, divided by elapsed seconds. That ledger is not
a complete exchanged-mass enthalpy and mechanical-energy boundary budget.
The reports separately retain thermodynamic-convention differences and finite
CP mapping/carrier/geometry differences. No energy repair was applied. These
numbers are not assertions of energy error in the upstream Fortran port.

Halving the timestep leaves the mean heat-ledger difference near 0.4 W/m²;
it does not establish convergence to a common energy convention. Final mean
precipitation also differs between the two timesteps. This is a short integration
and sensitivity check, not temporal convergence or climate qualification. The
128-cell case has a different duration, so it is not a resolution-convergence pair.

## Reproduce

After installing the wheel and pinned optional dependency as described in the
workflow README, copy `tests`, `tools`, `examples`, `validation` and `workflows`
to a directory outside the checkout. Run there with the installed environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable --report baseline-portable.json
python -I tools/check_dry_helpers.py --mode portable --report dry-portable.json
python -I workflows/pyam21/run_column.py --source /path/to/PyAM21 --hours 6 --dt 60 --output column6h --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 6 --dt 60 --dynamics-dt 15 --output aqua6h --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 6 --dt 30 --dynamics-dt 7.5 --output aqua6h-dt30 --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --cells 128 --hours 1 --dt 60 --dynamics-dt 15 --output aqua128 --verbose
```

The first next step is to reconcile the shared thermodynamic and mass-source
energy convention, including exchanged-mass enthalpy and mechanical work.
Then qualify real radiation with its data manifest and extend to interactive
ocean/ice exchange. GPU, long climate integrations, large production meshes,
threshold-crossing derivatives and long-horizon differentiation remain untested.
