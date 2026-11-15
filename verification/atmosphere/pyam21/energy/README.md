# Conservative AM2–VFS energy transfer

VFS base `74f23640042b5a9f69e4aaf5b96fdf214d5795e8`, PyAM21
`e1bdfcc3b23c05adea80b35af880702bb85718eb`; separately installed packages,
CPU float64, one thread. Tests and drivers ran outside the checkout against the
installed VFS wheel. Package source/defaults and upstream physics are unchanged.
See [environment](environment.json), [the energy derivation](../../../../workflows/pyam21/ENERGY.md)
and [usage](../../../../workflows/pyam21/README.md).

## Regression and independent checks

- **1,366 passed, 2 skipped** in the installed-wheel portable suite,
  including all 36 PyAM21 workflow checks. [Log](installed-wheel-pytest.log).
- Corrected Gibbs portable comparison: **142 tensor comparisons passed**.
  [Report](baseline-portable.json).
- Dry-helper portable comparison: **93 tensor comparisons passed**.
  [Report](dry-portable.json).

Tests independently recompute native layer and global full-H changes for sensible
heating, evaporation, dew, rain, snow, phase changes, energy-returning drag and
an intentionally undeclared heat source. They check explicit entering-water
momentum/kinetic energy and nonzero surface geopotential, actual radiation-driver
flux/cache behavior with a synthetic package, restart separation between modes,
and smooth derivatives including a zero-valued energy source. Disabling actual
AM2 dissipative heating leaves its energy deficit visible after transfer.

Two prototype rejection logs are retained: [subtractive condensate exhaustion](initial-column-rejection.log)
and [tendency endpoint cancellation](tendency-endpoint-rejection.log). The first
left a negative CP remnant after an almost exhausted store; the second reconstructed
a negative fraction from tendencies when the actual upstream endpoint was zero.
Final code uses remaining-amount donor mapping and the checked actual upstream
endpoint. Targeted tests reproduce both cases without moisture floors or state
clipping; a truly negative supplied endpoint is retained and substantive
inadmissibility still rejects. These logs are development failures, not failures
of the final cases below.

## Final conservative integrations

All cases use 24 levels, a 1000 Pa lid, an analytic 70%-RH profile and prescribed
SST. Radiation is off. The BOMEX namelist selects the physics; these are not
reproductions of the BOMEX SCM forcing/reference. Columns hold dynamics off;
aquaplanets include native bounded SSPRK3 dynamics and cloud transport.

| Case | Grid; duration | Physics / dynamics dt (s) | Final full-H minus complete boundary energy (J/m²) | Final accumulated AM2 residual (J/m²) | Final accumulated transfer residual (J/m²) |
| --- | --- | --- | --- | --- | --- |
| [Columns](column6h/summary.json) | 32×24; 6 h | 60 / off | +1.030e-6 | +4.461e-7 | +4.903e-7 |
| [Aquaplanet](aqua6h/summary.json) | 32×24; 6 h | 60 / 15 | +1.218e-3 | +1.946e-7 | +1.159e-7 |
| [Half timestep](aqua6h-dt30/summary.json) | 32×24; 6 h | 30 / 7.5 | +6.312e-4 | +5.618e-7 | +3.946e-8 |
| [Larger mesh smoke](aqua128/summary.json) | 128×24; 1 h | 60 / 15 | −3.589e-5 | +3.566e-8 | +2.930e-8 |

The four cases complete 1,500 physics intervals over 19 aggregate model hours.
The first numeric column includes native dynamics drift. The two six-hour
aquaplanet values correspond to +5.64e-8 and +2.92e-8 W/m² averaged over the run;
nearly all of that residual comes from dynamics timestepping. The isolated column
value is near the floating-point resolution of subtracting full background energies.
The per-layer enthalpy-constraint residual peaks at 8.67e-8 J/m² across the cases;
all solves take at most two Newton steps. No upstream energy residual is removed.

Peak water-flux ledger residual is 4.81e-12 kg/m², relative dry-air change is
2.13e-13, and per-step species mapping residual is below 6.66e-15 kg/m². Every
summary records all workflow source hashes; its adjacent `series.csv` retains
every physics interval. Console logs are alongside the case directories.

Halving the timestep reduces the observed total energy residual, but does not
establish an asymptotic order or converged cloud physics. Final mean precipitation
is 5.78e-5 versus 6.99e-5 kg/m²/s in the six-hour aquaplanets, and peak CP temperature
at the final time is 300.80 versus 301.15 K. These remain timestep-sensitive
trajectories. The 128-cell case has a shorter duration and is not a spatial
convergence pair. Wall timings overlap with the test suite and are not performance
measurements.

## Legacy reproduction

The original mapping is retained behind `--energy-mode legacy`. Its six-hour
comparison and exact scalar-record report are saved as
[legacy6h/summary.json](legacy6h/summary.json) and
[legacy-comparison.json](legacy-comparison.json).
All **7,920 comparisons** (22 common scalar fields across 360 records) match
the archived PR86 aquaplanet records exactly. This is a scalar-history comparison,
not a claim that archived full-state snapshots were available. Measured against
the complete surface ledger, that unchanged legacy trajectory has a +4.595 W/m²
mean residual; the conservative trajectory's corresponding residual is +5.64e-8
W/m². The modes intentionally have different temperature/cloud trajectories.
The original 0.424 W/m² comparison used the incomplete AM2 heat ledger. The complete
surface ledger here also includes transported-water enthalpy and mechanical energy,
so those two residual definitions must not be presented as the same quantity.

## Reproduce

Install the VFS wheel and pinned optional PyAM21 dependency, then copy `tests`,
`tools`, `examples`, `validation` and `workflows` to an external harness. From there:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable --report baseline-portable.json
python -I tools/check_dry_helpers.py --mode portable --report dry-portable.json
python -I workflows/pyam21/run_column.py --source /path/to/PyAM21 --hours 6 --dt 60 --output column6h --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 6 --dt 60 --dynamics-dt 15 --output aqua6h --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 6 --dt 30 --dynamics-dt 7.5 --output aqua6h-dt30 --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --cells 128 --hours 1 --dt 60 --dynamics-dt 15 --output aqua128 --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 6 --dt 60 --dynamics-dt 15 --energy-mode legacy --output legacy6h --verbose
```

The boundary is a stationary prescribed ocean, with explicit assumptions for
water temperatures and velocities. Interactive ocean/ice needs matching enthalpy
datums, exchanged-water states and reciprocal stress/work on both sides. Real
SeaEsfRad and its GFDL input bundle, large production meshes, long climate runs,
entropy stability and long-horizon differentiation remain unqualified.
