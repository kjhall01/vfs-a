# Radiation-layer heating transfer — 17 November 2026

The user supplied six-hour diagnostics from their 16,384-cell real-data campaign:
combined energy minus TOA -43948.38670008676 J/m², upstream residual
-43947.26916393397 J/m², remaining numerical residual -1.1175361527930363 J/m²,
and peak absolute column radiation residual 17.362086380084815 W/m².
These are user-reported values, not a reproduced run in this runtime.

## Diagnosis and correction

In pinned PyAM21 `e1bdfcc3b23c05adea80b35af880702bb85718eb`, `atmos_input`
constructs radiation `pflux` at averages of adjacent full pressures, with zero
top pressure. SW/LW K/s heating is divided by the mass of those radiation
layers. The previous host passed these tendencies directly to AM2 physics
using hybrid `phalf` mass weights. These grids differ even on a fresh call;
holding the native K/s cache while host mass changes adds a second discrepancy.

The workflow's `HostRadiationDriver` now conserves layer-integrated native
SW/LW heating by fractional-pressure overlap before dividing by the current
host mass. It maps the actual package heating rather than replacing it with
flux divergence, so genuine native heating/flux errors remain visible. Native
interface, TOA and surface fluxes are unchanged. LW used by boundary-layer
turbulence is mapped consistently with total heating.

The native pressure grid is saved alongside native heating in the existing
radiation cache. After solar renormalization, each physics call maps once to
the current host grid. Cached heating is never replaced by already mapped
heating. The adapter's atomic cache handling and bitwise restart retain this
additional tensor.

The radiation column reaches p=0 whereas the VFS lid is 1000 Pa. The explicit
closure lumps above-lid absorption into the top host layer without changing
that layer's mass; it does not resolve an additional stratospheric layer.
Fractional-pressure coordinates keep the full cached column when surface
pressure changes. This convention is recorded in radiation manifest schema 2.

## Evidence

[comparison.json](comparison.json) uses analytic interface fluxes on the real
upstream radiation geometry and scheduler, not fabricated optical tables.
Across 12, 24 and 40 layers and fresh/cached calls with changing pressure:

| Quantity | Maximum/range |
| --- | --- |
| Previous host-mass heating residual | 3.140–5.922 W/m² |
| Corrected host-mass heating residual | at most 1.706e-13 W/m² |
| Change in upstream SW/LW interface fluxes | exactly zero |

The focused installed-wheel suite completed **105 passed, 1 skipped** in 17.77 s:

```bash
python -I -m pytest tests/test_radiation_mapping.py tests/test_shared_exchange.py \
  tests/test_coupled_campaign.py tests/test_pyam21_workflow.py \
  --fixture-mode=portable -q --tb=short
```

Run from an external copy of the workflows/tests/validation tree with the VFS
wheel installed. Tests cover independent layer-intersection integrals, fresh
and cached radiation, 1200/3600 s cadence, solar renormalization on/off,
day/night, changing carrier masses, a retained deliberate +3 W/m² native
heating defect, atmosphere and coupled bitwise restart, and failed-step rollback.
The skipped test needs actual GFDL tables; it now checks the radiative energy
budget after fresh and cached calls, not merely finiteness.

The [corrected-Gibbs comparison](baseline.json) passes 142 tensors and the
[dry-helper comparison](dry.json) passes 93 tensors. Python 3.12, PyTorch
2.15.0+cpu, CPU float64, installed VFS outside the checkout. No packaged kernels
or upstream PyAM21 sources changed.

Reproduce the analytic numbers with [compare.py](compare.py):

```bash
python -I verification/atmosphere/pyam21/campaign/radiation-grid/compare.py \
  --output comparison.json
```

## Next real-data check

Start a fresh campaign segment after pulling the correction. Old checkpoints
reject the changed source/manifest identity. Keep the same timesteps and mesh.
The CSV now records signed cumulative `radiation_energy_residual_J_m2` and
`upstream_minus_radiation_residual_J_m2`, alongside the existing peak local
`radiation_residual` in W/m². These decompose the existing upstream residual;
they are not additional independent errors to add together.

The runtime still lacks the user's external tables and mesh. This demonstrates
a definite transfer defect and its correction, not that all of the reported
-2.035 W/m² mean loss is explained or that real-radiation/year-long stability
is established. A remaining upstream residual must be investigated separately.
