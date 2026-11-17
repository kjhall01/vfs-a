# Seasonal-radiation setup attempt — 17 November 2026

PR #89 was merged after its final installed-wheel CI succeeded. Base commit:
`116eed948874c4ffc7e2464618cc80dc3eec0d1b`. The requested seasonal-radiation
debugging run remains **blocked before initialization**, not a completed run.
The external GFDL `input_data/sea_esf` tables are absent from this environment.
Evan's pinned PyAM21 checkout excludes `reference/input_data/` from Git and
documents it as a separately acquired SCM data bundle.

The official quickstart now links
`ftp://ftp.gfdl.noaa.gov/perm/GFDL_pubrelease/SCM/input_data.tar.gz`.
Both that and the older `/perm/pjp/SCM/` endpoint failed via FTP from this
runtime. HTTPS on ftp.gfdl.noaa.gov returned 503; corresponding paths on
data1.gfdl.noaa.gov and nomads.gfdl.noaa.gov returned 404. The MOM6 examples
contain links to external radiation inputs, not a replacement copy of these
files in Git. No tables, climatologies or heating rates were fabricated.

## A separate setup defect corrected

The campaign inherits the BOMEX namelist. It disables both shortwave and
longwave aerosol optical effects. Running the actual pinned SeaEsfRad
configuration validator against that namelist rejects with
`aerosols must be active in both the SW and the LW`. ASTEX and MPACE enable
both and pass the same validator. This is independent of missing data.

When a radiation-data directory is supplied, setup now explicitly enables both
flags before applying caller overrides. An explicit unsupported override still
rejects. Radiation-off configuration is unchanged. The table download URL in
the missing-data error now follows the current official quickstart.

The two regressions execute the upstream configuration validator unmocked and
intercept setup before loading external tables. They establish configuration
compatibility and override behavior, **not radiative flux correctness**.
External installed-wheel shared-exchange/campaign checks: **47 passed, 1 skipped**;
the skip is the real-data radiation test. Package kernels and upstream sources
are unchanged. The frozen 142/93 portable comparisons are recorded here too.

## Requested configuration

The supplied one-year YAML retains seasonal/diurnal geometry, 1361 W/m2 solar
constant, 23.44-degree obliquity, eccentricity 0.01671 and the requested
1200/240/1200/3600-second ocean/atmosphere/physics/radiation intervals. It enables
the implemented aquaplanet atmosphere, turbulent/convective/cloud physics,
Roquet ocean, prognostic TKE, GM–Redi and mesoscale energy, funded backscatter,
biharmonic dissipation, ALE, moving ice/snow and prognostic ice temperature.
Radiation-on setup now also selects both supported aerosol optical effects.
The active configuration and attempted preflight status are archived alongside
this file. Zero timesteps were advanced by the radiation-on attempt.

Once the actual directory is available, use the normal campaign launcher:

```bash
python -I workflows/coupled/campaign.py --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output seasonal-preflight --preflight --verbose
python -I workflows/coupled/campaign.py --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output seasonal-debug-2d --stop-after-days 2 --verbose
```

Add `--mesh /path/to/mesh.npy` for saved generators. Start fresh after source
changes. The preflight must execute successfully with real data before a
radiation-on debugging result can be reported. See the
[data inventory and radiation contract](../../../../../workflows/pyam21/RADIATION.md).
