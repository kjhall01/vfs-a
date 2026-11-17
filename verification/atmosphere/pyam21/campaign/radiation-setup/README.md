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
intercept setup before loading external tables. They establish compatibility
with that validator and override behavior, **not full constructor compatibility
or radiative flux correctness**.
External installed-wheel shared-exchange/campaign checks: **47 passed, 1 skipped**;
the skip is the real-data radiation test. Package kernels and upstream sources
are unchanged. The frozen 142/93 portable comparisons are recorded here too.

## Follow-up: complete aerosol optical settings

The user's real-data initialization on 16,384 saved generators passed the
package-level validator but then rejected in `AerosolOptics` with
`NotImplementedError: aerosol_data_set other than shettle_fenn`. No timestep had
committed. The BOMEX namelist has a blank dataset and no optical-model list;
enabling the SW/LW flags alone was insufficient.

Setup now reads `do_lwaerosol`, `do_swaerosol`, `aerosol_data_set`,
`optical_filename` and `aerosol_optical_names` from the pinned PyAM21 checkout's
`reference/input/gcss_astex/input.nml`. BOMEX aerosol composition, fixed-year
climatology and other physics choices remain unchanged. Caller overrides apply
last, and radiation-off initialization does not read the ASTEX namelist.

Tests use unmodified namelist sections from PyAM21 commit
`e1bdfcc3b23c05adea80b35af880702bb85718eb`, parse them with the actual reader and
execute both `_check_config` and the actual `AerosolOptics` constructor. Before
the fix, two cases reproduced the user's exact exception. After the fix, the
constructor reaches its expected optical-file open; the selected model names
cover every configured species and required sulfate humidity bin. Invalid
explicit band/dataset overrides still reject, and an explicit optical filename
is honored. No optical tables or radiative fluxes are fabricated by these tests.

External installed-wheel shared/campaign checks: **50 passed, 1 skipped**;
see [test output](aerosol-followup.txt). The skip remains the real-data test.
The corrected-Gibbs **142** and dry-helper **93** portable comparisons were
rerun and pass. The runtime still lacks the external tables, so this correction
does not establish a completed radiation-on integration or 16,384-cell stability.

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
