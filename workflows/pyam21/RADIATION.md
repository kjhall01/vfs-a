# Actual SeaEsfRad setup

`radiation.build_radiation` constructs **PyAM21's real `SeaEsfRad` and
`RadiationDriver`**, using the supplied GFDL radiation tables and climatologies.
It computes the two reference pressure columns from the VFS hybrid grid using
the same Simmons–Burridge full-pressure readout as the adapter. Neither upstream
code nor package defaults are changed. Missing inputs fail with an actionable
path; there is no synthetic fallback or network access during stepping.

**Status: full SeaEsfRad execution has not been verified.** The radiation-on
preflight on 2026-11-17 could not initialize because the external data are absent.
Both the historical `/perm/pjp/SCM/` and currently advertised
`/perm/GFDL_pubrelease/SCM/` FTP locations failed from this environment; HTTPS on
the FTP host returned 503 and the candidate nomads/data1 HTTPS locations returned
404. No reference data were fabricated or substituted. See the
[recorded attempt and configuration correction](../../verification/atmosphere/pyam21/campaign/radiation-setup/README.md).

PyAM21 includes the parameterization algorithms, but its `reference/input_data/`
is excluded from Git. Supply the extracted `input_data/sea_esf` directory. These
are physical lookup tables and prescribed composition, not a saved sequence of
radiative heating rates:

| Input family | Role |
| --- | --- |
| `stdlvls`, `cns_*` (or supported `.nc` equivalents) | Reference pressures and gas-transmission lookup tables |
| `h2ocoeff_*`, `h2o12001400_hi00_data`, `h2ockd2.1_data`, `h2ockd2.1_corrdata`, `o39001200_hi00_data`, `radfn_5-2995_100-490k` | Longwave gas/continuum absorption and supporting spectral data |
| `esf_sw_input_data_n38b18` | Solar spectrum and coefficients for the configured 18-band shortwave scheme |
| `aerosol.optical.dat` | Aerosol optical properties |
| `o3.climatology.nc`, `aerosol.climatology.nc` | Prescribed monthly ozone and aerosol composition |
| `co2_gblannualdata`, `ch4_gblannualdata`, `n2o_gblannualdata`, `f11_gblannualdata`, `f12_gblannualdata`, `f113_gblannualdata`, `f22_gblannualdata` | Gas concentration records, sampled at configured fixed entry dates |

Take the complete directory rather than guessing individual table variants.
The model computes seasonal/diurnal solar geometry from date and orbital
settings, then computes radiation using its simulated temperature, humidity and
clouds. The campaign's fixed solar constant does not require a historical solar
irradiance forcing series.

Obtain the SCM input bundle linked from the
[GFDL quickstart](https://www.gfdl.noaa.gov/scm-quickstart/):

```bash
curl -O ftp://ftp.gfdl.noaa.gov/perm/GFDL_pubrelease/SCM/input_data.tar.gz
```

Inspect/extract the archive locally and supply the directory containing `stdlvls`,
`cns_*`, the continuum/solar tables, gas time series, ozone and aerosol files
(normally `input_data/sea_esf`). The full bundle is several GB. No downloader is
hidden inside model initialization.

```bash
python -I workflows/pyam21/run_column.py --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf --radiation-step 3600 \
  --climatology-width-degrees 1 --hours 6 --dt 60 --output real-radiation
```

The same flags are available for `run_aquaplanet.py`; the coupled driver also
accepts `--radiation-data` and `--radiation-step`. Cadences are positive whole
seconds, at least one physics interval and multiples of the example physics step.
The examples enable SW renormalization at every physics step and share LW/SW
cadence. Upstream gas configuration, aerosol/cloud optics and fixed-year
climatology settings otherwise come from the pinned BOMEX namelist. When data
are supplied, setup explicitly enables shortwave and longwave aerosol optical
effects, matching the supported ASTEX/MPACE radiation configuration. BOMEX has
those flags off; leaving them off caused the actual SeaEsfRad constructor to
reject even before reading tables. Caller overrides are still honored and an
unsupported override still rejects. The radiation-off setup is unchanged. Enabling
radiation does not turn this analytic experiment into a BOMEX reproduction.

The default climatology footprint is an explicit 1° rectangle around each cell
center, truncated at the poles with periodic longitude. **These are local
climatology samples, not conservative Voronoi cell averages.** The width is
configurable and included in the manifest. Radiation still uses each actual cell's
latitude/longitude for solar geometry. A production climate configuration should
replace this sampling convention with a validated mesh mapping.

The manifest hashes every supplied data file (streamed SHA256), records byte
sizes, sampling boxes, reference pressures and namelist, and contributes its
identity to restart validation. It is written into the run summary. Initialization
therefore reads all supplied bytes once; point at the radiation directory rather
than the entire SCM bundle.

The opt-in regression runs the actual package, a cached step, a fresh radiation
step and a bitwise atmospheric restart replay when both variables are provided:

```bash
PYAM21_SOURCE=/path/to/PyAM21 PYAM21_RADIATION_DATA=/path/to/sea_esf \
  python -I -m pytest tests/test_shared_exchange.py -k real_seaesf_opt_in -q
```

Until that test and day/night, cloud/clear-sky, cadence, flux-divergence and longer
coupled runs execute against the real tables, radiation remains **unqualified**.
The regular suite explicitly labels its synthetic package and mocks only builder
orchestration; those tests are not evidence of real radiative physics.
