# Actual SeaEsfRad setup

`radiation.build_radiation` constructs **PyAM21's real `SeaEsfRad` and
`RadiationDriver`**, using the supplied GFDL radiation tables and climatologies.
It computes the two reference pressure columns from the VFS hybrid grid using
the same Simmons–Burridge full-pressure readout as the adapter. Neither upstream
code nor package defaults are changed. Missing inputs fail with an actionable
path; there is no synthetic fallback or network access during stepping.

**Status: wiring and synthetic orchestration tests pass; full SeaEsfRad execution
has not been verified.** On 2026-11-15 the advertised GFDL FTP connection failed
from this environment; HTTP and HTTPS on that host returned 503. Candidate
nomads/data1 HTTPS paths returned 404. No reference data were fabricated or
substituted, and none are committed to this repository.

Obtain the SCM input bundle linked from the
[GFDL quickstart](https://www.gfdl.noaa.gov/scm-quickstart/):

```bash
curl -O ftp://ftp.gfdl.noaa.gov/perm/pjp/SCM/input_data.tar.gz
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
climatology settings otherwise come from the pinned BOMEX namelist. Enabling
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
