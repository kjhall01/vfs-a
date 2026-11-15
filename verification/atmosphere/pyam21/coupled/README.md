# Shared exchange verification — 15 November 2026

Actual pinned PyAM21 `e1bdfcc3b23c05adea80b35af880702bb85718eb`, installed VFS
wheel, Python 3.12.14, PyTorch 2.15.0+cpu, float64, one thread per experiment.
The harness was copied outside the checkout and executed with `python -I`.
Package sources and upstream PyAM21 were unchanged. No GPU, long climate or
coupled backward qualification is claimed.

- Full installed-wheel portable suite: **1,381 passed, 3 skipped**.
  [Log](installed-wheel-pytest.log).
- Final focused suite after the extra shared-record consistency guard and datum
  diagnostic correction: **52 passed, 1 skipped**. [Log](final-focused.log).
  The full count predates that one added rejection test; required PR CI tests
  the final tree. The real-data opt-in regression is explicitly skipped.
- Frozen corrected-Gibbs and dry-helper comparisons pass: **142 and 93 tensors**.
  [Atmospheric report](baseline-portable.json), [dry report](dry-portable.json).
- Tests cover moving open-water/mixed-ice exchange, reciprocal generalized
  impulses, total water, retained salt, post-implicit tile flux summation, shared
  energy, cached LW with changed SST, snowfall/deposition/sublimation, exhausted
  stores, malformed intervals, rollback and deliberate upstream heating defects.
  Radiation builder mocks test orchestration/provenance only.

All three six-hour experiments below have **radiation off**, a 32-cell mesh,
24 atmospheric levels and three ocean layers. Mixed-ice experiments use prescribed
270 K ice skin temperature, evolving SST, actual atmospheric/ocean/ice dynamics,
internal drag, compression, phase change and snow. They do not solve the ice skin
energy balance. The open-column run uses physics/exchange without dynamics or
internal phase integration. No scientifically subjective pass thresholds were
introduced; the numbers are observed diagnostics.

| Run | Physics / atmosphere dynamics step | Peak numerical combined-H residual, J/m² | Peak combined-water residual, kg/m² | Peak relative salt error |
| --- | --- | ---: | ---: | ---: |
| [Open columns](open-columns/summary.json) | 60 s / off | 1.069e-5 | 6.424e-11 | 0 |
| [Moving ocean/ice](ice-dynamics-dt60/summary.json) | 60 s / 15 s | 1.347e-3 | 6.424e-11 | 5.905e-16 |
| [Half timestep](ice-dynamics-dt30/summary.json) | 30 s / 7.5 s | 6.184e-4 | 1.285e-10 | 1.476e-15 |

Numerical residual means combined full-H change minus declared TOA/external input
and the separately retained AM2 residual. The latter ends at approximately
-1.35e-7, 4.49e-7 and 4.96e-7 J/m², respectively. Peak local surface transfer error
is below 5.80e-6 J/m²; peak atmospheric layer transfer error below 8.38e-8 J/m².
The approximately twofold timestep reduction is evidence about this short split
integration, not a convergence theorem or climate accuracy result.

The runs include exact code hashes. They precede the final diagnostic correction
and consistency guard, which leave valid step endpoints unchanged; focused final
regressions cover those changes. Each directory retains its every-step CSV and
configuration. No binary artifacts or large input datasets are committed.

## Radiation status and next qualification

The code constructs real upstream SeaEsfRad from explicit external data with
hashed file contents, reference pressures and climatology footprint metadata.
**No real SeaEsfRad run has completed here.** The advertised GFDL FTP connection
failed; HTTP/HTTPS returned 503; candidate data1/nomads HTTPS mirrors returned 404.
The opt-in real-data test is skipped, rather than silently replaced with synthetic
physics. See [instructions and remaining qualification](../../../../workflows/pyam21/RADIATION.md).

The shared surface currently uses caller-supplied ice skin T, common roughness,
a bulk mosaic, an emission-equivalent radiation skin T and combined net heating.
A coupled radiative/conductive skin solve, precise ocean in-situ temperature,
production climatology mapping and longer radiating tests remain necessary.
See the [full contract](../../../../workflows/coupled/README.md).
