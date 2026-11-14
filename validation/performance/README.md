# Model throughput on a local CPU

Short forward-only CPU/float64 timings for an installed VFS wheel. Each case runs
in a fresh process; cases run sequentially. No atmospheric/ocean kernels change.
The default `fast` mixing closure is an **opt-in research solver**, not the
installed default. All cases use compatible rotation on a reproducible Fibonacci
Voronoi sphere with zero Lloyd iterations. These initial states exercise operators;
they are not balanced jets or calibrated physical benchmarks.

## Install and run

From the repository root, in your normal VFS environment:

```bash
python -m pip install build
python -m build --wheel --outdir /tmp/vfs-performance-wheel
python -m pip install --force-reinstall --no-deps /tmp/vfs-performance-wheel/*.whl
python -I validation/performance/run.py --preset smoke --steps 1 --repeats 1 --output perf-smoke
python -I validation/performance/run.py --preset quick --output perf-quick
```

Install the package dependencies first if this is a new environment. Python 3.12
and PyTorch 2.15.0 were used for the earlier CPU measurements; the campaign records
actual versions and does not replace your PyTorch installation. `-I` prevents an
accidental source-tree VFS import; only explicit validation helpers are loaded
from the checkout. Keep the checkout and installed wheel at the same revision.

Start with `quick` on the Mac: 128 cells × 16 layers, 1/4/8 threads, all six
workloads. Default: one warm-up step, three timed evolving steps per repeat,
three repeats, 60 s actual timestep. A case has a 30-minute timeout including
setup. This is a short campaign, not an estimate of its own completion time.

Then select the best measured thread count and scale up:

```bash
python -I validation/performance/run.py --preset scaling --threads 4 --output perf-scale
python -I validation/performance/run.py --sizes 1024x64 --threads 1 4 8 \
  --cases mixing --closures supported semi fast --dt 300 --output perf-mixing
python -I validation/performance/run.py --sizes 1024x64 --threads 4 \
  --cases ocean full full_ale --dt 600 --mixing-substeps 8 --output perf-ocean600
python -I validation/performance/run.py --sizes 1024x24 4096x24 16384x24 \
  --cases dry moist --threads 4 --dt 60 --output perf-atmosphere
```

`scaling` uses 128×32, 1024×64, 4096×64. Large cases can be slow: select cases,
threads and sizes rather than launching a huge Cartesian product. Choose a new
output directory for every run. A failed case is retained and the campaign
continues; the command exits nonzero if any case fails or times out. No automatic
timestep reductions or altered tolerances rescue a failure.

## What each row measures

| Case | Timed operations |
|---|---|
| dry | Dry atmosphere SSPRK3, constant potential temperature, smooth zonal wind |
| moist | Bounded moist atmosphere SSPRK3, smooth positive vapor/liquid/ice |
| ocean | Active JMD95 ocean SSPRK3 plus horizontal TKE transport; no vertical physics |
| mixing | Vertical momentum/scalar mixing, TKE transport, dissipation and energy accounting |
| full | Ocean dynamics + TKE transport + prescribed water/heat exchange + stress/bottom drag + radiation + vertical mixing |
| full_ale | `full` followed by explicit adaptive thermal remapping at the requested cadence |

`full` is a directly timed composition, not a sum of separate microbenchmarks.
It is a fully wet 100 m ocean with no coasts, sea ice, atmospheric flux calculation
or file output. Shortwave, longwave, sensible/latent heat, precipitation/evaporation
and wind stress are prescribed nonzero synthetic inputs. No surface TKE source.
`full_ale` remaps **after physics**, every `--remap-every N` steps, to equal layer
fractions at the current depth. The initial layer thicknesses vary horizontally,
so its first remap is nontrivial. Use enough steps to include the desired cadence;
the first remap cost may differ from subsequent ones.

`--closures supported semi fast` compares the installed TKE closure with the
research semi-implicit version and its faster thermodynamic implementation.
Only `semi`/`fast` use `--mixing-substeps`; `supported` takes one mixing step.
Consequently `supported` versus `fast` is not a matched-accuracy speed comparison.
`semi` versus `fast` holds the formulation and substeps fixed.

## Interpretation

* Setup, warm-up, output writing and endpoint diagnostics are outside the timer.
  Internal validation, reports and physical budget guards remain inside it.
* Every repeat resets to the same initial state, then evolves for the requested
  steps. Warm-up is separate and never becomes the starting state for timing.
* Median, minimum and maximum wall seconds/step are reported. Process CPU time
  is also recorded but is **not** used for SYPD; CPU time can exceed wall time
  with several threads. Peak RSS includes setup and is normalized to bytes on
  macOS and Linux.
* `SYPD = timestep_seconds / (365 * wall_seconds_per_step)`. For mixing alone,
  this is an operator-equivalent rate, not whole-model throughput. Actual
  timestep and fixed mixing substeps are in every row. No extrapolation from a
  different timestep is hidden in the score.
* Reports include endpoint mass/salt/tracer/energy, TKE minima, limiter counters,
  external energy input and remap energy change. Energy drift of a forced
  dynamics run must be interpreted after subtracting prescribed external inputs;
  closure of the mixing budget does not imply zero SSPRK3 drift.
* A short accepted run does not qualify 600 s stability, profile accuracy,
  climate throughput, backward differentiation, or GPU execution. State-dependent
  iteration costs may change over long runs. Failed and timed-out cases get no SYPD.
* This campaign deliberately preserves float64 on CPU. It does not provide an
  MPS/float32 comparison. Keep the Mac otherwise idle, disable competing workloads,
  and repeat a representative case to assess temperature/background-load effects.

`summary.md` is a compact table; `summary.json` contains the campaign manifest.
Per-case JSON includes raw timings, endpoints, package/source fingerprints,
thread counts and platform/build information. Logs preserve errors. Return the
summary plus the relevant case JSON when comparing machines. Do not upload any
individual artifact above 128 KiB; keep it local.

### Validated field reuse and warm inversion

`StratifiedOcean` reuses checked physical fields within a step and precomputes
layer connectivity. Rebuild the model after changing mesh connectivity, its
active-edge mask, layer count, or device; stale prepared connectivity is rejected.
No thermodynamic fields are cached between public calls.

`StratifiedOcean(mesh, column, warm_start=True)` additionally starts stage entropy
inversions from the previous checked temperature. It retains the reference-domain
and residual checks, Newton-polishes the result, and falls back to the original
solver if necessary. The default is `False`; warm starts can change roundoff and
roundoff-sensitive limiter diagnostics. This option does not warm every mixing or
forcing inversion.

For independent wheel comparisons, unpack each wheel into a separate directory
and run `field_reuse.py --package /path/to/unpacked-wheel --output /tmp/sample.json`.
Use `--warm` for warm inversions or `--unprepared` to disable prepared connectivity.
Run samples sequentially, alternate wheel order, and repeat them. `.pt` tensor
captures are local comparison intermediates; do not commit oversized captures.
See `verification/ocean/FIELD_REUSE.md` for measured scope and limitations.

`compare_field_reuse.py parent.json candidate.json --output comparison.json`
checks every captured state/report tensor for exact equality. With
`--allow-roundoff`, it records per-field differences without asserting an
accuracy threshold; use this for warm-inversion diagnostics, alongside residual,
positivity, inventory and energy-budget checks.

The main campaign accepts `--warm-start` for the ocean dynamics in `ocean`,
`full`, and `full_ale`. It does not alter atmosphere or mixing-only calculations.
For example, compare separate output directories with and without this flag:

```sh
python -I validation/performance/run.py --sizes 1024x64 --threads 1 \
  --cases ocean full full_ale --dt 60 --warm-start --output perf-warm
```

### Energy quadrature experiments

`--energy-quadrature 2` selects four EOS locations per layer instead of the
sixteen used by default (`4`). It changes the discrete Hamiltonian; operators,
mixing and remap corrections use the same selected energy. Order `3` uses nine
locations. The option applies to ocean cases and does not change atmosphere
calculations. See `verification/ocean/ENERGY_QUADRATURE.md` for accuracy limits.
The corresponding model constructor option is `quadrature_order=2`.

The isolated `field_reuse.py` harness additionally accepts `--compile-thermo`
and `--compile-gibbs` for research kernel-fusion experiments. The combined flags use per-model compilation; individual flags isolate kernels
for research comparisons. Checks are retained. Compilation warm-up is excluded
from step timings; compiled outputs may differ at roundoff.

The main runner accepts `--compile-thermodynamics` to select the combined path.
For direct use, call `ocean.compile_thermodynamics()` once after constructing the
model. This returns the same model, leaves other model instances unchanged, and
compiles lazily on first use. CPU forward qualification is limited; no Mac/MPS,
GPU or full-model backward performance claim is made.
