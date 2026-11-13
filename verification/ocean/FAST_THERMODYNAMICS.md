# Faster thermodynamic evaluation for ocean mixing

`SubcycledLaggedClosure(2., substeps=4, fast_thermodynamics=True)` selects an
opt-in research implementation of the same semi-implicit JMD95 mixing update.
The default remains the retained reference implementation. JMD95, the discrete
Hamiltonian and its quadratures, native momentum/TKE allocation, conservative
scalar transport, funding retries and dissipation split are unchanged. This is
not the proposed frozen-conjugate work approximation and changes no physics-call
schedule.

## What is reused

A local workspace owns the layer quadrature pressures and reference-density
values for one fixed-geometry mixing substep. Actual parcel energies and their
derivatives retain the original expressions and summation order. The accepted
physical temperature is passed explicitly into the next mixing substep instead
of independently decoding the same entropy again. Nothing is cached on a model
or state object between calls.

Within an inversion, salinity-dependent Gibbs factors are calculated once. The
caloric kernel computes only energy/entropy and their temperature derivatives;
it preserves the original polynomial accumulation order. A nearby temperature
starts a safeguarded Newton solve, with the same declared bracket and original
128-epsilon residual criterion. Every accepted inverse is checked independently
using the original full reference evaluator. A final Newton polishing step
preserves the implicit first derivative even when the warm guess already meets
the residual threshold. This is not full-model backward qualification.

Heating evaluates only the JMD95 temperature derivative, omitting reference
terms whose temperature derivative is zero. It retains the same three-point
cancellation-safe increment integral, stops when its residual is satisfied, and
uses at most the original six Newton corrections. The independent full-H and
entropy checks remain. Cached physical fields enter the existing validation
routine only after their entropy residual has passed; all physical-domain,
finite-value, positive-salinity and tracer checks still execute. Public model
validation continues to reconstruct fields independently by default.

The scalar trial and path-work routine share their midpoint conjugates, and the
unused default buoyancy-work calculation is omitted. The original nonlinear
path-integrated work and its local flux allocation still determine whether TKE
can fund mixing. No state or negative TKE is clipped, and no tolerance is loosened.

## Reproduction

Build/install the wheel and copy the validation and tests outside the checkout.
The installed wheel includes the internal validation reuse hook; default
validation behavior is unchanged.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/benchmark_fast.py --cells 1024 --layers 64 --repeats 3 --output large.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/benchmark_fast.py --cells 128 --layers 64 --repeats 2 --output small.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/benchmark_fast.py --cells 128 --layers 64 --patch --repeats 2 --output patch.json
python validation/ocean_column/qualify_fast.py --hours 24 --output wind24.json
python validation/ocean_column/qualify_fast.py --hours 12 --scenario cooling --output cooling12.json
python validation/ocean_column/qualify_fast.py --hours 12 --scenario rain_heating --output rain12.json
python validation/ocean_column/analyze_fast.py --input results --reference baseline-resolution-results --output evidence
python -I -m pytest tests --fixture-mode=portable -q
```

Timing excludes one warmup for each method, then alternates baseline/fast and
fast/baseline order on identical fixed inputs. The benchmark cases execute
serially without overlapping qualification jobs. CPU and wall samples are both
retained. All cases use Torch 2.15.0+cpu, float64 and one thread. The mesh update
covers 300 seconds of vertical mixing with four internal substeps; it is not a
full global timestep. Numerical comparisons use the same current wheel for
both paths. The default path's output state and complete report also match a
separately loaded parent implementation bitwise on a pinned small native case.

Only the unmodified `SurfaceGibbs` and `JMD95Energy` kernels are eligible for
this optimized workspace. Alternative/custom energy kernels are rejected,
rather than silently evaluated using the specialized JMD95 expressions.
An initial large-batch probe exposed an overly narrow numerical-bracket check
on the polishing step. The corrected acceptance checks the physical bracket
and unchanged residual; a mixed-convergence batch regression covers this case.
The rejected probe log is retained with the evidence.

## Measured speed and agreement

| Cells–layers (patch where marked) | Baseline CPU s | Optimized CPU s | Speedup | TKE relative L2 difference |
|---|---:|---:|---:|---:|
| 128-64 | 11.680 | 2.964 | 3.94× | 1.290e-14 |
| 128-64-patch | 11.065 | 2.929 | 3.78× | 1.300e-14 |
| 1024-64 | 106.153 | 29.942 | 3.55× | 1.313e-14 |

At 1,024 cells × 64 layers the optimized update removes 71.8% of mixing CPU
cost. Individual optimized samples are 28.34–30.44 s; original samples are
105.74–109.02 s. The velocity relative L2 difference is 5.04e-17 and temperature
RMS difference 6.82e-15 C. Both paths use 11 trials and have identical counts of
limited interfaces and zero whole-column deferrals. Both have a final energy
residual of -1.03e-11 J/kg. The weak-TKE cap control retains its 12-trial count
and nonnegative TKE; its optimized energy residual is 5.13e-12 J/kg.

Against the previous provisional 27.41 s budget, 29.94 s remains 9.2% above
target. Using the earlier five-dynamics-step cost of 109.64 s gives an estimated
21.5% mixing share of the two-block total. Those dynamics measurements came
from a different session: this is planning context, not a measured coupled
throughput result or a validated multi-rate schedule.

The large timing run predates only the final unsupported-kernel type guard;
`source-provenance.json` verifies that this guard and its imports are the sole
source difference. Numerical expressions are unchanged. Small/patch benchmarks
and the long-column/installed-wheel checks include the guard. The unmodified
benchmark source records all input-source hashes and individual samples.

## Longer forcing and regression

| 64-layer case | Completed | Energy residual (J/kg) | TKE relative L2 difference from original | Temperature RMS difference (C) |
|---|---:|---:|---:|---:|
| Wind | 24 h | -9.03e-11 | 4.05e-13 | 3.72e-13 |
| Cooling with wind | 12 h | 3.43e-12 | 1.02e-13 | 4.94e-14 |
| Rain/heating with wind | 12 h | 2.48e-11 | 1.37e-13 | 8.69e-14 |

All three complete, with zero whole-column deferrals. Trial counts remain
3,455 for the 24-hour case and 1,727 for each 12-hour case. Thus the speedup
comes from cheaper evaluation, not fewer physics substeps, reduced retry
bounds, or hidden mixing suppression. These comparisons use retained original
resolution-campaign outputs; timing claims use only the interleaved benchmark.

Installed-wheel suite outside checkout: **1,052 passed, one CUDA skip**,
249.06 s, with the existing tensor-to-scalar warning. An additional composed
horizontal-dynamics/TKE-mixing regression passes separately (1 test, 6.47 s).
Corrected-Gibbs and dry-helper portable comparisons pass. New tests cover
independent AD checks of the JMD95 temperature derivative; exact caloric
evaluation; warm/fallback inverse residuals; inverse sensitivities to enthalpy,
entropy and salinity; mixed-convergence batches; physical-domain guards;
stable/unstable/zero/weak-TKE budgets; and finite heat increments.

No GPU, full-model backward, long global circulation, or multi-rate integration
qualification is claimed. The original vertical-resolution sensitivity and
strong local scalar limiting remain; faster evaluation does not resolve them.
The optimized path is explicit and opt-in, with the original path retained for
comparison. Three oversized raw long-column files are omitted, with compact
endpoint/hourly evidence and hashes in `fast-thermodynamics/manifest.json`.
All uploaded files are below 128 KiB.
