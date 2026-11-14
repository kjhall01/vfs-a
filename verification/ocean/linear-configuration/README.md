# Linear ocean qualification

The [explicit configuration](../../../validation/temperature_coordinate/LINEAR.md)
uses affine density and constant-cp calorics throughout its Hamiltonian,
dynamics, TKE mixing, forcing and ALE. Supported JMD95 defaults are unchanged.
The research composition inherits bounded transport and local funding guards.

## Independent checks

The installed-wheel portable suite passed **1130 tests, one CUDA skip**, with
one existing warning. Separate portable fixture comparisons passed all 142
moist and 93 dry tensors. The historical ocean archive exact comparison returned
nonzero: 1411/1416 arrays match bitwise, with five source-term differences
no larger than 1.31e-18. Running the previous wheel gives the identical report;
no archive or tolerance was changed. The suite included the first 12 linear-configuration tests.
A subsequent focused run passed all **13** tests, including a new 6000-m column
check for analytic heating and uniform rest. No production expression changed
after the wheel build. The research heat adapter additionally checks shape,
dtype and device before arithmetic.

The tests cover:

- Analytic full-H derivatives versus independent autograd, including coasts;
  zero spatial work on the actual limited dynamics RHS.
- Constant-cp Gibbs face identity, symmetry and comparison with independent
  24-point integration; analytic entropy/enthalpy inverse round trips.
- Finite flux mixing work versus the actual Hamiltonian energy difference.
- Stable/unstable profiles with zero, 1e-12 and 1e-4 J/kg initial TKE;
  positive contents and unchanged local funding/energy acceptance.
- Complete forced step plus ALE with JMD95 partials, polynomial Gibbs evaluation
  and iterative caloric inversion explicitly disabled.
- Finite backward gradients through a small complete forced step; invalid
  configuration rejection. This is a smoke test, not gradient qualification
  at every limiter/acceptance branch.

## Sustained run

`day-32x16.json`: 32 cells, 16 layers, dt600 s, eight mixing substeps,
144 full steps (24 hours), ALE every step, float64 CPU, one thread.
Prescribed wind stress, bottom drag, shortwave/surface heat and freshwater
exchange are all active. Mesh: Fibonacci Voronoi, zero Lloyd iterations,
compatible rotation; 100-m water column and the fixed smooth thermocline/shear
profiles in the campaign. This deliberately small mesh is a budget/startup
exercise, not a resolved global circulation test.

- Energy minus external inputs: -1.9154e-8 J/kg.
- Mass minus freshwater input: -9.84e-15 relative.
- Salt/passive inventory changes: -8.39e-15 / -7.69e-15 relative.
- Minimum final TKE: 9.08e-21 J/kg; minimum salinity: 34.509.
- Maximum local mixing budget/tolerance ratio: 3.57e-7 (acceptance requires <=1).

Finite-step drift remains; exact timestep energy conservation is not claimed.
Near-zero final TKE also illustrates that positivity is not a turbulence/profile
accuracy test. This does not overturn the older coupled-column linear-EOS
control's failures: the algorithm and case are different.

## Throughput

Full dynamics + surface exchange/stress/heating + TKE mixing + ALE, dt60 s,
four mixing substeps, float64 CPU, one thread. Two warmup steps, three timing
samples and two sequential budget steps per configuration. Construction and mesh
generation are outside the timer; no test jobs intentionally overlap timings.
JMD95 uses the existing compiled thermodynamics and temperature-coordinate
three-point mixing path; linear uses analytic eager thermodynamics and exact
enthalpy-flux work. Both retain full acceptance checks. Raw samples are retained.

| Cells × layers | Compiled JMD95, s/step | Linear, s/step | Ratio |
|---|---:|---:|---:|
| 128 × 16 | 0.4378 | 0.1171 | 3.74× |
| 1024 × 64 | 14.4114 | 2.8995 | 4.97× |

The larger samples are noisy: JMD95 11.94–17.51 s, linear 2.56–3.86 s.
They were measured sequentially, not as alternating pairs. The linear
120-second energy-minus-input residual is 2.02e-11 J/kg; salt/passive
residuals are about 1.2e-16 relative.

These short CPU timings are exploratory, not a Mac/GPU/global scaling claim.
The EOS and calorics differ, so states and TKE profiles need not agree. No
matched physical-accuracy benchmark, long wave/jet test, convergence study or
climate validation is claimed. Linearization removes nonlinear thermodynamic
work but leaves mesh operations, tridiagonal mixing, funding retries and ALE
projection. Larger dt or arbitrary target grids can still reject.

## Reproduction

Run the commands in `LINEAR.md`; the day case uses
`--cells 32 --layers 16 --dt 600 --mixing-substeps 8 --steps 144 --ale`.
Timing cases use `--compare-jmd95 --steps 2 --ale` at the listed dimensions.
Research helpers/tests were copied outside the checkout alongside the installed
wheel. Python 3.12.14, PyTorch 2.15.0+cpu, no-grad timing, one Torch/OMP/MKL thread.

Wheel SHA256:
`77e7066f7a56699f13e164b27052b4095be2ef453308b706b8d8bb6431d1c6a1`.
The wheel is not uploaded. Raw JSON records include source hashes; compact
results and the suite log are retained. No large field arrays are published.
