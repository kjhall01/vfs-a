# JMD95 conservative-path mixing work

Research-only `PathWorkClosure` evaluates buoyancy work from analytic derivatives
along conservative potential-enthalpy and salinity exchanges. It retains JMD95,
the surface Gibbs caloric reference, native momentum metric, and the parent
coupled solver's midpoint energy flux and layer allocation. Supported VFS runtime
code, defaults, fixtures, solver tolerances and acceptance guards are unchanged.

## Discrete formulation

For fixed layer thickness h, let fH and fS be time-integrated upward content
fluxes. Define Df as incoming minus outgoing flux, with zero boundary fluxes.
The conservative path is H(lambda)=H0+lambda*DfH/h and
S(lambda)=S0+lambda*DfS/h, for 0<=lambda<=1. Temperature follows the existing
enthalpy inverse at each path point. At the Hamiltonian's vertical quadrature
points, the analytic conjugates are

    a = <Psi_T>/h0_T,
    b = <Psi_S> - a*h0_S.

Gauss-Legendre integration gives path averages abar and bbar (default order 3).
For alpha=h_upper/(h_upper+h_lower), define
face(q)=(1-alpha)*q_upper+alpha*q_lower. Interface work is

    W = rho0*A * [(abar_upper-abar_lower)*fH
                 +(bbar_upper-bbar_lower)*fS].

Allocate alpha*W to the upper layer and (1-alpha)*W to the lower layer, then add

    D {rho0*A [face(abar-amid)*fH + face(bbar-bmid)*fS]}.

Here amid,bmid are the parent's endpoint-midpoint conjugates. This correction
preserves the parent's local energy flux; omitting it would change local TKE
funding. Algebraically this is the path-integrated parcel-energy change minus
the original dynamic energy-flux divergence. Every term multiplies a content
flux, avoiding subtraction of nearly equal finite parcel energies. No forward
autograd is used. The conservative caloric transfer cancels algebraically.

Quadrature is approximate, and enthalpy/entropy inversions retain floating-point
errors. The parent reports actual finite-H reconstruction residuals and checks
full column H, entropy, nonnegative TKE and the actual local nonlinear budget.
No clipping, floor, compensating heat, or tolerance change is added here.

## Reproduction

Use an installed wheel outside the checkout with sibling copies of `validation`,
`tests`, `tools` and `examples`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/path_matrix.py --output results --long
python validation/ocean_column/analyze_path.py --input results --output summary.json
python -I -m pytest tests --fixture-mode=portable -q
```

The matrix retains transactional rejections rather than treating prefixes as
completed runs. Endpoint comparisons require equal completed times and grids.
The 75-second reference is numerical, not an analytic solution. Recorded campaign
wall times overlap other jobs and must not be interpreted as a speed benchmark.

## Scope

These are CPU float64 column experiments, not horizontal dycore, GPU, long
backward, dry-cell or coupled atmosphere-ocean qualification. Path averaging
adds enthalpy inversions per nonlinear iteration. Performance optimization and
long-run solver robustness remain separate requirements. The implementation
remains outside the installed package.

## Short-run results

All eight path-work one-hour cases complete: 16 layers at dt=600,300,150,75 s;
32 layers at dt=300 s; order-5 quadrature at 16/300; surface cooling and
rain/heating at 16/300. The matched accelerated finite-work 16/300 control
rejects at 40 minutes (iterate residual 74.799 against a limit of 1). The cautious
finite-work control completes, using 1296 iterations versus 190 for path work.

At the common one-hour endpoint, path work versus cautious finite work gives
velocity relative L2 difference 4.57e-10, TKE relative L2 2.54e-8, temperature
RMS 2.40e-10 C and maximum absolute TKE difference 1.43e-13 m2/s2. This supports
work-evaluation sensitivity as a cause of this failure without changing the
intended endpoint physics. It does not identify every possible solver failure.

Order 3 versus 5 gives velocity relative L2 3.29e-14, TKE relative L2 2.03e-12,
and temperature RMS 3.11e-15 C. These are endpoint sensitivities, not universal
quadrature-error bounds. Independent tests compare nonlinear layer work with
finite parcel-energy differences, orders 5 and 9, exact affine work on unequal
layers, zero flux and tiny-flux scaling.

| dt (s) | Velocity relative L2 vs dt75 | TKE relative L2 vs dt75 | Temperature RMS (C) |
|---:|---:|---:|---:|
| 600 | 0.01651 | 0.6799 | 0.0004552 |
| 300 | 0.007812 | 0.2939 | 0.0002357 |
| 150 | 0.002805 | 0.1006 | 0.00007822 |

Thus nonlinear convergence and energy closure do not establish temporal accuracy:
TKE remains substantially timestep-sensitive. The largest absolute one-hour
path-work energy residual is 3.47e-11 J/kg. Full raw profiles, solver rejection
messages and comparison metrics are retained in `path-work/`.

## Long attempt and regression checks

The requested 48-hour, 16-layer, dt300 run **rejects after 6 hours**, at the
attempted 6 h 5 min endpoint. Iteration and reconciled errors are both 0.00486001
(limit 1), but the actual layer-budget residual is 5.25097 (limit 1). The last
accepted full-energy residual is 2.07e-12 J/kg. No rejected step is accepted,
no tolerance is relaxed, and this is not a completed 48-hour result. The earlier
linear exact-work control also had a late budget rejection; cancellation-safe
work alone does not solve this remaining weak-TKE/local-budget issue.

Next investigate the rejecting layer's signed TKE budget, transport and source/
sink balance and the positive iteration's actual residual, rather than loosening
the gate. Retain the JMD95 path-work calculation as a research candidate; qualify
longer runs and timestep convergence before supported-model integration.

Installed parent wheel (runtime source unchanged), outside-checkout portable
suite: **1017 passed, 1 CUDA skip**, one existing tensor-to-scalar warning,
223.08 s. Seven new focused tests pass. Corrected-Gibbs and dry-helper portable
fixture comparisons pass. PyTorch 2.15.0+cpu. Logs and all campaign raw JSON
files are below 128 KiB. The failed long attempt and failed old-method control
remain in the evidence. No GPU or long-backward qualification is claimed.

## Fixed-input performance check

`benchmark_path.py` interleaves three repeats per method after warm-up on the
same frozen 16-layer, dt150 input, with one CPU thread and no other campaign
jobs running. It compares finite work with cautious iteration, path work with
the same cautious settings, and path work with accelerated iteration. This is
one input, not a sustained-run or dycore performance claim; see `benchmark.json`.

| Work / iteration | Median seconds | Iterations |
|---|---:|---:|
| Finite / cautious | 4.916 | 122 |
| Path / cautious | 11.142 | 122 |
| Path / accelerated | 1.722 | 18 |

The path calculation costs more at equal iteration count. Its ability to use
accelerated iteration makes this particular comparison about 2.85 times faster
than the cautious finite-work reference. Do not generalize that ratio to the
failed long rollout or a mesh-wide model.
