# Boundary stress qualification

CPU float64. Prescribed surface stress, stationary flat-bed drag, native
momentum/work accounting and drag-to-bottom-TKE transfer are implemented as an
optional `TKEOcean` composition. Original atmospheric and ocean dynamics,
fixtures and tolerances are unchanged. No automatic merge is implied.

## Independent column references

`validation/ocean_boundary/run.py` exercises shared wind/drag/diffusion kernels.
The steady reference is the finite-depth constant-viscosity analytic Ekman
spiral, averaged over layers. Its transport is -0.9756097561 i m²/s for
rho0=1025 kg/m³, f=1e-4 s⁻¹, stress=0.1 Pa, depth=200 m and nu=0.01 m²/s.

| Layers | Relative steady profile error |
|---:|---:|
| 16 | 5.3311% |
| 32 | 1.5540% |
| 64 | 0.4023% |
| 128 | 0.1014% |

The 32-layer six-hour startup uses an independent matrix exponential reference.
Relative errors for dt=600/300/150/75 s are 3.3758/1.6906/0.8460/0.4232%.
This is first-order temporal convergence; work minus kinetic energy and
implicit diffusive loss closes within 4.55e-13 J/m². Fixed-direction quadratic
spin-down agrees with the analytic solution within 1.56e-15 relative error.
See [machine-readable results](boundary/reference.json).

These are isolated constant-viscosity f-plane checks, not a replacement for
spherical rotation validation or a claim of physically calibrated TKE mixing.

## Spherical integration

The [example](../../examples/ocean_boundary_stress.py) runs 32 cells, 16 layers,
100 m depth and five 10-second steps with resolved dynamics, wind, drag and
TKE mixing/dissipation. It explicitly seeds k=0.001 m²/s². Surface work is
1.68788e14 J; drag transfers 3.81822e13 J into TKE. The summed boundary energy
residual is -0.3105 J (about 2e-15 of wind work). The full-step energy change
minus external wind work is 8.50e-12 J/kg; that includes dynamics and thermal
inversion error. Minimum final k is 0.0006135 m²/s².
See [results](boundary/spherical.json). This is a short smoke test, not a
resolution study of a realistic basin or a long-run stability result.

## Regression scope

Installed-wheel portable suite outside checkout: **941 passed, one CUDA skip**.
Corrected-Gibbs (142 tensor comparisons) and dry-helper (93 comparisons) portable
regressions passed without changing fixtures or tolerances. The existing test
warning concerns a differentiable timestep converted to a scalar.

Fifteen focused tests cover wind work of either sign, finite-step drag loss,
large-step drag without reversal, one-layer overlap, rest, unequal neighbouring
thicknesses and independent column kinetic allocation, coasts/dry cells,
default no-op behavior, invalid input and slope rejection, smooth directional
derivatives, combined stepping and analytic spin-down.

Separate surface TKE input, sloping-bottom drag, seabed heat partition,
roughness/length-scale calibration, evolving-TKE Ekman profile accuracy,
long trajectories, GPU and float32 remain unqualified or absent.
