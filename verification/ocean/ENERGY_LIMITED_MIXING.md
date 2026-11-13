# Energy-limited adjacent-layer mixing

The scalar mixing update reduces a proposed paired exchange until its actual
Hamiltonian energy cost fits both adjacent TKE reservoirs. Conservative salt,
potential-enthalpy and passive-content fluxes share one coefficient. The original
implicit pair coefficient is an upper bound, so reduction preserves convex pair
mixing. Subsequent sweeps use the remaining reservoirs. There is no remote
borrowing, TKE floor, post-update clipping or compensating heat injection.

The energy-ratio reduction is a trial, not an assumption that work is linear.
Every trial reevaluates thermodynamics and full energy cost. After 48 failed
reductions the affected exchange is set to zero and checked again. Existing
energy, entropy and positivity guards remain. The factor one half provides
backtracking margin; it is a numerical choice, not a calibrated physical mixing
efficiency. Maximal admissible flux and limiter-parameter sensitivity remain
unqualified. Gradients at limiter switching points are not qualified.

## Evidence

Against the same initial profiles and forcing used in the original
[column campaign](COLUMN_CAMPAIGN.md), all 18 combinations of 16/32/64 layers,
300/150-second timesteps and wind/cooling/rain-heating now complete one hour.
Zero and 1e-12 m²/s² seed probes also complete one hour (16 layers, dt=300 s).
Maximum absolute energy residual across these 20 cases: 2.94e-11 J/kg.
See [records](energy-limited-mixing/short-matrix.json).

Zero TKE remains zero: this closure has K_m proportional to sqrt(k), so vertical
shear cannot bootstrap production from exactly zero turbulence. Successful
execution of this case is not evidence of physically adequate mixing. A source
or startup treatment needs independent physical justification.

Reproduce the short matrix with an installed wheel:

```sh
python -I validation/ocean_column/limited_campaign.py --output /tmp/limited-columns
```

This is an isolated f-plane column test using production physics, not a spherical
dycore, ALE, EKE, GPU, or coupled atmosphere qualification. Full active-profile
convergence and independent mixed-layer accuracy remain unresolved. The original
32-case 48-hour campaign has not been repeated in full; its small-step cases
become substantially more expensive when they no longer reject immediately.

Installed-wheel suite outside checkout: **985 passed, one CUDA skip** in
portable fixture mode. Exact mode: 982 passed, one skip and three historical
fixture failures (ocean archive capture, expanded dry baseline, checkpoint-54).
No fixture or tolerance was modified. Logs are retained beside the results.
CPU float64 forward behavior is tested; limited-branch backward behavior and
GPU execution are not qualified.

The 16-layer wind case with dt=300 s completes **48 hours** with active scalar
mixing. Final energy residual is -5.54e-11 J/kg; mass/salt residuals are zero,
and relative passive inventory change is 1.78e-16. Integrated transport is
(-0.975645592, -0.974436146) m²/s versus analytic
(-0.975609006, -0.974399605). See [48-hour evidence](energy-limited-mixing/wind-48h.json).
This single sustained case does not establish profile convergence.

```sh
python -I validation/ocean_column/run.py --layers 16 --dt 300 --hours 48 --output /tmp/wind-48h.json
```

Baseline parent is eef2fa97b822558b22cbfd1bc1cc14472ce38ede (draft #57).
Records were generated with a noneditable installed wheel, Python 3.12,
PyTorch 2.15.0+cpu. The 48-hour raw file is represented by selected profiles and
hourly energy errors; its size and hash are recorded. Oversized files are not
uploaded or split.
