# Forced-column campaign

Run against an installed VFS wheel (CPU float64):

```sh
python -I validation/ocean_column/campaign.py --output /tmp/ocean-columns --workers 4
```

A single case is available through `run.py --layers 32 --dt 150 --hours 48
--scenario wind --output /tmp/column.json`. `--momentum-only` disables scalar
mixing for a diagnostic control. `--seed` changes initial specific TKE; primary
runs keep it fixed at 1e-4 m²/s². Failures are recorded, not retried, clipped or
counted as completed simulations. No production model code is changed.

## Geometry and scope

This is an isolated **f-plane column harness**, not a new dycore grid. One m²
cell has two orthogonal velocity degrees of freedom with native metric rho*h.
The incidence/interpolation adapter maps each component to the same column,
so its cell kinetic energy is exactly rho/2 sum h*(u²+v²). Tests independently
check that identity, wind impulse and rotation work. Scalar, momentum and TKE
mixing, boundary stresses, radiation, freshwater exchange and thermodynamics
are the installed production operators.

Rotation is the analytic u/v rotation at f=1e-4 s⁻¹, split half before/after
physics. There is no horizontal transport or pressure gradient, ALE, spherical
TRiSK qualification or basin boundary current in this test. The static bottom
has zero stress. Independent depth-integrated transport follows
M(t)=(tau/rho)*(1-exp(-i*f*t))/(i*f), where M=integral(u+i*v) dz. Closed vertical
mixing cannot change it. The split wind impulse introduces a temporal error
that should decrease under timestep refinement.

## Fixed experiment

- Water depth 100 m; 16/32/64 equal layers; f=1e-4 s⁻¹ (17.45 h inertial period).
- Initial upper fraction a(d)=0.5*[1-tanh((d-20 m)/5 m)].
- Potential temperature 10+5*a C; salinity 35-0.5*a; passive tracer a.
- Initial currents zero; k=1e-4 m²/s² everywhere; prescribed mixing length 2 m.
- Existing default C_mu=0.1, C_epsilon=0.7, Pr=sigma_k=1; no coefficient tuning.
- Eastward wind stress 0.1 Pa in every case, for a requested 48 h.
- `wind`: no heat/freshwater forcing.
- `cooling`: additional -100 W/m² surface heat from the start.
- `rain_heating`: additional 200 W/m² shortwave and -50 W/m² surface heat;
  precipitation 0.001 kg/m²/s for the first 6 h, then zero. Supplied water energy
  is rain rate times 50,000 J/kg in the model reference, with zero water impulse.
  Optical fraction/lengths are 0.6 and 0.5/20 m; bottom light is exported.

The initial profile is sampled at layer centres. Refinement changes that initial
sampling as well as the discrete closure. This is not an exact common
cell-average initial condition or a formal order-of-accuracy experiment.

Primary steps are 300/150 s. Additional wind probes use 30/3 s. Momentum controls
use 300/150 s and `mix_scalars=False`; they keep shear production, TKE transport
and dissipation, but cannot establish entrainment or scalar-mixing accuracy.
Two separate 32-layer/3-second probes use seeds 1e-3 and 1e-2; they are sensitivity
experiments, not replacements for failed primary cases.

## Acceptance and records

A case must reach 48 h without rejection to pass robustness. Inventory and
energy errors must be reported against actual external inputs, with the native
operator guards retained. Successful momentum controls must approach the
independent transport reference under timestep refinement. Mixed-layer depth,
profile and timestep refinement are **not qualified** if the active scalar
cases fail early; a guard rejection is a physical-campaign failure even though
it safely protects the numerical budget.

Hourly profiles record u, v, temperature, salinity, k and K_m. The reported MLD
is the contiguous surface thickness with T>=T_surface-0.2 C, a temperature
threshold diagnostic, not a density MLD (important with a fresh surface layer).
No observational or independent turbulence-model entrainment reference is
provided, so this campaign cannot certify real-world mixed-layer depth.

Full output can exceed the repository's per-file upload cap. Repository evidence
retains compact scalar summaries and selected profiles; oversized raw files
are omitted and listed explicitly, not split to bypass the cap.

Generate compact evidence and the figure (requires NumPy and Matplotlib only
for this optional analysis script):

```sh
python validation/ocean_column/summarize.py --input /tmp/ocean-columns --output /tmp/ocean-column-evidence
```

The runner uses parameter-based filenames; archived seed probes retain their
original shorter names. Compare parameters and numerical records, not filenames
or execution wall times. Profile comparison norms use the finer result as the
denominator, with pair-averaging for vertical refinement.

## Sustained limiter qualification

```sh
python -I validation/ocean_column/sustained_campaign.py --output /tmp/sustained-columns --workers 6
python validation/ocean_column/analyze_sustained.py --input /tmp/sustained-columns --output /tmp/sustained-evidence
```

This requests 48 hours for all 18 primary forcing/layer/timestep combinations.
Five additional 32-layer wind cases use dt=75 s, seeds 1e-6/1e-3 m²/s² at dt=150 s,
or backtracking safety factors 0.25/0.75 at dt=150 s. All other inputs remain fixed.
Existing result files are resumed; use a fresh output directory after code or
parameter changes. Progress logs record every simulated hour. This campaign
can take substantial CPU time; no failed case is silently shortened or retried.

Hourly profiles retain cumulative limiter diagnostics. Analysis uses thickness-
weighted RMS differences and conservative overlap averaging of the fine profile
onto coarse layers, including rain-induced surface expansion. Velocity relative
errors use the fine/reference vector norm. Absolute temperature/salinity RMS is
more interpretable than a percentage of their large background values. The
initial profiles are still centre sampled; no formal spatial order is claimed.
Seed and safety comparisons use the primary 32-layer/dt=150 s case as reference.
They are sensitivity tests, not comparisons with a known physical solution.

The sustained runner saves atomic hourly tensor checkpoints and resumes in
roughly two-minute execution segments (finishing the current simulated hour).
Restarting the runner skips completed cases and resumes the others. Checkpoints
validate the experiment parameters and TKE runtime source hash. A regression
check verifies exact agreement between resumed and uninterrupted evolution.
The local checkpoint files are not publication artifacts.
