# Energy-dependent GM and dissipation to heat

Reference: merged `ebab9dd961d4cf3f167276d38c45c174057917b3`.
Production defaults and ordinary prescribed-GM/Redi behavior are unchanged.

## Explicit physical choices

`EddyMixingLength(length=L, efficiency=c, kinetic_fraction=f)` supplies

```text
E_edge = (E_left + E_right) / 2
kappa_GM = c L sqrt(2 f E_edge)
Psi_interface = average_vertical(kappa_GM * density_surface_slope)
```

L [m], c [dimensionless] and f [dimensionless, 0<f<=1] are all explicit.
E remains total mesoscale energy [J/kg]; f is an assumed kinetic fraction,
not a new prognostic EKE/EAPE partition. The cell/layer reservoir evolves through
existing transport and actual finite Hamiltonian exchange. This adds feedback
from that energy to GM transport strength. There is no prescribed diffusivity
added on top, no hidden seed/floor, and no coefficient clipping. Zero energy
means zero GM coefficient; sqrt's gradient is singular at zero and the code
explicitly selects zero subgradient at that point. General differentiability
across zero-energy or rejection boundaries is not claimed.

Redi retains its independent prescribed diffusivity. An energy-dependent GM
coefficient does not guarantee every locally assigned withdrawal can be funded;
the original local funding rejection and explicit transport bounds remain.

This is a simple mixing-length research closure, not an implementation of
GEOMETRIC. For comparison, Mak et al. (2017), section 1, distinguish square-root
mixing-length scalings from their total-eddy-energy-linear GEOMETRIC coefficient:
https://doi.org/10.1016/j.ocemod.2017.02.007
Their paper motivates energy feedback but does not calibrate the present L, c,
f or the existing VFS local energy-flux allocation.

`EddyDissipation(timescale=tau)` removes a fraction `-expm1(-dt/tau)` of each
cell/layer's eddy content. It converts that energy directly to thermal energy
in the same cell/layer, leaving momentum, h, SA, passives and turbulent TKE
unchanged. This is a declared direct-to-heat closure, not a claim that mesoscale
eddy energy is vertical turbulent TKE. The decay fraction is nonnegative and
at most one for positive finite dt/tau, including complete numerical depletion.
No explicit sink CFL or clipped negative reservoir is needed.

The existing `RoquetWorkspace.heat` solves the quadratic full-H heat increment,
including the potential-energy change associated with CT-dependent density.
It does not use heat/cp alone. Its coefficients and finite-change diagnostics
come from the same Roquet Hamiltonian as dynamics, GM, Redi and the reservoir.
The ledger reports the actual stored reservoir debit, thermal energy gain and
local full-H residual. Stored CT introduces ordinary floating-point rounding;
no residual is hidden by resetting the reservoir or changing tolerances.

## API and stage order

```python
from eddy import GMRedi
from eddy_energy import EddyMixingLength, EddyDissipation

eddy = GMRedi(
    kappa_redi=100.0,
    energy_closure=EddyMixingLength(
        length=10_000.0, efficiency=0.1, kinetic_fraction=0.5,
    ),
)
dissipation = EddyDissipation(timescale=30 * 86400.0)

# state must explicitly contain the existing EddyState reservoir.
state, dynamics_report = model.step_with_report(state, dt)
state, eddy_report = eddy.apply(model, state, dt)
state, decay_report = dissipation.apply(model, state, dt)
```

These are illustrative parameters, not recommended/calibrated ocean settings.
The composition is explicitly first-order split: resolved step, GM, refreshed
Redi, then decay. Report `gm_diffusivity` contains the applied active-edge/layer
coefficient when feedback is enabled (None for prescribed GM). Dissipation is
separate so callers can inspect each transfer and select their stage order.
Backscatter remains a subsequent task.

## Verification

All 487 previously recorded output tensors, including one backward sensitivity,
are bitwise identical when feedback is disabled. Only the optional diagnostic
field is new. Thirteen new tests cover coefficient scaling/zero-energy behavior,
closed circulation, open/coastal energy budgets, decay from dt/tau=1e-6 to 1000,
zero-decay-state identity, split exponential behavior, unchanged fields,
independent polynomial thermal work, model-entropy increase, invalid inputs,
and backward finite differences. Existing 32 GM–Redi/reservoir tests retain
their assertions and pass alongside them (45 focused tests total).

`campaign.json` contains 15 experiments: three density/CT/SA front configurations
at dt=120/60/30 s over one hour, plus a 128-cell, 8-layer closed zonal band with
resolved dynamics, prescribed/feedback GM, optional decay and dt=60/30 s over
ten minutes. Maximum local closure residual is 3.64e-12 J/kg, and maximum
absolute cumulative closure residual is 1.67e-12 J/kg. Reservoirs stay positive.
The channel reports resolved-step energy change separately; it is not absorbed
into eddy energy. Direct total-H differences remain roundoff limited.

The zonal band has stable stratification, a meridional salinity gradient,
vertically sheared zonal flow and a small zonal perturbation. Its initial
continuum thermal-wind velocity is only approximate, with irregular masked
coasts. This is a short composition/budget check, NOT a demonstrated baroclinic
instability, equilibrated channel, eddy saturation result or physical calibration.
A long, balanced, resolution-refined channel experiment remains necessary for
those scientific claims. No new planar geometry was introduced for this pass.

Reproduce with the installed VFS wheel:

```sh
python -I validation/temperature_coordinate/eddy_feedback_campaign.py \
  --output feedback-campaign.json
python -I validation/temperature_coordinate/eddy_performance.py \
  --mode time --feedback --output feedback-timing.json
```

Installed-wheel portable suite outside the checkout: **1203 passed, 2 skipped**
in 194.72 s. The unchanged corrected-Gibbs and dry-helper fixtures pass all
142 and 93 tensor comparisons respectively. `installed-wheel-tests.log`,
`baseline.json` and `dry-helpers.json` retain the evidence. No production VFS
source or wheel was changed.

## Performance method

The performance driver interleaves prescribed GM, energy-dependent GM,
dissipation alone and combined feedback+decay on the same initialized
16384×40 wave state. Initial E=.1 J/kg and the benchmark-specific mixing length
give kappa_GM=1000 m²/s, matching the prescribed comparison. It also times
unchanged dynamics with and without the reservoir. Each operation warms once,
then runs three measured repetitions in a single process; CPU float64, one
thread, no compilation, no concurrent tests. Summing component medians estimates
step cost but is not a timed long integration. Source hashes and raw samples
are retained. GPU/MPS, float32, compiled closure timing and long backward
rollouts remain unqualified.

## Measured performance

| Operation | Median seconds |
| --- | ---: |
| Dynamics with reservoir | 16.517 |
| Prescribed GM–Redi | 2.420 |
| Energy-dependent GM–Redi | 2.218 |
| Dissipation alone | 0.172 |
| Energy-dependent GM–Redi + dissipation | 2.497 |

The directly timed combined closure adds 0.077 s (3.2%) relative to prescribed
GM–Redi. Summed with the same dynamics median, estimated step cost is
18.937 → 19.014 s (0.4%). The energy-dependent operator's lower measured median
is timing/cache variation, not evidence that computing diffusivity is free or
intrinsically faster. Dissipation alone ranges 0.165–0.309 s; combined closure
ranges 2.365–2.500 s, versus prescribed 2.373–2.460 s. These overlapping samples
support no large performance penalty on this setup, not a precise universal
0.4% overhead claim. Whole-process peak RSS is 3.234 GB, including dynamics.
This is not a paired attribution of memory overhead, and compiled Mac timing
still needs measurement. Raw measurements are in `timing.json`.
