# VFS–PyAM21 workflow

An opt-in adapter and prescribed-SST integration **outside the VFS package**.
It calls the actual AM2.1 physics in Evan Wellmeyer's
[PyAM21](https://github.com/evanwellmeyer/PyAM21), pinned to
`e1bdfcc3b23c05adea80b35af880702bb85718eb`. Neither `vfs/` nor upstream PyAM21
is edited. VFS's installed dependencies and defaults are unchanged.

This first pass establishes state transfer, water/momentum accounting, explicit
physics memory, restarts, and a short dynamics/physics workflow. **It does not
establish VFS Hamiltonian closure for AM2 physics, or connect an interactive ocean
or ice model.** Energy differences are measured and retained, not corrected.

## Install and run

Install the VFS wheel and the separate optional dependency:

```bash
python -m pip install .
python -m pip install -r workflows/pyam21/requirements.txt
git clone https://github.com/evanwellmeyer/PyAM21.git /path/to/PyAM21
git -C /path/to/PyAM21 checkout e1bdfcc3b23c05adea80b35af880702bb85718eb
python -I workflows/pyam21/run_column.py --source /path/to/PyAM21 --hours 1 --output column-check --verbose
python -I workflows/pyam21/run_aquaplanet.py --source /path/to/PyAM21 --hours 1 --dt 60 --dynamics-dt 15 --output aquaplanet-check --verbose
```

The checkout supplies its BOMEX **physics namelist**. These drivers initialize
an analytic profile, with a positive 1000 Pa lid, 70% initial relative humidity,
24 levels and prescribed SST. They do not reproduce the BOMEX SCM case or use
its large-scale forcing. The column driver holds horizontal dynamics off; the
aquaplanet driver adds actual VFS dynamics and a meridional temperature gradient.
Both calculate surface exchange, boundary-layer mixing, RAS convection, cumulus
momentum transport and stratiform cloud processes through PyAM21.

**These example runs have no radiation.** The optional radiation interface below
accepts PyAM21's radiation driver, but the GFDL input-data bundle is not downloaded
or bundled here. These short runs are integration checks, not climate simulations.

Use `-I` as shown: drivers append the workflow root after site-packages, and VFS
and PyAM21 resolve to separately installed packages. Output is text: every-step
`series.csv` and `summary.json` with configuration and source hashes. No binary
verification files or model dependencies are vendored into this workflow.

## Files and ownership

| File | Purpose |
| --- | --- |
| `adapter.py` | Staged upstream calls, finite physics update, exchange/budget reports, restart validation |
| `mapping.py` | Conservative column readout and explicit CP/cell/edge transfer operators |
| `state.py` | Atmospheric state, CP cloud fraction, physics memory, surface state, radiation cache and clock |
| `dynamics.py` | Existing bounded SSPRK3 with a passive cloudy/clear transport ledger |
| `setup.py` | Explicit configuration and analytic initial conditions for examples |
| `run_column.py`, `run_aquaplanet.py` | Standalone checks and text evidence |

`PhysicsState.atmosphere` is the unchanged VFS `MoistState`. Cloud fraction lives
at CP interfaces in the workflow, alongside PyAM21's diffusivities, convection
flag, gustiness, roughness and albedos. The physics clock uses whole seconds on
PyAM21's Julian calendar. All prototype verification is CPU float64.

PyAM21 uses gravity 9.8 m/s², while the VFS default is 9.80665. The example
explicitly sets `model.dynamics.kernel.gravity = pyam21.constants.GRAV` **before
initializing any state**. The adapter rejects unequal gravity, because different
pressure-to-mass conversions would introduce a systematic water-flux discrepancy.
This changes only the example's model instance, not VFS package defaults.

## Python interface

From an environment where the repository's `workflows` namespace is available:

```python
from workflows.pyam21.adapter import AM2Adapter
from workflows.pyam21.dynamics import step

# model is a CPU/float64 MoistAtmosphere, configured with gravity=9.8 first.
physics = AM2Adapter(model, namelist, start=(2000, 1, 1, 0, 0, 0))
state = physics.initialize(atmosphere, sst=prescribed_sst_kelvin)

# Isolated physics; optional omega is physical Dp/Dt in Pa/s on full levels.
state, report = physics.apply(state, 60)

# Or one sequential dynamics/physics interval, with four dynamics substeps:
state, report = step(physics, state, 60, dynamics_dt=15)
```

The split driver advances dynamics first and calls physics once at that interval's
clock time, passing zero dynamical tendencies into PyAM21 to avoid double counting.
It is a **first-order sequential composition**. Physics memory is not advanced
inside Runge–Kutta stages. The cloud transport wrapper uses the existing atmospheric
Euler maps and the same SSP combinations; tests check bitwise atmospheric equivalence.
Cloud/clear amounts use the native carrier graph and bounded transfer routine.
The current cloud wrapper supports bounded SSPRK3, not IMEX.

For convection, `pressure_velocity` converts the relative hybrid-coordinate mass
flux to a physical pressure-velocity diagnostic by adding coordinate motion and
horizontal pressure advection. Full-level interpolation is a diagnostic approximation;
it does not change VFS continuity. Column-only calls default to zero pressure velocity.

## Mapping and mass conventions

Readout averages interface water fractions to full levels; the associated pressure
weights preserve each column's integrated species amounts. Full-level temperature
is obtained from the native two-point **mixture enthalpy** integral and averaged
composition. Thus the column enthalpy readout matches the native thermal + potential
+ lid energy (apart from fixed surface geopotential) at roundoff. Native hydrostatic
heights and Simmons–Burridge full pressures are supplied explicitly.

Returning physics increments never replaces the original CP profile with an
interpolated round trip. Layer temperature increments use the pressure-weighted
transpose. Species additions are split equally across adjacent CP nodes; withdrawals
use each node's contribution to that layer's species store. This preserves supplied
amounts without making a clear node negative when a neighboring cloudy node loses
condensate. Cloud-fraction increases/decreases draw from clear/cloudy contributions.

The water tendencies are interpreted as amount sources on the **old** pressure
carrier: `delta W_layer = dt * dp * rdt_water`. Their column sum updates surface
pressure, retaining dry-air inventory. Water contents receive precisely those
sources; fractions are then recomputed on the new hybrid-coordinate mass weights.
This is an explicit finite source convention, not a claim of exact equivalence
to AM2's fixed-pressure fraction update or a resolved vertical mass-source remap.
Entropy is reconstructed from updated CP temperature, pressure and composition.

Wind readout uses VFS's existing cell-vector reconstruction. Column forces return
through its analytic transpose and the native edge mass metric. The tests check
the work identity independently using differentiation. Mass sources/departures
are assumed co-moving with the native air, and the resulting carrier impulse is
reported separately. Complete precipitation/vapor-carried momentum and energy
matching to a moving ocean is deferred.

## Exchange outputs and energy audit

`PhysicsReport.fluxes` exposes the **post-implicit-coupling** sensible heat,
water vapor and both stress components, plus liquid/frozen precipitation. These
outputs come from PyAM21's four public stages, rather than reconstructing stress
from friction velocity. Staged results are tested against its top-level `forward`.

| Fields | Units and signs |
| --- | --- |
| `flux_t` | sensible heat W/m², positive into atmosphere |
| `flux_q` | vapor kg/m²/s, positive into atmosphere; can be negative |
| `flux_u`, `flux_v` | east/north stress N/m² acting on atmosphere |
| `lprec`, `fprec` | liquid/frozen precipitation kg/m²/s, positive downward |
| `sw_net_surface`, `lw_net_surface` | net radiation W/m², positive into surface |
| `lw_down_surface` | downward longwave W/m² |
| `toa_net_down` | net shortwave + longwave W/m² entering at the top |

Water reports include the species changes, dry-air residual, mapping residual,
and actual water change minus `dt*(evaporation - precipitation)` in kg/m².
Momentum reports separate the column-physics impulse, co-moving mass impulse,
and independently recomputed native metric residual.

Energy reports retain all of these per-cell values in J/m²:

- the actual native full-H and kinetic-energy changes;
- the VFS-mixture column enthalpy change evaluated on AM2's fixed-mass endpoint;
- AM2's conventional `cp_air*T + Lv*qv - Lf*qi` thermal/latent proxy change;
- the proxy's sensible/latent/radiative forcing ledger, with frozen-precipitation
  latent reference included;
- the difference between thermodynamic conventions, and the remaining difference
  from CP mapping, carrier change and geometry.

The conventional heat ledger is **not** the complete energy carried by exchanged
mass and momentum. The named differences are diagnostics, not assertions of error
in the upstream Fortran port. VFS uses composition-dependent heat capacities and
different reference constants, and finite mass/coordinate updates introduce further
terms. No temperature rescaling, global energy correction or hidden source is used.
The next thermodynamic pass must reconcile those terms before interactive ocean/ice
coupling can claim a shared Hamiltonian budget.

## Radiation, memory and restart

An externally constructed `pyam21.radiation.driver.RadiationDriver` can be passed as
`radiation=driver`, together with `radiation_id="..."` identifying its package and
input-data/configuration manifest. The adapter captures its LW/SW alarms, saved
fluxes/heating and solar angle in `PhysicsState.radiation`. The ID is part of restart
validation; change it when external tables or climatology mapping change.

Surface and physics dictionaries are cloned without detaching gradients. Radiation
cache state is installed temporarily, collected on success, and restored in a
`finally` block, including on failure. One adapter instance is not reentrant or
thread-safe. Data interpolation caches upstream are read-only memoization, not
prognostic physics state.

`physics.pack(state)` returns only tensors and primitive containers; save it with
`torch.save`. Restore with `physics.unpack(torch.load(path, weights_only=True))`.
The fingerprint covers actual installed PyAM21 source, namelist, thermodynamics,
start date, radiation configuration/ID and relevant geometry. Matching configuration
and mesh must be recreated before loading; the payload is not a self-contained model.

## Rejection and tested scope

Invalid fields, nonpositive carrier/vapor/dry air, substantial negative condensates,
cloud fractions outside bounds, unsupported surfaces and fractional physics seconds
reject. No automatic retry, flux reduction, moisture floor or state clipping is
introduced. Existing VFS condensate roundoff is admitted unchanged; upstream
finite-step cloud fraction roundoff up to 128 float64 eps beyond [0,1] is likewise
retained rather than clipped. Tests distinguish that from substantive violations.

Radiation scheduling, exposed flux signs and cache restart are tested using the
real PyAM21 driver with a **synthetic radiation package**. Full `SeaEsfRad` with
the GFDL data bundle has not been exercised in this workflow. Selected smooth
backward checks do not qualify threshold crossings or long trajectories.

See [verification results](../../verification/atmosphere/pyam21/README.md).
