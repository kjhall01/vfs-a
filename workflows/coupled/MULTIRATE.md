# Conservative multirate coupled workflow (experimental)

**Scope:** workflow scheduling, not new component numerics. This branch is
stacked on the experimental SSPRK(4,3) ocean–ice branch. Do not merge without
CI, cold/developed-state tests, and an end-to-end throughput comparison.

## Ownership and transactional boundaries

| Process | Owner | Scheduling control |
|---|---|---|
| Atmosphere bounded dynamics + cloud ledger | `workflows.pyam21.dynamics` | `atmosphere_dt` |
| Ocean/ice/snow dynamics + drag/phase/snowfall | `SnowOceanIce.step` | `ocean_dt` |
| Ocean column mixing/TKE | `SnowOceanIce.mixing_step` | `mixing_dt` |
| GM–Redi and eddy dissipation | `mesoscale_step` | `mesoscale_dt` |
| Atmospheric AM2 physics **and** surface exchange | `exchange_step` | `physics_dt` |
| Full SW/LW optical calculation | PyAM21 radiative alarm/cache | `radiation_dt` |
| Post-exchange ice equilibration | `SnowOceanIce.phase_step` | `physics_dt` |
| ALE conservative remap | `ale_step` | `ale_dt` |

`physics_dt` remains the **atomic coupling transaction interval**. It must
not be conflated with `radiation_dt`. No PyAM21 boundary flux can be applied
separately from its vertical-diffusion/column physics solve: the tile fluxes,
moisture transfer, enthalpy and reciprocal impulse are tied to a single
implicit solve and immutable `Exchange` record.

Longer `physics_dt` is already supported, but requires a *fresh run* or an
approved source/config transition. Changing cadence alters the physical
trajectory; conservation is a budget requirement, not an accuracy guarantee.

## Deterministic integer event schedule

`workflows/coupled/schedule.py` constructs ocean substep events within each
coupled exchange interval. Every timestep completes before an exchange, and
every ocean-closure accumulator empties before the transaction is committed.
That means no new pending state has to be serialized in restarts.

- Defaults `mixing_dt=None` and `mesoscale_dt=None` preserve mixing and
  GM–Redi after **every** ocean step. The existing in-step mixing position is
  preserved in this case.
- A slower `mixing_dt` defers the conservative column solve and executes it
  after the completed ocean/ice update at the cadence boundary. This changes
  the **operator splitting** and is a deliberate experimental choice.
- A slower `mesoscale_dt` similarly defers GM–Redi and eddy decay. The existing
  `eddy_substeps` is the **number per mesoscale call**, not per ocean step.
- `ale_dt=None` means ALE at every coupled exchange. A larger ALE cadence is
  anchored to absolute integer model seconds, independent of checkpoint
  segmentation, and uses the same Hamiltonian-consistent remapper.
- The final, potentially truncated run interval flushes closure accumulators
  once. Every other interval uses exact integer cadence boundaries.
- `mixing_dt` and `mesoscale_dt` must be integer multiples of `ocean_dt`
  and exactly divide `physics_dt`. `ale_dt` must be a multiple of
  `physics_dt`. The pre-existing divisibility rules for radiation and
  dynamics are retained. Unsupported schedules **raise an error**.

No closure is silently disabled; positivity, energy settlement, tracer
funding, and local admissibility guards are unchanged.

## Candidate 3° throughput schedules

Use a **new copy** of the existing 4,096-cell YAML, and change only these
fields; leave the tested source, thermodynamics, radiation tables, mesh and
other physics enabled. The existing SSPRK3 atmosphere has not yet qualified
for a 1,200-second dynamics step.

| Control | Current 40-level campaign | First multirate pilot | Stretch pilot (unqualified) |
|---|---:|---:|---:|
| `physics_dt` | 1200 s | **2400 s** | **3600 s** |
| `atmosphere_dt` | 600 s SSPRK3 | 600 s SSPRK3 | 600 s SSPRK3 |
| `ocean_dt` | 600 s SSPRK3 | 600 s SSPRK3 | 1200 s SSPRK43 |
| `mixing_dt` | 600 s | 1200 s | 1800 s *not valid* |
| `mesoscale_dt` | 600 s | 1200 s | 1200 s |
| `ale_dt` | 1200 s | 7200 s | 7200 s |
| `radiation_dt` | 1200 s | 7200 s | 7200 s |

The 3600-s pilot must choose mixing 1200, 2400 **invalid** (not a divisor),
or 3600; 1800 is not a multiple of a 1200-s ocean step. For an explicitly
valid 3600-s pilot use `mixing_dt: 3600` instead.

The table is a **test plan**, not evidence that the larger timesteps are
stable or accurate. In particular, fewer mixing calls with unchanged
`mixing_substeps` means larger individual mixing substeps: this may reject
on TKE affordability, entropy, positivity or budget checks. Keeping a
constant physical mixing substep requires increasing `mixing_substeps`
proportionally, sacrificing some of the speedup. Likewise, a larger GM–Redi
interval with unchanged `eddy_substeps` increases each explicit
subinterval and must satisfy the conservative Redi/GM admissibility tests.

**Recommended first pilot (1200-s closures, 2400-s coupling):**

```yaml
physics_dt: 2400
atmosphere_dt: 600
ocean_dt: 600
ocean_dynamics_integrator: ssprk3
mixing_dt: 1200
mesoscale_dt: 1200
ale_dt: 7200
radiation_dt: 7200
# Other source configuration parameters remain as in the reference experiment.
```

This reduces column-physics calls from 72 to 36/day, expensive optical calls
from 72 to 12/day, and ALE remaps from 72 to 12/day. Mixing and mesoscale
are reduced from 144 to 72 calls/day each, while dynamics keep their original
600-s baseline. Fewer calls **do not imply proportional wall-time savings**.
Time the actual coupled interval and inspect the internal substep counts.

## Validation before a long integration

1. Small fully coupled PyAM21 + ice smoke: 120-s exchange, 60-s ocean
   dynamics, 120-s mixing/GM, 240-s ALE; verify source ledger, positivity,
   snow/ice invariants, no secret state, exact event counts.
2. Check default-cadence trajectory equivalence on the *same branch*.
   Changing source/config adds explicit fingerprinted provenance; it does not
   authorize loading an older production checkpoint without an audited bridge.
3. Measure paired throughput for two 1200-s baseline intervals versus one
   2400-s candidate interval, using identical initial states and exact
   initial data. Keep each method's runtime, snapshots and budget reports.
4. Repeat from a developed coupled checkpoint and over multi-day trajectories:
   precipitation and cloud fraction, temperature and humidity extrema,
   surface heat/water fluxes, tropical SST, ocean TKE/EKE, ocean max speed,
   minimum layer thickness, global inventories and numerical energy.
5. Only if conservation and stability qualify, explore a clean split between
   AM2 physics and surface exchange. That requires new physically consistent
   time-integrated boundary fluxes, owner-managed propensity/accumulators
   in restart state, and changes to the implicit column solve. It is **not**
   accomplished by this scheduling PR.

## Known limitations

- AM2 RAS convection, stratiform clouds and turbulence are evaluated once
  per `physics_dt`; a longer interval is not a subcycled application of
  those routines. Nonlinearity and precipitation timing can change strongly.
- Radiation caches preserve their original driver-owned alarm/memory state,
  including repeated heating on non-optical intervals. A larger
  `radiation_dt` does not mean zero radiation between optical calculations.
- Delayed mixing and delayed mesoscale operators intentionally change
  splitting order and local states; do not claim bitwise equivalence except
  at the unchanged default cadence.
- When `ale_dt` exceeds `physics_dt`, evolving layer thickness may depart
  farther from the target stretched coordinate before remapping.
- The SSPRK43 atmosphere is **not implemented in this branch**. Use 600-s
  SSPRK3 for the initial multirate physics pilot.
