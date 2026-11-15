# Snow-on-ice verification

The [zero-layer snow wrapper](../../../../validation/sea_ice/SNOW.md) adds snow
mass as a labelled subset of existing frozen-water stores. The existing common
Hamiltonian counts that mass once, including latent energy, loading and KE.
Mechanics use ice-equivalent geometry; porous snow depth controls conduction
and albedo. The snowfall boundary explicitly supplies mass, solid enthalpy and
mechanical incorporation work. No total-energy residual is repaired.

## Coupled experiments

Five CPU float64, one-thread experiments run the actual ocean/ice dynamics:
1980 steps, 54 aggregate model hours. Layers are 20/60/120 m. The first half
has air at -15°C, no solar input and snowfall 2e-4 kg/m²/s per covered area.
The second has air at +5°C, 400 W/m² sunlight and no snowfall. Wind is 4 m/s
at the equator. The coastal case starts with hemispheric ice and static land.
The complete-melt stress case instead uses +30°C and **20000 W/m²** in its
warm half; this deliberately unrealistic forcing tests loss of the entire
solid reservoir, not environmental realism.

| Case | Cells | Hours | dt (s) | Peak energy residual (J/m²) | Final snow (kg) |
|---|---:|---:|---:|---:|---:|
| [cycle32](cycle32/summary.json) | 32 | 12 | 120 | 1.26934e-05 | 0 |
| [cycle32-dt60](cycle32-dt60/summary.json) | 32 | 12 | 60 | 1.016e-06 | 0 |
| [no-feedback](no-feedback/summary.json) | 32 | 12 | 120 | 1.29226e-05 | 0 |
| [coast128](coast128/summary.json) | 128 | 12 | 120 | 1.78357e-05 | 0 |
| [full-melt](full-melt/summary.json) | 32 | 6 | 120 | 4.37468e-06 | 0 |

Residual is H(t)-H(0)-accumulated boundary energy divided by wet area. Water
residual also subtracts accumulated snowfall mass; salt has no boundary source.
Across these runs, peak relative water and salt errors are
6.16e-16 and
1.03e-15. All runs accumulate then
melt snow, and the strong-warming case ends with zero snow **and** zero ice.

The ordinary 32-cell cycle reaches about 1.30 cm snow depth. Just after the
warming transition, absorbed shortwave is about 220 W/m², compared with
243 W/m² when snow's optical/thermal effects are disabled. The control retains
the same mass/latent-energy processes but assigns snow the compact carrier's
density, conductivity and albedo. This is an illustrative feedback comparison,
not a calibration or an observed climate response. Independent fixed-state
tests isolate insulation and reflection separately.

Halving dt reduces the ordinary cycle's peak energy error from 1.27e-5 to
1.02e-6 J/m². This is evidence of timestep sensitivity, not a coupled-order
proof; phase-branch changes and source splitting affect trajectory convergence.
Observed times are roughly 0.09–0.11 s/step on these tiny grids. Other checks
shared the host; these timings are not a controlled overhead or GPU benchmark.

Each directory contains a step-by-step CSV and JSON with forcing, timing and
source hashes. Reproduce from the repository with the VFS wheel installed:

```sh
python -I validation/sea_ice/run_snow.py --output snow-cycle
python -I validation/sea_ice/run_snow.py --dt 60 --output snow-refined
python -I validation/sea_ice/run_snow.py --no-snow-feedback --output snow-control
python -I validation/sea_ice/run_snow.py --cells 128 --coast --patch --output snow-coast
python -I validation/sea_ice/run_snow.py --hours 6 --warm-air 30 --warm-sun 20000 --output snow-melt
```

## Independent checks

Fifteen new tests cover incoming snow mass/energy/momentum, retained salt,
snow-first melt, fixed underlying-ice volume and footprint during snow-only
melt, exact full-H phase derivatives, nonnegative constituent stores,
carrier-consistent transport/ridging, coastal cells, open water, reflected plus
absorbed shortwave, insulation direction and smooth surface backward sensitivity.
The bare zero-solar coupled step matches its predecessor bit-for-bit. Existing
ice tests and frozen tolerances are unchanged. `focused.log` records the new tests.

The complete portable suite passes **1308 tests, with 2 skipped** (205.08 s).
Both frozen comparisons pass: 142 corrected-Gibbs tensors and 93 dry helpers.

The external regression harness imports VFS from its installed wheel and copies
research modules explicitly. `wheel-tests.log`, `baseline.json` and
`dry-helpers.json` retain the portable regression and frozen-comparison evidence.

## Scope

Snowfall on open water must be supplied separately by the forcing driver; it is
not silently added by this snow-on-ice component. Snow is zero-layer, with no
internal sensible-energy storage, compaction/aging law, melt ponds, blowing-snow
loss, flooding-to-ice conversion or brine. Net melt preferentially consumes snow
without resolving distinct surface and basal fronts. Porous-snow gravitational
geometry is approximated by the existing ice-equivalent geometry. This gives
closure for the stated H, not a resolved snow-column Hamiltonian.

Only CPU float64 and selected local backward derivatives are qualified here.
Climate realism, long trajectories, compilation, float32 and GPU execution
remain untested. No packaged source or default changed.
