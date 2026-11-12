# Local ocean buoyancy-work qualification

The TKE scalar-mixing operator uses adjacent-layer budgets. This replaces the
initial column allocation documented in [the preserved first report](TKE.md).
The core ocean dynamics, shear operator, dissipation, TKE diffusion, horizontal
transport and ALE algorithms are unchanged.

## What is local

Each scalar substep contains disjoint neighbouring pairs. A closed implicit
exchange moves potential enthalpy, salinity and passives between the two layers.
The actual change in their combined JMD95/caloric energy is interface mixing
work. Its TKE debit or credit is distributed by adjacent half-layer thicknesses.
The remaining energy transfer between the layers is recorded as an interface
flux. Thus a layer's energy change is accounted for by local mixing work and
flux divergence, rather than by charging transported heat to turbulence.

A layer cannot use a remote reservoir or a favourable column-wide net buoyancy
work to pass its affordability check. Later physical TKE transport can move
energy; that is separate from the local work check. A zero local reservoir at a
costly interface can require more than reducing dt: a physical source/transport
or a future local scalar-flux limiter may be needed. No fallback is applied.

Scalar exchanges use even-half/odd-full/even-half ordering with frozen
conductances. This differs numerically from the original column tridiagonal
scalar solve and remains first order. See [equations and units](../../docs/OCEAN_TKE.md).

## Reproduce

Install a wheel and copy tests/validation outside the checkout:

```bash
python -I -m pytest -q tests/ocean/test_tke.py
python -I validation/ocean_tke/local_buoyancy.py --output /tmp/local-pairs.json
python -I validation/ocean_tke/run.py --output /tmp/local-columns.json
```

No datasets or new dependencies are required. CPU float64 is the tested scope.
The [pair evidence](tke/local-buoyancy.json) and
[column evidence](tke/local-column-results.json) include implementation/driver
hashes. The original #51 artifacts remain unchanged.

## Checks

Twenty-four focused tests retain the original 17 TKE checks and add:

- Rejection when the local pair has no TKE despite very large remote reservoirs.
- Exact unchanged remote state and TKE for an isolated active interface.
- Independent layer-energy changes against the reported interface flux, with
  heat transfer much larger than buoyancy work in the chosen example.
- Local results independent of remote reservoir magnitude.
- Interface work compared with the analytic instantaneous Hamiltonian
  derivatives in potential-enthalpy/salinity coordinates.
- Exactly uniform scalar fields produce no spurious buoyancy work.
- Unequal adjacent thicknesses receive the geometric 1:3 work allocation for
  the tested 1:3 thickness ratio, rather than reservoir-dependent allocation.
- A finite-difference derivative check around a uniform temperature state,
  ensuring exact no-op values do not suppress the linear mixing response.

Some checks share a test function, hence the test count is not the bullet count.
Full installed-wheel regression and unchanged portable atmospheric fixture
results are recorded in [the validation summary](tke/local-validation.txt).

## Recorded measurements

The isolated pair uses a 32-cell mesh, layer thicknesses 10/30/25/35 m and
only the top interface active. The mixing work is 5.46e-6 J/kg of full-column
mass. Its upper/lower allocation is 0.25/0.75. Remote entropy and TKE remain
bitwise unchanged; remote funding is rejected. Salt inventory error is zero
and potential-enthalpy inventory error is 6.55e-12 J/kg. The computed layer
ledger residual is zero at the recorded precision; this is an algebraic budget
check, not a claim of exact physical truth.

Against the exact continuous two-box solution at 100 s, maximum potential-
enthalpy errors for 1/2/4/8/16 steps are 0.1310/0.06567/0.03287/0.01644/0.008224
J/kg. The approximately halving error confirms first-order convergence.

The repeated mixing campaign retains the initial report's setup, with the local
scalar operator substituted (20 steps of 10 s):

| Layers | Combined H change (J/kg) | Maximum step/column residual (J/kg) | Final minimum k (m²/s²) |
| ---: | ---: | ---: | ---: |
| 4 | 1.03e-11 | 2.20e-11 | 4.95e-4 |
| 16 | 1.03e-11 | 1.01e-11 | 4.88e-4 |
| 32 | 0 | 9.84e-12 | 4.86e-4 |

## Interpretation limits

The pair experiment compares refinement against the independently known
continuous two-box diffusion solution, with frozen conductance. This tests the
exchange discretization; it does not calibrate ocean entrainment or a mixing
length. Repeated 4/16/32-layer column runs test budgets and admissibility.

The instantaneous work comparison uses a finite, small step that resolves the
buoyancy work above caloric-inversion roundoff. At vanishing dt or extremely
weak stratification, subtracting caloric energies limits relative work accuracy.
The layer budget identities still expose the actual computed changes; they do
not prove arbitrary relative accuracy for tiny physical work.

No prognostic length scale, wind/wave boundary source, bottom drag or EKE is
added. Physical mixed-layer validation, local flux limiting, long integrations,
GPU/float32 and long backward trajectories remain outstanding. Locality and
energy closure are necessary properties, not evidence of calibrated turbulence.
