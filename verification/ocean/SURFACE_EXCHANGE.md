# Prescribed ocean surface exchange qualification

`OceanSurfaceExchange` applies supplied precipitation/evaporation mass rates,
sensible/latent heat, water-carried energy and native momentum fluxes. It changes
top-layer thickness while retaining salt and nonvolatile passive contents.
Actual column Hamiltonian changes close against the supplied energy after a
local top-layer thermal solve. All water-carried energy and momentum conventions
are explicit caller inputs; no atmospheric bulk calculation is performed.
See [API and conventions](../../docs/OCEAN_SURFACE_EXCHANGE.md).

Installed-wheel portable suite outside checkout: **982 passed, one CUDA skip**.
Corrected-Gibbs (142 comparisons) and dry-helper (93 comparisons) portable
regressions passed with unchanged fixtures and tolerances.

## Focused tests

Fifteen tests cover rain/evaporation mass and concentration changes, native
momentum with zero and nonzero impulse, sensible/latent heat cancellation,
balanced opposing water rates, dry/coast masking, exact no-op behavior, required
carrier inputs, invalid shapes/rates, layer exhaustion and salinity rejection
without mutation, a smooth derivative, a manufactured full-energy/velocity
target, and combined exchange/radiation/stress/TKE stepping. Independent
Hamiltonian and momentum budgets are checked alongside exact scalar contents.

The manufactured test prescribes the energy difference and native impulse for
a known final mass, salinity, temperature and velocity. The thermal inversion
recovers the target temperature within 1e-10 C. This tests the energy convention
and inversion, not an empirical precipitation or evaporation law.

## Constant-flux subdivision

`validation/ocean_boundary/exchange.py` applies 3600 seconds of fixed freshwater,
heat and water-carried energy fluxes, in 1/2/4/8 substeps on a 32-cell varying-depth
four-layer ocean. Analytic final thickness and salt-content/thickness references
have maximum discrepancies 5.33e-15 m and 2.85e-14 salinity units. Salt/passive
contents remain exactly unchanged. Final temperature differs from the single
step by at most 2.49e-14 C; energy-input residual is <=8.33e-12 J/kg.
[Results](surface-exchange/subdivision.json). This isolates boundary exchange;
it is not a coupled temporal-order study.

## Combined forcing

`examples/ocean_surface_fluxes.py` runs three 60-second steps on 32 cells and
8 layers, with varying depth, supplied rain/evaporation/sensible/latent/carried
energy, radiation, wind stress, bottom drag and per-layer TKE. Net water input
is 0.1791 kg/m². Mass residual relative to initial water mass is -1.27e-16;
relative salt change is -1.80e-16. Energy change minus all supplied inputs is
-8.40e-12 J/kg; exchange-only residual is 2.64e-12 J/kg. Minimum final specific
TKE is 0.0002507 m²/s², from an explicit 0.001 initial seed.
[Results](surface-exchange/combined.json).

## Limits

Short CPU float64 trajectories and one smooth directional derivative only.
No atmospheric flux calculator, automatic water enthalpy/reference conversion,
ice, wetting/drying, volatile tracer exchange, rain microphysics or advective
TKE boundary flux. Extensive TKE is retained by the wrapper. The caller must
supply mutually consistent water energy, latent heat and momentum fluxes; the
operator cannot infer physical consistency from independent tensors. Shared-edge
kinetic allocation can affect adjacent-cell thermal corrections. First-order
process splitting and existing thermodynamic-domain restrictions remain.
Finite-energy checks are scaled by the thermal background; no relative-accuracy
claim for arbitrarily small flux increments, GPU or long climate runs is made.
