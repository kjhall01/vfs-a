# Optional GM–Redi research operator

`eddy.GMRedi` adds prescribed-coefficient eddy transport to the **Roquet CT/SA
research configuration**. It transports CT, SA, every passive tracer and the
existing turbulent TKE concentration. That TKE is not mesoscale eddy energy.
Layer thickness and resolved velocity are unchanged during this operator.
Nothing enables it automatically; the supported installed ocean API and all
existing defaults remain unchanged.

This is a **potential-density, small-slope GM–Redi-like prototype**, not a fully
neutral, energetically closed climate parameterization. Its purpose is to make
the discrete circulation, diffusion and energy exchange inspectable before
adding more closure physics. The prescribed coefficients are independent and
in m²/s. The examples' 1000 m²/s is an experiment choice, not a calibration for VFS.

## Use with the current ocean

As with `roquet.py`, put `validation/temperature_coordinate` and
`validation/ocean_column` on the Python import path. Use the same CT/SA state as
the timestep campaign:

```python
from eddy import GMRedi

eddy = GMRedi(kappa_gm=1000., kappa_redi=1000.,
              max_slope=.01, min_n2=1e-8)

# Explicit operator splitting; call every timestep, before any optional remap.
state, physics_report = model.step_with_report(state, dt)
state, eddy_report = eddy.apply(model, state, dt)
# If desired: state, remap_report = model.remap_with_report(state, target_h)
```

The model must be `RoquetOcean`. Slopes are rebuilt from the input state on every
call and frozen during that GM-then-Redi step. The operator has first-order time
accuracy even when the resolved dynamics use SSPRK3. Choose smaller explicit
substeps if its CFL/Redi bound rejects; this routine does not silently change dt.
There is no CLI option added to the existing timestep campaign.

Run the reproducible isolated-front experiment with:

```sh
python validation/temperature_coordinate/eddy_campaign.py \
  --cells 32 --layers 8 --radius 100000 --days 2 \
  --dt 3600 1800 900 450 --kappa 1000 --output eddy-front.json
```

This small-radius sphere accelerates lateral relaxation. It has no resolved
momentum evolution, forcing or vertical mixing; it is not an Earth simulation.
The separate composition test exercises resolved dynamics followed by eddy
transport. Results and limitations are in
[the verification report](../../verification/ocean/gmredi/README.md).

## Geometry and equations

Layer indices increase downward; physical height z increases upward. Define
surface-referenced buoyancy from the Roquet density anomaly:

    b = -g/rho0 * anomaly(CT, SA, z_reference=0).

For an edge from cell i to j, let dn be its center-to-center distance, ell its
face width, Dz a vertical derivative and angle brackets an adjacent-cell mean.
The physical normal component of the density-surface slope is

    S = (z_j-z_i)/dn - [(b_j-b_i)/dn] / <Dz b>.

The first term corrects the horizontal difference for unequal layer-center
heights. On a vertically linear, horizontally flat density profile it cancels
the second term, even on tilted model layers. Interior Dz rows average the two
adjacent secants; endpoints use one-sided secants. At least two layers are required.

If `<Dz b> <= min_n2` the slope and whole Redi row are disabled. Otherwise S is
explicitly capped at ±max_slope. These are declared slope regularizations, not
clipping of prognostic tracer concentrations. The potential-density directions
omit thermobaricity; they are not exact neutral directions in the nonlinear EOS.

### GM circulation

At internal interfaces, average the two neighboring layer slopes and set
`Psi = kappa_gm * S_interface`; set Psi exactly zero at the top and bottom. The
horizontal volume transport per layer is

    Q_h = ell * (Psi_lower - Psi_upper).

Construct downward interface transports from the cumulative horizontal inflow
above each interface. Their divergence cancels horizontal divergence, and
column-integrated Q_h telescopes to zero. Top/bottom fluxes and wet–land face
fluxes are closed. This is the finite-volume counterpart of

    u* = -partial_z(kappa_gm S),   w* = div_h(kappa_gm S).

Use donor advection of all concentrations along this circulation. Its outflow
fraction per cell-layer must not exceed `cfl` (default 0.8). Donor advection adds
numerical mixing; this is not a nondiffusive parcel rearrangement.

### Redi diffusion

Define one directional gradient row per active edge/layer:

    G q = (q_j-q_i)/dn + (S - (z_j-z_i)/dn) * <Dz q>.
    W   = kappa_redi * ell * dn * <h>.
    M   = diag(cell_area * h).
    dq/dt = -M^-1 G^T W G q.

This is a discrete small-slope rotated diffusion operator, expressed in the
current layer coordinates. Since G annihilates constants, inventories are
conserved. Since W is nonnegative,

    d( q^T M q / 2 )/dt = -(Gq)^T W(Gq) <= 0.

The Euler step uses a conservative Gershgorin bound on the operator spectral
radius, requiring `dt * bound <= cfl`. This ensures variance stability in exact
arithmetic, **not a maximum principle**. A localized nonnegative tracer can go
negative even at a small dt. Such a step raises without mutating its input; it
is not clipped or silently repaired. Positivity-preserving neutral diffusion
is an outstanding requirement for general tracer/biogeochemistry use.

With frozen slopes, the Redi density nullspace applies to the input buoyancy.
GM changes that buoyancy before Redi is applied, so the combined finite step
can have additional Redi energy exchange. This lagging error decreases with dt.

## Energy and diagnostics

`EddyReport` retains tensors and their gradient graphs. Transport orientation
matches `cellsOnEdge` for active wet–wet edges in original edge order. Vertical
transport is positive downward. Report fields:

| Field | Meaning / units |
|---|---|
| `slope` | Regularized normal physical slope, edge × layer, dimensionless |
| `streamfunction` | Interface Psi, active edge × (layer+1), m²/s |
| `horizontal_transport`, `vertical_transport` | Signed volume transports, m³/s |
| `volume_residual` | Combined incoming volume rate, cell × layer, m³/s |
| `gm_outflow_fraction`, `redi_step_number` | Local explicit-step restrictions, dimensionless |
| `inventory_change` | Concentration-update volume integrals in CT, SA, passive(s), TKE order; concentration units × m³ |
| `redi_variance_change` | Redi-only change of half volume-weighted squared concentration, same field order |
| `energy_change`, `gm_energy_change`, `redi_energy_change` | Finite Hamiltonian changes, J |

At fixed h/u the Roquet Hamiltonian is quadratic in CT and linear in SA/TKE.
The diagnostic evaluates the exact finite polynomial difference using changed
contents; it avoids subtracting two large global energy totals. An independent
autograd midpoint-gradient test checks this work formula. Forward transport
uses no autograd-generated tendencies.

GM normally removes resolved available potential energy, but this prototype
does **not** place it into an eddy-energy reservoir or heat. Report the measured
energy change in the budget; do not label it a numerical conservation residual.
There is no backscatter, eddy momentum forcing, prognostic EKE, energetic
constraint on kappa, or resolution-dependent taper. It also lacks mixed-layer
and bottom boundary tapering beyond zero interface streamfunction, exact neutral
geometry, and qualification on steep bathymetry/outcropping layers. These are
reasons to keep it opt-in while testing, not claims that the full problem is solved.
