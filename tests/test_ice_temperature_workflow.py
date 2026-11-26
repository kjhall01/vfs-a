"""Independent sensible/latent budgets and conservative cold-content transport."""

from dataclasses import replace
import pytest
import torch

pytest.importorskip("pyam21")
from test_pyam21_workflow import model, initial, PrescribedRadiationPackage
from test_shared_exchange import surface_case, total_water
from workflows.coupled.ice_temperature import ThermalIceReceiver
from vfs.ice import OceanIce
from vfs.ice.thermal_phase import ThermalPhase
from workflows.coupled.driver import exchange_step
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig


def thermal_case(model, cover=True, temperature=260.0):
    receiver, bare = surface_case(model, cover)
    s = bare.ocean.resolved
    physical = s.with_tensors(s.u_e, s.h, -1.836 * s.h, s.salt_h, s.tracer_h)
    owner = OceanIce(receiver.ocean_ice.ocean)
    state = owner.state(bare.ocean.with_layers(physical), area=bare.ice.area, volume=bare.ice.volume,
                        velocity=bare.ice.velocity, snow_mass=bare.snow_mass, temperature=temperature)
    return ThermalIceReceiver(owner), state


@pytest.mark.parametrize("radiation", [False, True])
@pytest.mark.parametrize("cover", [False, True])
def test_combined_energy_with_prognostic_storage(model, radiation, cover):
    rad = (
        RadiationDriver(
            RadiationDriverConfig(rad_time_step=120), PrescribedRadiationPackage()
        )
        if radiation
        else None
    )
    adapter, air = initial(model, radiation=rad)
    receiver, sea = thermal_case(model, cover)
    A = model.mesh.cell_area
    E0 = model.diagnostics(air.atmosphere)["energy"] + receiver.ocean_ice.total_energy(sea)
    W0 = total_water(adapter, air, receiver, sea)
    out, end, x, ar, sr = exchange_step(adapter, air, receiver, sea, 60)
    E1 = model.diagnostics(out.atmosphere)["energy"] + receiver.ocean_ice.total_energy(end)
    assert (
        abs(
            float(
                (
                    E1
                    - E0
                    - (A * (x.toa + ar.energy_changes["am2_energy_residual"])).sum()
                )
                / A.sum()
            )
        )
        < 3e-6
    )
    assert abs(float((total_water(adapter, out, receiver, end) - W0) / A.sum())) < 1e-9
    assert sr.energy_residual.abs().max() < 3e-6
    if cover:
        assert not torch.equal(end.ice.cold_content, sea.ice.cold_content)
        assert (receiver.ocean_ice.temperature(end) <= 273.15).all()
    else:
        assert torch.count_nonzero(end.ice.cold_content) == 0
    # Tile thermal totals account for the common heat/enthalpy boundary exactly.
    expected = (
        x.sw_surface
        + x.lw_surface
        - x.terms["sensible"]
        - x.terms["vapor_enthalpy"]
        - x.terms["rain_enthalpy"]
        - x.terms["snow_enthalpy"]
    )
    torch.testing.assert_close(
        x.surface_thermal.sum(-1), expected, rtol=1e-12, atol=1e-8
    )


def test_conduction_backward_euler_and_snow_insulation(model):
    receiver, sea = thermal_case(model)
    owner = receiver.ocean_ice
    dt = 3600
    q = torch.zeros_like(sea.ice.cold_content)
    end, conducted, excess = owner.surface_heat(sea, sea, q, dt)
    p = owner.hamiltonian.slab.parameters
    C = owner.capacity(sea)
    hi, hs = owner.depths(sea)
    K = sea.ice.area / (
        hi / p.conductivity + hs / owner.parameters.conductivity
    )
    base = p.freezing_temperature(sea.ocean.resolved.salinity()[:, :1])
    expected = (C * (-13.15) + dt * K * base) / (C + dt * K)
    torch.testing.assert_close(
        owner.temperature(end) - 273.15, expected, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        end.ice.cold_content - sea.ice.cold_content, -conducted - excess, rtol=1e-12, atol=1e-8
    )
    snowy = replace(sea, ice=replace(sea.ice, snow_mass=0.2 * p.density * sea.ice.volume))
    insulated, _, _ = owner.surface_heat(snowy, snowy, q, dt)
    assert (owner.temperature(insulated) < owner.temperature(end)).all()


def test_surface_melting_complementarity(model):
    receiver, sea = thermal_case(model)
    q = -sea.ice.cold_content + 1e7
    end, conducted, excess = receiver.ocean_ice.surface_heat(sea, sea, q, 60)
    assert (end.ice.cold_content == 0).all() and (excess > 0).all()
    torch.testing.assert_close(
        end.ice.cold_content - sea.ice.cold_content,
        q - conducted - excess,
        rtol=1e-12,
        atol=1e-8,
    )


@pytest.mark.parametrize("heat", [-1000.0, 0.0, 2e6])
def test_phase_root_includes_cold_content(model, heat):
    receiver, sea = thermal_case(model)
    owner = receiver.ocean_ice
    A = model.mesh.cell_area
    before = owner.total_energy(sea)
    end, q, _ = owner.phase_step(sea, 60, torch.full_like(sea.ice.snow_mass[:, :1], heat))
    assert abs(float((owner.total_energy(end) - before - q) / A.sum())) < 3e-6
    if heat > 1e6:
        assert (end.ice.volume == 0).all() and (end.ice.cold_content == 0).all()


@pytest.mark.parametrize("active", ["none", "mixed", "all"])
def test_packed_bisection_matches_full_column_brackets(model, active):
    receiver, sea = thermal_case(model)
    rows = torch.arange(model.mesh.nCells)[:, None]
    # Alternate open water and existing ice. The mixed case includes both
    # new ice and snow-covered ice, with unevenly spaced active columns.
    old = sea.ice
    covered = rows % 2 == 0
    ice = replace(old, area=old.area * covered, volume=old.volume * covered)
    rho = receiver.ocean_ice.hamiltonian.slab.parameters.density
    sea = replace(sea, ice=replace(ice, cold_content=sea.ice.cold_content * covered,
                                   snow_mass=0.1 * rho * ice.volume))
    phase = ThermalPhase(receiver.ocean_ice, sea)
    initial = sea
    old_mass = phase.H.mass(initial)
    desired = torch.where(rows % 4 == 0, old_mass * 0.95, old_mass + 25.0)
    upper = torch.full_like(desired, 2000.0)
    target = phase.root_energy(initial, desired)
    melt_energy = phase.thermal_energy(phase.phase_state(initial, torch.zeros_like(desired)))
    frozen = {"none": rows < 0, "mixed": rows % 3 != 0, "all": rows >= 0}[active]
    target = torch.where(frozen, target, melt_energy + 1.0)
    assert torch.equal(target < melt_energy, frozen)
    with torch.no_grad():
        packed = phase._bisect_mass(initial, frozen, upper, target)
        full = super(ThermalPhase, phase)._bisect_mass(initial, frozen, upper, target)
    assert torch.equal(packed, full)


def test_phase_analytic_derivative(model):
    receiver, sea = thermal_case(model)
    phase = ThermalPhase(receiver.ocean_ice, sea)
    m = (phase.H.mass(sea) * 0.8).requires_grad_()
    value = phase.root_energy(sea, m)
    independent = torch.autograd.grad(value.sum(), m)[0]
    analytic = phase.derivative(sea, phase.phase_state(sea, m), m)
    torch.testing.assert_close(independent, analytic, rtol=1e-12, atol=1e-7)


def test_transport_and_ridging_preserve_cold_energy(model):
    receiver, sea = thermal_case(model)
    owner = receiver.ocean_ice
    A = model.mesh.cell_area[:, None]
    end, _ = owner.dynamics_step(sea, 1.0)
    assert (
        abs(float((A * (end.ice.cold_content - sea.ice.cold_content)).sum() / A.sum())) < 1e-8
    )
    assert (end.ice.cold_content <= 0).all()
    # Uniform specific enthalpy must stay uniform under carrier transport/ridging.
    torch.testing.assert_close(
        owner.temperature(end), owner.temperature(sea), rtol=1e-13, atol=1e-12
    )


def test_complete_thermal_step(model):
    receiver, sea = thermal_case(model)
    initial = receiver.ocean_ice.total_energy(sea)
    end, r = receiver.ocean_ice.step(sea, 1.0)
    assert (
        abs(
            float(
                (receiver.ocean_ice.total_energy(end) - initial - r.external_energy)
                / model.mesh.cell_area.sum()
            )
        )
        < 1e-3
    )


def test_bad_temperature_rejects(model):
    receiver, sea = thermal_case(model)
    with pytest.raises(ValueError):
        receiver.ocean_ice.state(sea.ocean, area=sea.ice.area, volume=sea.ice.volume,
                                 velocity=sea.ice.velocity, temperature=274.0)
    with pytest.raises(ValueError):
        receiver.tiles(sea, torch.full_like(sea.ice.snow_mass, 270.0))


def test_reduced_phase_energy_on_category_and_liquidus_branches(model):
    from vfs.ice.phase_energy import ReducedPhaseEnergy

    receiver, sea = thermal_case(model)
    ice = sea.ice
    row = torch.arange(model.mesh.nCells)[:, None]
    # Include open water, thin ice only, and two categories with different
    # snow-loss thresholds. Vary salt, liquid thickness and free surface.
    present = torch.cat((row % 3 != 0, row % 3 == 1), dim=1)
    ice = replace(ice, area=ice.area * present, volume=ice.volume * present)
    rho_i = receiver.ocean_ice.hamiltonian.slab.parameters.density
    water = sea.ocean.resolved
    h = water.h.clone()
    h[:, :1] += 0.3 * row
    h[:, 1:2] -= 0.25 * row
    salt = (row * 2.2).expand_as(h)
    water = water.with_tensors(water.u_e, h, -0.054 * salt * h, salt * h, water.tracer_h)
    ice = replace(ice, cold_content=sea.ice.cold_content * present,
                  snow_mass=rho_i * ice.volume * ice.volume.new_tensor([0.75, 0.125]))
    sea = replace(sea, ocean=sea.ocean.with_layers(water), ice=ice)
    phase = ThermalPhase(receiver.ocean_ice, sea)
    rows = row[:, 0]
    curve = ReducedPhaseEnergy(phase, sea, rows)
    old = phase.H.mass(sea)
    zero = torch.zeros_like(old)
    datum = phase.root_energy(sea, zero)
    # Cross partial snow melt, bare-ice retreat, growth at fixed thickness,
    # full-cover thickening, and nucleation on previously open water.
    for mass in (zero, old * 0.1, old * 0.5, old * 0.95, old,
                 old * 1.2 + 30, old * 12 + 112):
        energy = phase.root_energy(sea, mass)
        eps = torch.finfo(mass.dtype).eps
        error = (curve.value(mass) - (energy - datum)).abs()
        assert (error <= 16 * eps * (energy.abs() + datum.abs() + 1)).all()
        expected = phase.derivative(sea, phase.phase_state(sea, mass), mass)
        torch.testing.assert_close(curve.derivative(mass), expected, rtol=5e-13, atol=2e-8)
    # Independent AD check at smooth interior trial masses.
    mass = (old * 0.61 + 17).requires_grad_()
    derivative, = torch.autograd.grad(phase.root_energy(sea, mass).sum(), mass)
    torch.testing.assert_close(curve.derivative(mass), derivative, rtol=5e-13, atol=2e-8)
    # A tiny transfer can be below one ulp of absolute column energy.
    tiny = torch.full_like(old, 1e-15)
    torch.testing.assert_close(curve.value(tiny) / tiny, curve.derivative(tiny), rtol=5e-13, atol=2e-8)


@pytest.mark.parametrize("covered", [False, True])
def test_reduced_root_and_budgets_at_melt_boundary(model, covered):
    receiver, sea = thermal_case(model, cover=covered)
    phase = ThermalPhase(receiver.ocean_ice, sea)
    source = sea
    old = phase.H.mass(source)
    zero = torch.zeros_like(old)
    melt = phase.thermal_energy(phase.phase_state(source, zero))
    just_frozen = torch.nextafter(melt, torch.full_like(melt, -torch.inf))
    upper = torch.full_like(old, 1700.)
    near_upper = torch.nextafter(phase.root_energy(source, upper), melt)
    loads = torch.cat((just_frozen[:4], melt[4:8], near_upper[8:10], melt[10:12] - 3e6, melt[12:] + 1e6))
    wet = phase.H.model.energy.mass_area(source.ocean.resolved) > 0
    frozen = wet & (loads < melt)
    with torch.no_grad():
        actual = phase._solve_mass(source, wet, frozen, upper, loads, melt)
        reference = super(ThermalPhase, phase)._solve_mass(source, wet, frozen, upper, loads, melt)
    assert ((actual >= 0) & (actual <= upper)).all()
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=5e-13)
    assert torch.equal(actual[~frozen], zero[~frozen])

    flux = (loads - phase.thermal_energy(source)) / 600
    with torch.no_grad():
        end, report = phase.apply(source, 600, heat_flux=flux)
    before, after = receiver.ocean_ice.hamiltonian.total_energy(source), receiver.ocean_ice.hamiltonian.total_energy(end)
    area = model.mesh.cell_area[:, None]
    cold_delta = (area * (phase.cold(end.ice) - phase.cold(source.ice))).sum()
    assert abs(float((after - before + cold_delta - report.external_heat.sum()) / area.sum())) < 3e-6
    assert (report.water_residual / area[:, 0]).abs().max() < 5e-12


def test_phase_heat_flux_gradient_uses_reference_at_freezing(model):
    receiver, sea = thermal_case(model)
    phase = ThermalPhase(receiver.ocean_ice, sea)
    source = sea
    heat = torch.full_like(sea.ice.snow_mass[:, :1], -150.).requires_grad_()
    end, _ = phase.apply(source, 600, heat_flux=heat)
    grad, = torch.autograd.grad(phase.H.mass(end).sum(), heat)
    with torch.no_grad():
        plus, _ = phase.apply(source, 600, heat_flux=heat + 0.05)
        minus, _ = phase.apply(source, 600, heat_flux=heat - 0.05)
    fd = (phase.H.mass(plus) - phase.H.mass(minus)) / 0.1
    torch.testing.assert_close(grad, fd, rtol=2e-8, atol=1e-12)
