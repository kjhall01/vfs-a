"""Independent sensible/latent budgets and conservative cold-content transport."""

from dataclasses import replace
import pytest
import torch

pytest.importorskip("pyam21")
from test_pyam21_workflow import model, initial, PrescribedRadiationPackage
from test_shared_exchange import surface_case, total_water
from workflows.coupled.ice_temperature import (
    ThermalSnowOceanIce,
    ThermalIceReceiver,
    ThermalPhase,
)
from workflows.coupled.driver import exchange_step
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig


def thermal_case(model, cover=True, temperature=260.0):
    receiver, bare = surface_case(model, cover)
    s = bare.coupled.ocean.ocean
    physical = s.with_tensors(s.u_e, s.h, -1.836 * s.h, s.salt_h, s.tracer_h)
    bare = replace(
        bare,
        coupled=replace(
            bare.coupled, ocean=replace(bare.coupled.ocean, ocean=physical)
        ),
    )
    owner = ThermalSnowOceanIce(receiver.snow.coupled)
    return ThermalIceReceiver(owner), owner.initialize(bare, temperature)


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
    E0 = model.diagnostics(air.atmosphere)["energy"] + receiver.snow.total_energy(sea)
    W0 = total_water(adapter, air, receiver, sea)
    out, end, x, ar, sr = exchange_step(adapter, air, receiver, sea, 60)
    E1 = model.diagnostics(out.atmosphere)["energy"] + receiver.snow.total_energy(end)
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
        assert not torch.equal(end.cold_content, sea.cold_content)
        assert (receiver.snow.temperature(end) <= 273.15).all()
    else:
        assert torch.count_nonzero(end.cold_content) == 0
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
    owner = receiver.snow
    dt = 3600
    q = torch.zeros_like(sea.cold_content)
    end, conducted, excess = owner.surface_heat(sea, sea, q, dt)
    p = owner.hamiltonian.slab.parameters
    C = owner.capacity(sea)
    hi, hs = owner.depths(sea)
    K = sea.coupled.ice.area / (
        hi / p.conductivity + hs / owner.parameters.conductivity
    )
    base = p.freezing_temperature(sea.coupled.ocean.ocean.salinity()[:, :1])
    expected = (C * (-13.15) + dt * K * base) / (C + dt * K)
    torch.testing.assert_close(
        owner.temperature(end) - 273.15, expected, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        end.cold_content - sea.cold_content, -conducted - excess, rtol=1e-12, atol=1e-8
    )
    snowy = replace(sea, snow_mass=0.2 * p.density * sea.coupled.ice.volume)
    insulated, _, _ = owner.surface_heat(snowy, snowy, q, dt)
    assert (owner.temperature(insulated) < owner.temperature(end)).all()


def test_surface_melting_complementarity(model):
    receiver, sea = thermal_case(model)
    q = -sea.cold_content + 1e7
    end, conducted, excess = receiver.snow.surface_heat(sea, sea, q, 60)
    assert (end.cold_content == 0).all() and (excess > 0).all()
    torch.testing.assert_close(
        end.cold_content - sea.cold_content,
        q - conducted - excess,
        rtol=1e-12,
        atol=1e-8,
    )


@pytest.mark.parametrize("heat", [-1000.0, 0.0, 2e6])
def test_phase_root_includes_cold_content(model, heat):
    receiver, sea = thermal_case(model)
    owner = receiver.snow
    A = model.mesh.cell_area
    before = owner.total_energy(sea)
    end, q, _ = owner.phase_step(sea, 60, torch.full_like(sea.snow_mass[:, :1], heat))
    assert abs(float((owner.total_energy(end) - before - q) / A.sum())) < 3e-6
    if heat > 1e6:
        assert (end.coupled.ice.volume == 0).all() and (end.cold_content == 0).all()


@pytest.mark.parametrize("active", ["none", "mixed", "all"])
def test_packed_bisection_matches_full_column_brackets(model, active):
    receiver, sea = thermal_case(model)
    rows = torch.arange(model.mesh.nCells)[:, None]
    # Alternate open water and existing ice. The mixed case includes both
    # new ice and snow-covered ice, with unevenly spaced active columns.
    old = sea.coupled.ice
    covered = rows % 2 == 0
    ice = replace(old, area=old.area * covered, volume=old.volume * covered)
    rho = receiver.snow.hamiltonian.slab.parameters.density
    sea = replace(sea, coupled=replace(sea.coupled, ice=ice),
                  cold_content=sea.cold_content * covered, snow_mass=0.1 * rho * ice.volume)
    phase = ThermalPhase(receiver.snow, sea)
    initial = sea.coupled
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
    phase = ThermalPhase(receiver.snow, sea)
    m = (phase.H.mass(sea.coupled) * 0.8).requires_grad_()
    value = phase.root_energy(sea.coupled, m)
    independent = torch.autograd.grad(value.sum(), m)[0]
    analytic = phase.derivative(sea.coupled, phase.phase_state(sea.coupled, m), m)
    torch.testing.assert_close(independent, analytic, rtol=1e-12, atol=1e-7)


def test_transport_and_ridging_preserve_cold_energy(model):
    receiver, sea = thermal_case(model)
    owner = receiver.snow
    A = model.mesh.cell_area[:, None]
    end, _ = owner.dynamics_step(sea, 1.0)
    assert (
        abs(float((A * (end.cold_content - sea.cold_content)).sum() / A.sum())) < 1e-8
    )
    assert (end.cold_content <= 0).all()
    # Uniform specific enthalpy must stay uniform under carrier transport/ridging.
    torch.testing.assert_close(
        owner.temperature(end), owner.temperature(sea), rtol=1e-13, atol=1e-12
    )


def test_complete_thermal_step(model):
    receiver, sea = thermal_case(model)
    initial = receiver.snow.total_energy(sea)
    end, r = receiver.snow.step(sea, 1.0)
    assert (
        abs(
            float(
                (receiver.snow.total_energy(end) - initial - r.external_energy)
                / model.mesh.cell_area.sum()
            )
        )
        < 1e-3
    )


def test_bad_temperature_rejects(model):
    receiver, sea = thermal_case(model)
    with pytest.raises(ValueError):
        receiver.snow.initialize(sea, 274.0)
    with pytest.raises(ValueError):
        receiver.tiles(sea, torch.full_like(sea.snow_mass, 270.0))
