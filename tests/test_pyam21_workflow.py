"""Optional installed-PyAM21 workflow: invariants, actual physics and restarts."""

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

pytest.importorskip("pyam21")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflows.pyam21.adapter import AM2Adapter
from workflows.pyam21.state import copy_tree
from workflows.pyam21.dynamics import advance_dynamics, pressure_velocity, step
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)
from vfs.atmosphere.moist_cp import interface_pressure_mass
from pyam21.constants import GRAV, CP_AIR
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig


@pytest.fixture(scope="module")
def model():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a, b = make_hybrid_coefficients(12, p_top=1000)
    result = MoistAtmosphere(
        mesh, CharneyPhillipsColumn(12, a, b, phi_full_mode="direct_u_gl2_adjoint")
    )
    result.dynamics.kernel.gravity = GRAV
    return result


def nml():
    return dict(
        moist_processes_nml=dict(
            do_lsc=False, do_mca=False, do_ras=True, do_strat=True, do_cmt=True
        ),
        vert_turb_driver_nml=dict(
            do_mellor_yamada=False,
            do_entrain=True,
            do_stable_bl=True,
            use_tau=False,
            gust_scheme="beljaars",
        ),
        ocean_albedo_nml=dict(ocean_albedo_option=3),
        damping_driver_nml=dict(do_conserve_energy=True),
    )


def initial(model, *, radiation=None, cloudy=False):
    adapter = AM2Adapter(
        model,
        nml(),
        radiation=radiation,
        radiation_id=(
            None if radiation is None else "synthetic-prescribed-radiation-test-v1"
        ),
    )
    p = model.interface_pressure(100000)
    T = 205 + 95 * (p / 100000) ** 0.7
    q = 2e-6 + 0.013 * (p / 100000) ** 4
    liq = torch.zeros_like(q)
    if cloudy:
        liq[:, -3] = 2e-4
    s = model.state_from_fields(
        temperature=T, surface_pressure=100000, normal_wind=0, vapor=q, liquid=liq
    )
    return adapter, adapter.initialize(s, sst=torch.full((model.mesh.nCells,), 301.0))


def empty(columns):
    z = torch.zeros_like(columns["T"])
    r = torch.zeros_like(columns["tracers"])
    return dict(
        tdt=z,
        udt=z,
        vdt=z,
        rdt=r,
        flux_t=z[:, 0],
        flux_q=z[:, 0],
        flux_u=z[:, 0],
        flux_v=z[:, 0],
        lprec=z[:, 0],
        fprec=z[:, 0],
        radiation=None,
    )


def apply_output(adapter, state, out, dt=60):
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    return adapter.apply_result(
        state,
        dt,
        c,
        g,
        out,
        copy_tree(state.surface),
        copy_tree(state.memory),
        copy_tree(state.radiation),
    )


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_tree(a[k], b[k])
    else:
        assert a == b


def test_readout_preserves_water_and_full_enthalpy(model):
    adapter, state = initial(model, cloudy=True)
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    s = state.atmosphere
    dp = g["phalf"].diff()
    torch.testing.assert_close(
        (dp[..., None] * c["tracers"][..., :3]).sum(1),
        s.water_content.sum(1),
        rtol=2e-15,
        atol=1e-12,
    )
    q = c["tracers"][..., :3]
    mixture = torch.cat((1 - q.sum(-1, keepdim=True), q), -1)
    readout = (dp * model.dynamics.thermodynamics.enthalpy(c["T"], mixture)).sum(
        -1
    ) / GRAV
    torch.testing.assert_close(
        readout, adapter.mapping.energy(s), rtol=3e-15, atol=1e-6
    )
    torch.testing.assert_close(
        (model.mesh.cell_area * adapter.mapping.energy(s)).sum(),
        model.diagnostics(s)["energy"],
        rtol=32 * torch.finfo(torch.float64).eps,
        atol=0,
    )
    assert (g["zhalf"].diff() < 0).all() and (g["zfull"] > g["zhalf"][:, 1:]).all()


def test_wind_transpose_work_identity_and_independent_derivative(model):
    adapter, state = initial(model)
    generator = torch.Generator().manual_seed(28)
    u = torch.randn(state.atmosphere.u_e.shape, generator=generator, requires_grad=True)
    force = torch.randn(
        model.mesh.nCells, model.column.n_levels, 3, generator=generator
    )
    power = (adapter.mapping.vector(u) * force).sum()
    transpose = adapter.mapping.transpose(force)
    torch.testing.assert_close(power, (u * transpose).sum(), rtol=1e-14, atol=1e-13)
    torch.testing.assert_close(
        torch.autograd.grad(power, u)[0], transpose, rtol=2e-14, atol=1e-14
    )


def test_no_physics_does_not_smooth_native_state(model):
    adapter, state = initial(model, cloudy=True)
    c, _ = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    end, report = apply_output(adapter, state, empty(c))
    for k in ("u_e", "ps", "water_content", "entropy_content"):
        same_tree(getattr(end.atmosphere, k), getattr(state.atmosphere, k))
    same_tree(end.cloud_fraction, state.cloud_fraction)
    assert report.energy_changes["vfs_full_h"].abs().max() == 0


@pytest.mark.parametrize(
    "kind", ["vapor_source", "liquid_sink", "phase_change", "wind"]
)
def test_finite_sources_preserve_amounts_dry_air_and_native_momentum(model, kind):
    adapter, state = initial(model, cloudy=True)
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out = empty(c)
    out["rdt"] = out["rdt"].clone()
    dt = 60
    if kind == "vapor_source":
        out["rdt"][:, -1, 0] = 1e-7
    elif kind == "liquid_sink":
        out["rdt"][..., 1] = -0.9 * c["tracers"][..., 1] / dt
    elif kind == "phase_change":
        out["rdt"][..., 1] = -0.9 * c["tracers"][..., 1] / dt
        out["rdt"][..., 0] = -out["rdt"][..., 1]
    else:
        out["udt"] = c["T"] * 0 + 1e-4
    out["flux_q"] = (g["phalf"].diff()[..., None] / GRAV * out["rdt"][..., :3]).sum(
        (1, 2)
    )
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = apply_output(adapter, state, out, dt)
    assert end.atmosphere.water_content.min() >= 0
    assert report.water_mapping_residual.abs().max() < 1e-12
    assert report.dry_air_residual.abs().max() < 3e-12
    assert report.water_flux_residual.abs().max() < 1e-12
    impulse_scale = report.edge_impulse.abs() + report.carrier_impulse.abs()
    assert (
        report.momentum_residual.abs()
        <= 8 * torch.finfo(torch.float64).eps * impulse_scale.clamp_min(1)
    ).all()
    before = model.diagnostics(state.atmosphere)
    after = model.diagnostics(end.atmosphere)
    error = (
        after["energy"]
        - before["energy"]
        - (model.mesh.cell_area * report.energy_changes["vfs_full_h"]).sum()
    )
    assert error.abs() < 64 * torch.finfo(torch.float64).eps * (
        before["energy"].abs() + after["energy"].abs()
    )


def test_actual_staged_call_matches_upstream_forward(model):
    adapter, state = initial(model)
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    z = torch.zeros_like(c["T"])
    reference = adapter.physics(
        60,
        adapter.start,
        c,
        g,
        dict(udt=z, vdt=z, tdt=z, rdt=torch.zeros_like(c["tracers"])),
        copy_tree(state.surface),
        copy_tree(state.memory),
    )
    out, _, _, _ = adapter.column_call(state, 60, c, g)
    for k in ("tdt", "rdt", "udt", "vdt", "flux_t", "flux_q", "lprec", "fprec"):
        same_tree(out[k], reference[k])


def test_actual_water_budget_and_input_is_not_mutated(model):
    adapter, state = initial(model)
    original = adapter.pack(state)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = adapter.apply(state, 60)
    same_tree(adapter.pack(state), original)
    assert end.time_seconds == 60
    assert report.water_flux_residual.abs().max() < 1e-12
    assert report.dry_air_residual.abs().max() < 3e-12
    assert report.energy_changes["thermodynamic_convention_difference"].abs().max() > 0
    assert report.fluxes.keys() >= {
        "flux_u",
        "flux_v",
        "sw_net_surface",
        "lw_net_surface",
        "toa_net_down",
    }


def test_cloud_roundoff_is_retained_without_clipping(model):
    adapter, state = initial(model)
    c, _ = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out = empty(c)
    out["rdt"][..., 3] = (1 + 16 * torch.finfo(torch.float64).eps) / 60
    end, _ = apply_output(adapter, state, out)
    assert end.cloud_fraction.max() > 1
    out["rdt"][..., 3] = 1.001 / 60
    with pytest.raises(ValueError, match="inadmissible"):
        apply_output(adapter, state, out)


@pytest.mark.parametrize("dt", [0, -1, 0.5, float("nan"), float("inf"), True])
def test_bad_physics_interval_rejects(model, dt):
    adapter, state = initial(model)
    with pytest.raises(ValueError, match="integer"):
        adapter.apply(state, dt)


def test_restart_reproduces_next_actual_step_and_checks_configuration(model, tmp_path):
    adapter, state = initial(model)
    state, _ = adapter.apply(state, 60)
    path = tmp_path / "restart.pt"
    torch.save(adapter.pack(state), path)
    restarted = adapter.unpack(torch.load(path, weights_only=True))
    a, ra = adapter.apply(state, 60)
    b, rb = adapter.apply(restarted, 60)
    same_tree(adapter.pack(a), adapter.pack(b))
    same_tree(ra.energy_changes, rb.energy_changes)
    payload = adapter.pack(state)
    payload["fingerprint"] = "different-configuration"
    with pytest.raises(ValueError, match="match"):
        adapter.unpack(payload)


class PrescribedRadiationPackage:
    """Synthetic radiation only for testing the real scheduler/cache boundary."""

    def __call__(self, atmos, surface, astro, time, do_lw, do_sw, extra):
        p = atmos["phalf"]
        c, interfaces = p.shape
        fraction = torch.linspace(0, 1, interfaces, dtype=p.dtype)[None].expand(c, -1)
        sw = 300 - 200 * fraction
        lw = 240 - 190 * fraction
        factor = GRAV / CP_AIR * 86400 / p.diff()
        return dict(
            hsw=-sw.diff() * factor,
            heatra=lw.diff() * factor,
            dfsw=sw,
            ufsw=torch.zeros_like(sw),
            flxnet=lw,
        )


def test_radiation_clock_fluxes_restart_and_failure_rollback(model):
    driver = RadiationDriver(
        RadiationDriverConfig(rad_time_step=120), PrescribedRadiationPackage()
    )
    adapter, state = initial(model, radiation=driver)
    a, report = adapter.apply(state, 60)
    assert a.radiation["saved"] is not None and driver.saved is None
    assert a.radiation["lw_alarm"] == 61
    torch.testing.assert_close(
        report.fluxes["toa_net_down"], torch.full((model.mesh.nCells,), 60.0)
    )
    torch.testing.assert_close(
        report.fluxes["lw_net_surface"], torch.full((model.mesh.nCells,), -50.0)
    )
    restarted = adapter.unpack(adapter.pack(a))
    b, _ = adapter.apply(a, 60)
    br, _ = adapter.apply(restarted, 60)
    same_tree(adapter.pack(b), adapter.pack(br))
    assert b.radiation["lw_alarm"] == 1
    before = adapter.pack(b)
    with patch.object(
        adapter, "apply_result", side_effect=ValueError("deliberate rejection")
    ):
        with pytest.raises(ValueError, match="deliberate"):
            adapter.apply(b, 60)
    same_tree(before, adapter.pack(b))
    assert driver.saved is None and driver.lw_alarm == 1


def test_cloud_dynamics_retains_core_bitwise_and_conserves_passive_amount(model):
    adapter, state = initial(model)
    cloud = (
        torch.sigmoid(model.mesh.cell_xyz[:, 0, None] * 5)
        .expand_as(state.cloud_fraction)
        .clone()
    )
    wind = torch.full_like(state.atmosphere.u_e, 1.0)
    state = replace(
        state, cloud_fraction=cloud, atmosphere=replace(state.atmosphere, u_e=wind)
    )
    reference = model.step(state.atmosphere, 1)
    end = advance_dynamics(adapter, state, 1)
    for k in ("u_e", "ps", "water_content", "entropy_content"):
        same_tree(getattr(reference, k), getattr(end.atmosphere, k))

    def amount(x):
        mass = interface_pressure_mass(
            model.column.interface_pressure(x.atmosphere.ps).diff()
        )
        return (model.mesh.cell_area[:, None] * mass * x.cloud_fraction).sum()

    torch.testing.assert_close(amount(state), amount(end), rtol=2e-15, atol=0)
    assert end.cloud_fraction.min() >= 0 and end.cloud_fraction.max() <= 1
    same_tree(end.memory, state.memory)
    assert end.time_seconds == 0


def test_split_step_calls_physics_once_and_rest_omega_is_zero(model):
    adapter, state = initial(model)
    assert pressure_velocity(adapter, state).abs().max() == 0
    with patch.object(adapter, "column_call", wraps=adapter.column_call) as call:
        end, _ = step(adapter, state, 10, dynamics_dt=2)
    assert call.call_count == 1 and end.time_seconds == 10


def test_surface_temperature_backward_against_finite_difference(model):
    adapter, state = initial(model)

    def diagnostic(delta):
        surface = dict(state.surface, t_surf=state.surface["t_surf"] + delta)
        result, _ = adapter.apply(replace(state, surface=surface), 60)
        return model.physical_fields(result.atmosphere)["temperature"].sum()

    x = torch.tensor(0.0, requires_grad=True)
    derivative = torch.autograd.grad(diagnostic(x), x)[0]
    finite = (diagnostic(0.01) - diagnostic(-0.01)) / 0.02
    torch.testing.assert_close(derivative, finite, rtol=2e-4, atol=1e-7)
