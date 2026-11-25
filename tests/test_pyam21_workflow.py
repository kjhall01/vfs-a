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
from workflows.pyam21.energy import (
    WaterBoundary,
    exchanges,
    layer_energy,
    transfer_energy,
)
from workflows.pyam21.mapping import average, species_increment, species_endpoint
from workflows.pyam21.vapor import VAPOR_EPSILON, restore_positive_vapor
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
        tdt=z.clone(),
        udt=z.clone(),
        vdt=z.clone(),
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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("liquid,ice", [(1e-4, 0), (0, 1e-4), (3e-5, 7e-5)])
def test_zero_vapor_is_funded_by_local_condensate(dtype, liquid, ice):
    from pyam21.constants import HLV, HLF
    T = torch.tensor([[180., 280.]], dtype=dtype)
    q = torch.tensor([[[0., liquid, ice, .5], [1e-30, 0., 0., 0.]]], dtype=dtype)
    out = dict(tdt=torch.zeros_like(T), rdt=torch.zeros_like(q))
    before = copy_tree(dict(T=T, q=q, out=out))
    new_T, new_q, corrected, ledger = restore_positive_vapor(T, q, out, 1200)
    same_tree(dict(T=T, q=q, out=out), before)
    assert new_q[0, 0, 0] > 0
    assert torch.equal(new_q[0, 1], q[0, 1])  # even tiny positive vapor is untouched
    assert torch.equal(new_T[:, 1], T[:, 1])
    assert (new_q[..., 1:3] >= 0).all()
    transferred = q[0, 0, 1:3] - new_q[0, 0, 1:3]
    assert new_q[0, 0, 0] == transferred.sum()
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(new_q[..., :3].sum(-1), q[..., :3].sum(-1),
                               rtol=4 * eps, atol=0)
    # Independent AM2 latent-energy check using the ACTUAL stored fields.
    d = new_q - q
    residual = CP_AIR * (new_T - T) + HLV * d[..., 0] - HLF * d[..., 2]
    assert (residual.abs() <= 2 * eps * CP_AIR * T).all()
    torch.testing.assert_close(ledger["am2_energy_residual"], residual[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(T + 1200 * corrected["tdt"], new_T, rtol=2 * eps, atol=0)
    torch.testing.assert_close(q + 1200 * corrected["rdt"], new_q, rtol=4 * eps, atol=0)
    assert corrected["endpoint_temperature"] is new_T
    assert corrected["endpoint_tracers"] is new_q


def test_positive_vapor_is_an_exact_noop_and_unfunded_zero_rejects():
    T = torch.full((1, 2), 250., dtype=torch.float64)
    q = torch.zeros(1, 2, 4, dtype=T.dtype)
    q[..., 0] = 1e-30
    out = dict(tdt=torch.zeros_like(T), rdt=torch.zeros_like(q))
    new_T, new_q, corrected, report = restore_positive_vapor(T, q, out, 60)
    assert new_T is T and new_q is q and corrected is out and report is None
    q[0, 0, 0] = 0
    with pytest.raises(ValueError, match="lacks local condensate; cell=0 level=0"):
        restore_positive_vapor(T, q, out, 60)


@pytest.mark.parametrize("undeclared_heat", [0., 2.])
def test_zero_vapor_correction_preserves_original_native_energy_target(model, undeclared_heat):
    from pyam21.constants import HLS
    adapter, state = initial(model)
    c, geom = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out, dt = empty(c), 1200
    T, q = c["T"].clone(), c["tracers"].clone()
    # Closed vapor -> ice deposition exhausts exactly one top layer.
    removed = q[0, 0, 0].clone()
    q[0, 0, 0] = 0
    q[0, 0, 2] += removed
    T[0, 0] += HLS * removed / CP_AIR + undeclared_heat
    out.update(endpoint_temperature=T, endpoint_tracers=q,
               tdt=(T - c["T"]) / dt, rdt=(q - c["tracers"]) / dt)
    original = copy_tree(out)
    budget = exchanges(adapter, state, dt, c, geom, out)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = apply_output(adapter, state, out, dt)
    same_tree(out, original)
    correction = report.vapor_correction
    assert correction["indices"].tolist() == [[0, 0]]
    assert abs(float(correction["vapor_added"][0]) / VAPOR_EPSILON - 1) < 1e-8
    assert correction["liquid_removed"][0] == 0
    assert correction["ice_removed"][0] == correction["vapor_added"][0]
    assert correction["temperature_change"][0] < 0
    assert (model.physical_fields(end.atmosphere)["vapor"] > 0).all()
    assert report.water_flux_residual.abs().max() < 1e-12
    assert report.dry_air_residual.abs().max() < 1e-10
    assert report.water_mapping_residual.abs().max() < 1e-12
    torch.testing.assert_close(report.energy_transfer["target_layer_increment"], budget["target"],
                               rtol=0, atol=0)
    torch.testing.assert_close(report.energy_changes["am2_energy_residual"], budget["am2_residual"],
                               rtol=0, atol=0)
    energy_close(adapter, state, end, report)
    if undeclared_heat:
        assert report.energy_changes["am2_energy_residual"][0] > 100
    corrected = report.column_result
    eps = torch.finfo(T.dtype).eps
    torch.testing.assert_close(c["T"] + dt * corrected["tdt"], corrected["endpoint_temperature"],
                               rtol=128 * eps, atol=0)
    torch.testing.assert_close(c["tracers"] + dt * corrected["rdt"], corrected["endpoint_tracers"],
                               rtol=0, atol=128 * eps * q[..., :3].abs().max())
    # New corrections remain ordinary explicit state across pack/unpack.
    replay = adapter.unpack(adapter.pack(end))
    same_tree(adapter.pack(end), adapter.pack(replay))
    # An actual negative endpoint or a mismatched report still rejects first.
    out["endpoint_tracers"][0, 0, 0] = -1e-8
    out["rdt"][0, 0, 0] = (-1e-8 - c["tracers"][0, 0, 0]) / dt
    with pytest.raises(ValueError, match="inadmissible finite column update"):
        apply_output(adapter, state, out, dt)
    out["rdt"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="endpoint and tendencies disagree"):
        apply_output(adapter, state, out, dt)


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
    from pyam21.physics.moist import MoistProcesses

    moist = adapter.physics.moist
    with patch.object(adapter.physics, "moist", MoistProcesses(moist.c, moist.ras, moist.strat)):
        reference = adapter.physics(
            60, adapter.start, c, g,
            dict(udt=z, vdt=z, tdt=z, rdt=torch.zeros_like(c["tracers"])),
            copy_tree(state.surface), copy_tree(state.memory),
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


def test_exact_calm_zero_gust_identifies_upstream_surface_failure(model):
    adapter, state = initial(model)
    state = replace(state, memory={**state.memory, "gust": torch.zeros_like(state.memory["gust"])})
    before = adapter.pack(state)
    # The pinned upstream surface routine divides by wind/gust magnitude.
    # Report that upstream degeneracy at its origin, without inventing a floor.
    with pytest.raises(FloatingPointError, match="PyAM21 surface layer returned nonfinite"):
        adapter.apply(state, 1200)
    same_tree(adapter.pack(state), before)


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
    energy_close(adapter, state, a, report)
    torch.testing.assert_close(
        report.energy_changes["boundary_radiation"],
        torch.full((model.mesh.nCells,), 600.0),
    )
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


def test_cloud_dynamics_reuses_exact_moist_carrier_without_second_graph(model):
    """Same state, same RK arithmetic, no extra graph or transport primitive."""
    from unittest.mock import patch
    import workflows.pyam21.dynamics as dynamics_module
    from vfs.atmosphere.bounded_transport import graph as original_graph
    adapter, state = initial(model)
    # Existing moist Euler calls graph internally; the cloud ledger must not
    # independently call dynamics_module.graph after the authoritative stage.
    with patch.object(dynamics_module, "graph",
                      side_effect=AssertionError("cloud recomputed graph")):
        optimized = dynamics_module.advance_dynamics(adapter, state, 1)

    def baseline_cloud_step(current):
        area = model.mesh.cell_area[:, None]
        mass = lambda s: area * interface_pressure_mass(
            s.column.interface_pressure(s.ps).diff()
        )
        start = current.atmosphere
        initial_cloud_amount = mass(start) * current.cloud_fraction
        def old_euler(s, amount):
            end, _ = model.transport.euler_with_info(s, 1)
            m, i, j, F = original_graph(model.dynamics, s)
            q = (amount / m).flatten()
            fractions = torch.stack((1 - q, q), -1)
            face = 0.5 * (fractions[i] + fractions[j])
            from vfs.atmosphere.bounded_transport import bounded_transfer
            amounts, _, _ = bounded_transfer(
                m.flatten(), m.flatten()[:, None] * fractions,
                i, j, F, F[:, None] * face,
                target_mass=mass(end).flatten(),
            )
            return end, amounts[:, 1].reshape_as(m)
        s1, c1 = old_euler(start, initial_cloud_amount)
        s1e, c1e = old_euler(s1, c1)
        s2 = start.combine(s1e, a=0.75, b=0.25)
        c2 = 0.75 * initial_cloud_amount + 0.25 * c1e
        s2e, c2e = old_euler(s2, c2)
        end = start.combine(s2e, a=1 / 3, b=2 / 3)
        cloud = ((1 / 3) * initial_cloud_amount + (2 / 3) * c2e) / mass(end)
        return end, cloud

    old_air, old_cloud = baseline_cloud_step(state)
    for name in ("u_e", "ps", "water_content", "entropy_content"):
        assert torch.equal(getattr(optimized.atmosphere, name), getattr(old_air, name))
    assert torch.equal(optimized.cloud_fraction, old_cloud)


def test_bounded_euler_reports_authoritative_carrier_graph(model):
    adapter, state = initial(model)
    rhs, info = model.transport.evaluate(
        state.atmosphere, 1, include_carrier_graph=True
    )
    mass, i, j, flux = info["carrier_graph"]
    # Ordinary public report consumers retain the original schema.
    _, default_info = model.transport.evaluate(state.atmosphere, 1)
    assert "carrier_graph" not in default_info
    assert flux is info["carrier"]
    assert mass.shape == state.atmosphere.entropy_content.shape
    assert i.ndim == j.ndim == flux.ndim == 1
    assert i.shape == j.shape == flux.shape
    assert torch.all(torch.isfinite(flux))


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


def energy_close(adapter, state, end, report):
    """Independently recompute native H, without trusting reported H changes."""
    old = adapter.model.diagnostics(state.atmosphere)["energy"]
    new = adapter.model.diagnostics(end.atmosphere)["energy"]
    expected = (
        adapter.model.mesh.cell_area
        * (
            report.energy_changes["external_energy"]
            + report.energy_changes["am2_energy_residual"]
        )
    ).sum()
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    q = c["tracers"][..., :3]
    mixture = torch.cat((1 - q.sum(-1, keepdim=True), q), -1)
    thermal_scale = (
        g["phalf"].diff()
        / GRAV
        * adapter.model.dynamics.thermodynamics.heat_capacity_p(mixture)
        * c["T"]
    )
    torch.testing.assert_close(
        new - old,
        expected,
        rtol=0,
        # Net H may nearly cancel between the thermal datum and topography.
        # Bound floating-point evaluation by absolute thermal energy instead.
        atol=float(
            512
            * torch.finfo(torch.float64).eps
            * (adapter.model.mesh.cell_area[:, None] * thermal_scale).sum()
        ),
    )
    actual_layers = layer_energy(adapter, end.atmosphere) - layer_energy(
        adapter, state.atmosphere
    )
    scale = thermal_scale
    error = actual_layers - report.energy_transfer["target_layer_increment"]
    assert (error.abs() < 512 * torch.finfo(torch.float64).eps * scale).all()


@pytest.mark.parametrize(
    "kind", ["heat", "evaporation", "dew", "rain", "snow", "phase", "drag", "defect"]
)
def test_energy_transfer_and_independent_surface_ledger(model, kind):
    adapter, state = initial(model, cloudy=True)
    if kind == "snow":
        water = state.atmosphere.water_content.clone()
        water[..., 2] = water[..., 1]
        water[..., 1] = 0
        state = replace(
            state, atmosphere=replace(state.atmosphere, water_content=water)
        )
    if kind in ("drag", "evaporation", "rain", "snow", "dew"):
        state = replace(
            state,
            atmosphere=replace(
                state.atmosphere, u_e=torch.ones_like(state.atmosphere.u_e) * 5
            ),
        )
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out, dt = empty(c), 60
    mass = g["phalf"].diff() / GRAV
    out["rdt"] = out["rdt"].clone()
    if kind == "heat":
        out["flux_t"] = out["flux_t"] + 20
        out["tdt"][:, -1] = 20 / (CP_AIR * mass[:, -1])
    elif kind in ("evaporation", "dew"):
        out["flux_q"] = out["flux_q"] + (1e-5 if kind == "evaporation" else -1e-5)
        out["rdt"][:, -1, 0] = out["flux_q"] / mass[:, -1]
    elif kind in ("rain", "snow"):
        species = 1 if kind == "rain" else 2
        out["rdt"][..., species] = -0.2 * c["tracers"][..., species] / dt
        out["lprec" if species == 1 else "fprec"] = -(
            mass * out["rdt"][..., species]
        ).sum(-1)
    elif kind == "phase":
        out["rdt"][..., 1] = -0.2 * c["tracers"][..., 1] / dt
        out["rdt"][..., 0] = -out["rdt"][..., 1]
        out["tdt"] = -2.5e6 / CP_AIR * out["rdt"][..., 0]
    elif kind == "drag":
        out["udt"], out["vdt"] = -0.02 * c["u"] / dt, -0.02 * c["v"] / dt
        out["tdt"] = (
            -(
                (c["u"] + 0.5 * dt * out["udt"]) * out["udt"]
                + (c["v"] + 0.5 * dt * out["vdt"]) * out["vdt"]
            )
            / CP_AIR
        )
    else:
        out["tdt"] = out["tdt"] + 1e-5  # deliberately undeclared upstream heat
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = apply_output(adapter, state, out, dt)
    energy_close(adapter, state, end, report)
    e = report.energy_changes
    if kind == "defect":
        assert e["external_energy"].abs().max() == 0
        assert e["am2_energy_residual"].min() > 100
        torch.testing.assert_close(
            e["full_h_minus_external"], e["am2_energy_residual"], atol=2e-6, rtol=0
        )
    else:
        assert e["am2_energy_residual"].abs().max() < 2e-10
    if kind == "evaporation":
        th = model.dynamics.thermodynamics
        expected = (
            dt
            * out["flux_q"]
            * (th.L_v_ref + th.cp_v * (state.surface["t_surf"] - th.T_ref))
        )
        torch.testing.assert_close(
            e["external_energy"], expected, rtol=1e-15, atol=1e-12
        )
        assert report.carrier_impulse.abs().max() == 0  # stationary entering water
    if kind == "drag":
        assert e["external_energy"].abs().max() == 0
        assert e["native_kinetic"].max() < 0


def test_explicit_water_boundary_and_potential_energy(model):
    column = CharneyPhillipsColumn(
        model.column.n_levels,
        model.column.A_interface.clone(),
        model.column.B_interface.clone(),
        phi_s=5000.0,
        phi_full_mode="direct_u_gl2_adjoint",
    )
    raised = MoistAtmosphere(model.mesh, column)
    raised.dynamics.kernel.gravity = GRAV
    adapter, state = initial(raised)
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out, dt = empty(c), 30
    out["flux_q"] = out["flux_q"] + 1e-5
    out["rdt"][:, -1, 0] = out["flux_q"] * GRAV / g["phalf"].diff()[:, -1]
    boundary = WaterBoundary.prescribed_ocean(adapter, state, c, out)
    boundary = replace(
        boundary,
        vapor_temperature=boundary.vapor_temperature + 3,
        vapor_velocity=adapter.mapping.east * 2,
    )
    end, report = adapter.apply_result(
        state,
        dt,
        c,
        g,
        out,
        copy_tree(state.surface),
        copy_tree(state.memory),
        {},
        water_boundary=boundary,
    )
    energy_close(adapter, state, end, report)
    torch.testing.assert_close(
        report.energy_changes["boundary_water_potential"], dt * out["flux_q"] * 5000
    )
    torch.testing.assert_close(
        report.energy_changes["boundary_water_kinetic"], dt * out["flux_q"] * 2
    )
    assert report.carrier_impulse.abs().max() > 0
    with pytest.raises(ValueError, match="tangential"):
        exchanges(
            adapter,
            state,
            dt,
            c,
            g,
            out,
            replace(boundary, vapor_velocity=raised.mesh.cell_xyz),
        )


def test_energy_solve_derivative_at_zero_source_and_rejection(model):
    adapter, state = initial(model)
    s = state.atmosphere
    target = torch.zeros(
        (model.mesh.nCells, model.column.n_levels), dtype=torch.float64
    )
    x = torch.tensor(0.0, requires_grad=True)
    end, _ = transfer_energy(adapter, s, s, target + x)
    derivative = torch.autograd.grad(adapter.mapping.energy(end).sum(), x)[0]
    torch.testing.assert_close(
        derivative, torch.tensor(float(target.numel())), rtol=2e-13, atol=0
    )
    before = adapter.pack(state)
    with pytest.raises(FloatingPointError, match="converge"):
        transfer_energy(adapter, s, s, target - 1e10, max_iterations=0)
    same_tree(adapter.pack(state), before)


def test_nonzero_wind_zero_physics_and_legacy_restart_separation(model):
    adapter, state = initial(model)
    state = replace(
        state,
        atmosphere=replace(
            state.atmosphere, u_e=torch.ones_like(state.atmosphere.u_e) * 3.2
        ),
    )
    c, _ = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    end, _ = apply_output(adapter, state, empty(c))
    for k in ("u_e", "ps", "water_content", "entropy_content"):
        same_tree(getattr(end.atmosphere, k), getattr(state.atmosphere, k))
    with pytest.raises(ValueError, match="must be conservative"):
        AM2Adapter(model, nml(), energy_mode="legacy")


def test_actual_energy_residual_is_upstream_not_mapping(model):
    adapter, state = initial(model)
    for _ in range(5):
        end, report = adapter.apply(state, 60)
        energy_close(adapter, state, end, report)
        torch.testing.assert_close(
            report.energy_changes["full_h_minus_external"],
            report.energy_changes["am2_energy_residual"],
            atol=3e-6,
            rtol=0,
        )
        state = end


def test_actual_disabled_dissipative_heating_is_not_silently_repaired(model):
    adapter, state = initial(model)
    assert not adapter.physics.vdiff.c.do_conserve_energy
    state = replace(
        state,
        atmosphere=replace(
            state.atmosphere, u_e=torch.ones_like(state.atmosphere.u_e) * 10
        ),
    )
    end, report = adapter.apply(state, 60)
    energy_close(adapter, state, end, report)
    assert report.energy_changes["am2_energy_residual"].abs().max() > 1
    torch.testing.assert_close(
        report.energy_changes["full_h_minus_external"],
        report.energy_changes["am2_energy_residual"],
        atol=3e-6,
        rtol=0,
    )


def test_exhausted_condensate_uses_remaining_amount_not_subtractive_cancellation():
    dp = torch.tensor([[8595.237114325872]], dtype=torch.float64)
    water = torch.tensor(
        [[[4.7761293471473453e-05], [9.630083529881845e-05]]], dtype=torch.float64
    )
    q = water / interface_pressure_mass(dp)[..., None]
    full = average(q)
    tendency = -full / 60
    endpoint = full + 60 * tendency
    source = 60 * dp[..., None] * tendency
    assert endpoint.abs().max() == 0
    assert (water + species_increment(q, source)).min() < 0
    result = species_endpoint(water, dp, source, endpoint)
    assert result.abs().max() == 0  # exact prescribed exhaustion, no mass floor

    # Admitted signed CP roundoff may remain at an unforced node adjacent to an
    # exhausted layer. Reconstructing from endpoints removes the null-space
    # tail without inventing or discarding the prescribed layer amount.
    dp = torch.tensor([[2000.0, 3000.0]], dtype=torch.float64)
    q = torch.tensor([[[-1e-30], [1e-30], [1e-8]]], dtype=torch.float64)
    water = interface_pressure_mass(dp)[..., None] * q
    source = -dp[..., None] * average(q)
    result = species_endpoint(water, dp, source, torch.zeros_like(source))
    assert result.abs().max() == 0
    # A truly negative upstream endpoint is not silently clipped away.
    endpoint = torch.full_like(source, -1e-35)
    negative = species_endpoint(water, dp, source, endpoint)
    assert negative.min() < 0
    torch.testing.assert_close(
        negative.sum(1), (dp[..., None] * endpoint).sum(1), rtol=1e-15, atol=0
    )


def test_explicit_upstream_endpoint_avoids_reconstruction_roundoff_and_checks_consistency(
    model,
):
    adapter, state = initial(model, cloudy=True)
    c, g = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out = empty(c)
    out["rdt"][..., 1] = -c["tracers"][..., 1] / 60 - 1e-25
    out["endpoint_temperature"] = c["T"].clone()
    out["endpoint_tracers"] = c["tracers"].clone()
    out["endpoint_tracers"][..., 1] = 0
    out["lprec"] = (g["phalf"].diff() / GRAV * c["tracers"][..., 1]).sum(-1) / 60
    assert (c["tracers"] + 60 * out["rdt"])[..., 1].min() < 0
    end, report = apply_output(adapter, state, out)
    assert end.atmosphere.water_content[..., 1].abs().max() == 0
    energy_close(adapter, state, end, report)
    out["endpoint_tracers"][..., 0] += 0.001
    with pytest.raises(ValueError, match="beyond roundoff"):
        apply_output(adapter, state, out)


def test_cloud_fraction_cancellation_requires_consistent_process_terms(model):
    # Scalar operands from the 4096-cell checkpoint replay, with a 64-eps
    # cloud perturbation at cell 2289. Stratiform physics returns exactly 1.
    adapter, state = initial(model, cloudy=True)
    dt = 1200
    state = replace(state, cloud_fraction=torch.full_like(state.cloud_fraction, 0.1735546588214871))
    col, _ = adapter.mapping.columns(state.atmosphere, state.cloud_fraction)
    out = empty(col)
    out["endpoint_temperature"] = col["T"].clone()
    endpoint = out["endpoint_tracers"] = col["tracers"].clone()
    cloud = col["tracers"][..., 3]
    stages = out["cloud_stages"] = {
        "turbulent_rate": torch.zeros_like(cloud),
        "ras_rate": torch.zeros_like(cloud),
        "ras_end": cloud.clone(),
        "strat_change": torch.zeros_like(cloud),
    }
    cell, level = 3, 2
    stages["turbulent_rate"][cell, level] = -4.087763104273237e-05
    stages["ras_rate"][cell, level] = 0.13367977960071867
    stages["ras_end"][cell, level] = 160.54023702243262
    stages["strat_change"][cell, level] = -159.54023702243262
    out["rdt"][cell, level, 3] = 0.0006887044509820694
    endpoint[cell, level, 3] = 1.0
    reconstructed = cloud[cell, level] + dt * out["rdt"][cell, level, 3]
    assert abs(reconstructed.item() - 1.0) > 128 * torch.finfo(cloud.dtype).eps
    apply_output(adapter, state, out, dt)

    message = "beyond roundoff; field=cloud_fraction cell=3 level=2"
    endpoint[cell, level, 3] -= 1e-12
    with pytest.raises(ValueError, match=message):
        apply_output(adapter, state, out, dt)
    endpoint[cell, level, 3] = 1.0
    out["rdt"][cell, level, 3] += 1e-12 / dt
    with pytest.raises(ValueError, match=message):
        apply_output(adapter, state, out, dt)
    out["rdt"][cell, level, 3] = 0.0006887044509820694
    stages["ras_end"][cell, level] += 1
    stages["strat_change"][cell, level] -= 1
    with pytest.raises(ValueError, match=message):
        apply_output(adapter, state, out, dt)

    # A formally consistent pair can lose the entire initial cloud fraction
    # when its intermediates are too large. The physical cutoff still rejects it.
    stages["ras_rate"][cell, level] = 1e16 * (1.0 / dt)
    stages["ras_end"][cell, level] = 1e16
    stages["strat_change"][cell, level] = -1e16
    endpoint[cell, level, 3] = 0
    out["rdt"][cell, level, 3] = (
        stages["turbulent_rate"][cell, level] + stages["ras_rate"][cell, level]
    ) + stages["strat_change"][cell, level] * (1.0 / dt)
    with pytest.raises(ValueError, match=message):
        apply_output(adapter, state, out, dt)

    out["rdt"][cell, level, 3] = torch.finfo(cloud.dtype).max
    with pytest.raises(ValueError, match="invalid explicit PyAM21 endpoint or tendency reconstruction"):
        apply_output(adapter, state, out, dt)


def test_inadmissible_column_detail_identifies_field_and_location_without_clipping():
    from workflows.pyam21.adapter import _inadmissible_column_detail

    eps = torch.finfo(torch.float64).eps
    t = torch.full((3, 2), 275.0, dtype=torch.float64)
    q = torch.zeros((3, 2, 4), dtype=torch.float64)
    q[..., 0] = 0.005
    q[..., 3] = 0.4
    rates = {
        "tdt": torch.zeros_like(t),
        "rdt": torch.zeros_like(q),
        "udt": torch.zeros_like(t),
        "vdt": torch.zeros_like(t),
    }
    tol = torch.zeros(2, dtype=t.dtype)
    qbad = q.clone()
    qbad[1, 0, 0] = -1e-4
    detail = _inadmissible_column_detail(t, qbad, rates, eps, tol)
    assert "field=vapor_mass_fraction cell=1 level=0" in detail
    assert "value=-0.0001" in detail
    qbad[1, 0, 0] = 0
    qbad[1, 0, 2] = -1e-4
    detail = _inadmissible_column_detail(t, qbad, rates, eps, tol, allow_zero_vapor=True)
    assert "field=ice_mass_fraction cell=1 level=0" in detail
    qbad = q.clone()
    qbad[2, 1, 3] = 1.01
    detail = _inadmissible_column_detail(t, qbad, rates, eps, tol)
    assert "field=cloud_fraction cell=2 level=1" in detail
    tb = t.clone()
    tb[0, 1] = 0
    detail = _inadmissible_column_detail(tb, q, rates, eps, tol)
    assert "field=temperature_K cell=0 level=1" in detail
    assert "no endpoint violation" in _inadmissible_column_detail(t, q, rates, eps, tol)
