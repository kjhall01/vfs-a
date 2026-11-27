"""Opt-in donor-minus-centered lid flux and its energy return."""

import pytest
import torch

from test_model_api import init, model
from vfs import BoundedSSPRK3, MoistAtmosphere
from vfs.atmosphere.bounded_moist import BoundedGibbs


def test_top_donor_flux_matches_both_flow_directions(model):
    s = init(model, normal_wind=model.mesh.cell_area.new_tensor([20., -5., 8., 10.]))
    dyn, mesh = model.dynamics, model.mesh
    base = dyn.kernel._transport_primitives(dyn.transport_state(s))
    mu = base["mu"].clone()
    mu[:, 1] = 0.1  # Pa/s, tested independently of its diagnosed sign in the full RHS
    metric = dyn.kernel.metric_action(s, torch.ones_like(s.u_e), s.column.interface_pressure(s.ps).diff())
    gradient = dyn.energy_gradients(s).entropy_content
    donor = BoundedGibbs(dyn, top_momentum_upwind=True)
    shear = s.u_e[:, 0] - s.u_e[:, 1]
    for sign in (1, -1, 0):
        ctx = dict(base, mu=mu * sign)
        du, ds, heat = donor._top_momentum_correction(s, ctx, metric, gradient)
        F = (mesh.dc_edge * mesh.dv_edge) * (0.1 * sign) / dyn.kernel.gravity
        upwind = s.u_e[:, 0 if sign >= 0 else 1]
        flux = F * (upwind - (s.u_e[:, 0] + s.u_e[:, 1]) / 2)
        torch.testing.assert_close(metric[:, 0] * du[:, 0], -flux)
        torch.testing.assert_close(metric[:, 1] * du[:, 1], flux)
        torch.testing.assert_close(heat, 0.5 * F.abs() * shear.square())
        assert torch.count_nonzero(du[:, 2:]) == 0
        assert torch.count_nonzero(ds[:, [0, 2, 3, 4]]) == 0
        assert (ds >= 0).all()
        torch.testing.assert_close((gradient * ds).sum(), heat.sum())
        torch.testing.assert_close((metric * s.u_e * du).sum(), -heat.sum())


def test_no_momentum_shear_means_no_change(model):
    s = init(model, normal_wind=10.0)
    base, old = model.transport.evaluate(s, 10.)
    donor = BoundedGibbs(model.dynamics, top_momentum_upwind=True)
    altered, new = donor.evaluate(s, 10.)
    assert torch.count_nonzero(new["top_momentum_heat_W"]) == 0
    for k in ("u_e", "entropy_content", "ps", "water_content"):
        torch.testing.assert_close(getattr(altered, k), getattr(base, k), rtol=0, atol=0)
    assert "top_momentum_heat_W" not in old


def test_option_pairs_work_and_heat_in_bounded_stage(model):
    shear = model.mesh.edge_xyz[:, :1] * model.mesh.cell_area.new_tensor([[24., -6., 1., 2.]])
    s = init(model, normal_wind=shear)
    experimental = MoistAtmosphere(model.mesh, model.column, stepper=BoundedSSPRK3(top_momentum_upwind=True))
    old, ledger_old = model.transport.evaluate(s, 10.)
    new, ledger_new = experimental.transport.evaluate(s, 10.)
    heat = ledger_new["top_momentum_heat_W"].sum()
    assert heat > 0
    assert torch.equal(old.ps, new.ps)
    assert torch.equal(old.water_content, new.water_content)
    assert torch.equal(ledger_old["amounts"], ledger_new["amounts"])
    change = new.combine(old, b=-1)
    assert torch.equal(change.u_e[:, 2:], torch.zeros_like(change.u_e[:, 2:]))
    grad = model.dynamics.energy_gradients(s)
    torch.testing.assert_close((grad.u_e * change.u_e).sum(), -heat, rtol=1e-10, atol=0)
    torch.testing.assert_close((grad.entropy_content * change.entropy_content).sum(), heat, rtol=1e-10, atol=0)
    advected = experimental.step(s, 10.)
    centered = model.step(s, 10.)
    assert not torch.equal(advected.u_e, centered.u_e)
    assert torch.isfinite(advected.entropy_content).all()


def test_upwind_flag_rejects_nonboolean(model):
    with pytest.raises(TypeError, match="top_momentum_upwind"):
        MoistAtmosphere(model.mesh, model.column, stepper=BoundedSSPRK3(top_momentum_upwind="false"))
