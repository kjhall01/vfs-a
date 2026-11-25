"""Instantaneous Hamiltonian pairing, local funding and coupled RK stages."""

from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from test_gmredi import setup
from test_jmd95_hamiltonian import mesh
from vfs.ocean.temperature.backscatter import EddyBackscatter


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("nu", [1e5, 1e12])
def test_instantaneous_full_hamiltonian_and_local_power(mesh, coast, nu):
    m, initial = setup(mesh, coast)
    s = initial.ocean
    u = s.u_e.detach().clone().requires_grad_()
    e = initial.eddy_h.detach().clone().requires_grad_()
    state = replace(
        initial, ocean=s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h), eddy_h=e
    )
    op = EddyBackscatter(nu, 86400.0, 0.25)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        du, de, r = op.tendency(m, state)
    gu, ge = torch.autograd.grad(m.total_energy(state), (u, e))
    independent = (gu * du).sum() + (ge * de).sum()
    eps = torch.finfo(u.dtype).eps
    scale = r.local_kinetic_power.abs().sum()
    assert independent.abs() <= 256 * eps * scale
    assert r.local_power_residual.abs().sum() <= 256 * eps * scale
    assert r.kinetic_power > 0
    assert (
        r.local_kinetic_power
        <= r.power_allowance
        + 256 * eps * (r.power_allowance + r.local_kinetic_power.abs())
    ).all()
    if nu == 1e12:
        assert (r.vertex_scale < 1).any()
    if coast:
        assert torch.count_nonzero(du[~m.ocean.domain.active_edge]) == 0
        assert torch.count_nonzero(de[~m.ocean.domain.wet_cell]) == 0


@pytest.mark.parametrize("mode", ["empty", "rest", "disabled", "zero_viscosity"])
def test_zero_cases(mesh, mode):
    m, state = setup(mesh)
    s = state.ocean
    if mode == "empty":
        state = replace(state, eddy_h=torch.zeros_like(state.eddy_h))
    if mode == "rest":
        state = replace(
            state,
            ocean=s.with_tensors(
                torch.zeros_like(s.u_e), s.h, s.temp_h, s.salt_h, s.tracer_h
            ),
        )
    op = EddyBackscatter(
        0 if mode == "zero_viscosity" else 1e12,
        86400.0,
        0 if mode == "disabled" else 0.25,
    )
    du, de, r = op.tendency(m, state)
    assert torch.count_nonzero(du) == 0 and torch.count_nonzero(de) == 0


def test_empty_cell_has_only_local_effect(mesh):
    m, state = setup(mesh)
    op = EddyBackscatter(1e12, 86400.0, 0.25)
    du, de, r = op.tendency(m, state)
    e = state.eddy_h.clone()
    e[0] = 0
    vdu, vde, vr = op.tendency(m, replace(state, eddy_h=e))
    incident = (mesh.cellsOnEdge == 0).any(-1)
    vertices = incident[mesh.edgesOnVertex].any(-1)
    affected = vertices[mesh.verticesOnEdge].any(-1)
    assert (~affected).any()
    assert torch.equal(vdu[~affected], du[~affected])
    assert torch.count_nonzero(vdu[~affected]) > 0
    assert (vr.vertex_scale[vertices] == 0).all()
    assert torch.equal(vr.vertex_scale[~vertices], r.vertex_scale[~vertices])
    assert (vde[0] >= 0).all()


@pytest.mark.parametrize("limited", [False, True])
def test_backward(mesh, limited):
    m, state = setup(mesh)
    s = state.ocean
    op = EddyBackscatter(1e12 if limited else 1e5, 86400.0, 0.25)

    def f(a):
        x = replace(
            state, ocean=s.with_tensors(a * s.u_e, s.h, s.temp_h, s.salt_h, s.tracer_h)
        )
        du, de, _ = op.tendency(m, x)
        return du.square().sum() + de.square().sum()

    a = torch.tensor(1.0, requires_grad=True)
    derivative = torch.autograd.grad(f(a), a)[0]
    delta = 1e-5
    finite = (f(a.detach() + delta) - f(a.detach() - delta)) / (2 * delta)
    torch.testing.assert_close(derivative, finite, rtol=2e-6, atol=1e-20)


def test_coupled_three_stage_sources_and_rejection(mesh):
    m, state = setup(mesh, True)
    m.backscatter = EddyBackscatter(1e12, 86400.0, 0.25)
    with patch.object(
        EddyBackscatter, "tendency", autospec=True, side_effect=EddyBackscatter.tendency
    ) as call:
        end, report = m.step_with_report(state, 10.0)
        assert call.call_count == 3
    assert len(report.backscatter) == 3
    assert (end.eddy_h >= 0).all()
    assert torch.isfinite(m.total_energy(end))
    for r in report.backscatter:
        assert (
            r.local_power_residual.abs().sum()
            < 1e-12 * r.local_kinetic_power.abs().sum()
        )
    with pytest.raises(ValueError, match="positivity"):
        m.step_with_report(state, 1e6)


def test_invalid_parameters_and_state(mesh):
    for args in [
        (-1, 1, 0.5),
        (1, 0, 0.5),
        (1, 1, -1),
        (1, 1, 1.1),
        (float("inf"), 1, 0.5),
    ]:
        with pytest.raises(ValueError):
            EddyBackscatter(*args)
    m, state = setup(mesh)
    op = EddyBackscatter(1e5, 86400.0, 0.25)
    with pytest.raises(ValueError):
        op.tendency(m, replace(state, eddy_h=-state.eddy_h))
    from vfs.ocean.stratified import TKEState

    with pytest.raises(TypeError):
        op.tendency(m, TKEState(state.ocean, state.tke_h))


def test_composition_with_gmredi_and_decay(mesh):
    from vfs.ocean.temperature.eddy import GMRedi
    from vfs.ocean.temperature.eddy_energy import EddyDissipation

    m, state = setup(mesh, True)
    m.backscatter = EddyBackscatter(1e12, 86400.0, 0.25)
    gm = GMRedi(kappa_gm=100.0, kappa_redi=10.0)
    decay = EddyDissipation(864000.0)
    for _ in range(3):
        state, r = m.step_with_report(state, 10.0)
        state, _ = gm.apply(m, state, 10.0)
        state, _ = decay.apply(m, state, 10.0)
        assert (state.eddy_h >= 0).all()
        assert len(r.backscatter) == 3
