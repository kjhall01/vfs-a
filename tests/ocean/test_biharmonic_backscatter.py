"""Independent full-H work, spectral selectivity, coasts and empty reservoirs."""

from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from test_gmredi import setup
from test_jmd95_hamiltonian import mesh
from backscatter import EddyBackscatter


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("nu2", [0.0, 1e12])
def test_full_hamiltonian_and_positive_deposit(mesh, coast, nu2):
    m, initial = setup(mesh, coast)
    s = initial.ocean
    u = s.u_e.clone().requires_grad_()
    e = initial.eddy_h.clone().requires_grad_()
    state = replace(
        initial, ocean=s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h), eddy_h=e
    )
    op = EddyBackscatter(nu2, 86400.0, 0.25, biharmonic_viscosity=1e15)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        du, de, r = op.tendency(m, state)
    gu, ge = torch.autograd.grad(m.total_energy(state), (u, e))
    scale = (gu * du).abs().sum() + (ge * de).abs().sum()
    assert ((gu * du).sum() + (ge * de).sum()).abs() < 512 * torch.finfo(
        u.dtype
    ).eps * scale
    assert r.biharmonic_kinetic_power < 0
    torch.testing.assert_close(
        -r.biharmonic_kinetic_power, r.biharmonic_reservoir_power, rtol=2e-14, atol=0
    )
    assert (
        r.local_energy_redistribution.sum().abs()
        < 512 * torch.finfo(u.dtype).eps * scale
    )
    assert r.local_power_residual.abs().sum() < 512 * torch.finfo(u.dtype).eps * scale
    if nu2 == 0:
        assert (de >= 0).all()
    if coast:
        assert torch.count_nonzero(du[~m.ocean.domain.active_edge]) == 0
        assert torch.count_nonzero(de[~m.ocean.domain.wet_cell]) == 0


def test_empty_reservoir_is_filled_and_sink_is_local(mesh):
    m, initial = setup(mesh, True)
    initial = replace(initial, eddy_h=torch.zeros_like(initial.eddy_h))
    op = EddyBackscatter(1e12, 86400.0, 0.25, biharmonic_viscosity=1e15)
    du, de, r = op.tendency(m, initial)
    assert (de >= 0).all() and de.sum() > 0
    # A point velocity perturbation can propagate at most two curl/adjoint hops.
    s = initial.ocean
    edge = torch.where(m.ocean.domain.active_edge)[0][0]
    u = torch.zeros_like(s.u_e)
    u[edge] = 1
    state = replace(
        initial, ocean=s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h)
    )
    du, _, _ = op.tendency(m, state)
    affected = torch.zeros(mesh.nEdges, dtype=torch.bool)
    affected[edge] = True
    for _ in range(2):
        vertices = affected[mesh.edgesOnVertex].any(-1)
        affected = vertices[mesh.verticesOnEdge].any(-1)
    assert (~affected).any()
    assert torch.count_nonzero(du[~affected]) == 0


def test_quadratic_modal_damping(mesh):
    m, initial = setup(mesh)
    s = initial.ocean
    # Independently diagonalize the first-layer kinetic-metric Laplacian.
    he = mesh.cell_to_edge(s.h)[:, 0]
    hv = he[mesh.edgesOnVertex].mean(1)
    metric = (mesh.dc_edge * mesh.dv_edge) * he
    root = metric.sqrt()
    identity = torch.eye(mesh.nEdges, dtype=s.u_e.dtype)
    L = mesh.curl_adjoint(hv[:, None] * mesh.curl(identity)) / he[:, None]
    eig, modes = torch.linalg.eigh(root[:, None] * L / root[None, :])
    op = EddyBackscatter(0.0, 86400.0, 0.25, biharmonic_viscosity=1e15)
    for index in [-1, -10]:
        u = torch.zeros_like(s.u_e)
        u[:, 0] = modes[:, index] / root
        state = replace(
            initial, ocean=s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h)
        )
        du, _, _ = op.tendency(m, state)
        torch.testing.assert_close(
            du, -op.biharmonic_viscosity * eig[index] ** 2 * u, rtol=2e-12, atol=1e-25
        )


def test_sink_backward_and_coupled_stages(mesh):
    m, state = setup(mesh, True)
    m.backscatter = EddyBackscatter(1e5, 86400.0, 0.25, biharmonic_viscosity=1e15)
    s = state.ocean

    def objective(a):
        x = replace(
            state, ocean=s.with_tensors(a * s.u_e, s.h, s.temp_h, s.salt_h, s.tracer_h)
        )
        du, de, _ = m.backscatter.tendency(m, x)
        return du.square().sum() + de.square().sum()

    a = torch.tensor(1.0, requires_grad=True)
    grad = torch.autograd.grad(objective(a), a)[0]
    fd = (objective(a.detach() + 1e-5) - objective(a.detach() - 1e-5)) / 2e-5
    torch.testing.assert_close(grad, fd, rtol=2e-6, atol=1e-20)
    end, r = m.step_with_report(state, 10.0)
    assert len(r.backscatter) == 3 and (end.eddy_h >= 0).all()
    assert all(x.biharmonic_kinetic_power < 0 for x in r.backscatter)


@pytest.mark.parametrize("nu4", [-1.0, float("nan"), float("inf")])
def test_bad_coefficient(nu4):
    with pytest.raises(ValueError):
        EddyBackscatter(1.0, 86400.0, 0.25, biharmonic_viscosity=nu4)
