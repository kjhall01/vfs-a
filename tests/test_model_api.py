"""Physical initialization, validation, pairing and gradient regression tests."""

import torch
import pytest
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)
from vfs.atmosphere import MoistState
from vfs.atmosphere.moist_cp import interface_pressure_mass


@pytest.fixture
def model():
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(4, p_top=5000.0)
    column = CharneyPhillipsColumn(
        4, A, B, phi_s=100 * mesh.cell_xyz[:, 0], phi_full_mode="direct_u_gl2_adjoint"
    )
    return MoistAtmosphere(mesh, column)


def init(model, **changes):
    args = dict(
        temperature=280.0,
        surface_pressure=100000.0,
        normal_wind=0.0,
        vapor=0.01,
        liquid=0.002,
        ice=0.001,
    )
    args.update(changes)
    return model.state_from_fields(**args)


def test_physical_roundtrip_and_manual_equivalence(model):
    ref = model.mesh.cell_area
    ps = ref.new_full((32,), 100000)
    p = model.interface_pressure(ps)
    T = 280 * (p / 100000) ** 0.08
    s = init(model, temperature=T, surface_pressure=ps)
    dm = interface_pressure_mass(p.diff())
    water = p.new_tensor([0.01, 0.002, 0.001]).expand(32, 5, 3)
    fractions = torch.cat((1 - water.sum(-1, keepdim=True), water), -1)
    manual = MoistState(
        s.u_e,
        ps[:, None],
        dm * model.dynamics.thermodynamics.entropy(T, p, fractions),
        dm[..., None] * water,
        model.mesh,
        model.column,
    )
    for key in ("u_e", "ps", "entropy_content", "water_content"):
        assert torch.equal(getattr(s, key), getattr(manual, key))
    fields = model.physical_fields(s)
    torch.testing.assert_close(fields["temperature"], T, rtol=1e-14, atol=1e-12)
    for name, val in (("vapor", 0.01), ("liquid", 0.002), ("ice", 0.001)):
        torch.testing.assert_close(fields[name], p.new_full(p.shape, val))
    actual = model.step(s, 1.0)
    expected = model.transport.step(manual, 1.0)
    for key in ("u_e", "ps", "entropy_content", "water_content"):
        assert torch.equal(getattr(actual, key), getattr(expected, key))


def test_wind_projection(model):
    xyz = model.mesh.edge_xyz
    axis = xyz.new_tensor([0.0, 0.0, 1.0]).expand_as(xyz)
    expected = (
        10 * (torch.linalg.cross(axis, xyz) * model.mesh.edge_normal).sum(-1)[:, None]
    )
    actual = model.normal_wind(
        eastward=10 * torch.linalg.vector_norm(xyz[:, :2], dim=-1)[:, None],
        northward=0.0,
    )
    torch.testing.assert_close(actual, expected.expand(-1, 4), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": -1.0},
        {"vapor": -0.01},
        {"vapor": 1.0},
        {"liquid": float("nan")},
        {"surface_pressure": 4000.0},
        {"temperature": torch.ones(3, dtype=torch.float64)},
        {"normal_wind": torch.zeros(1, dtype=torch.float32)},
    ],
)
def test_invalid_fields_rejected(model, change):
    with pytest.raises(ValueError):
        init(model, **change)


def test_external_array_rejected(model):
    import numpy as np

    with pytest.raises(TypeError, match="convert external data explicitly"):
        init(model, temperature=np.array(280.0))


def test_zero_vapor_rejection_and_step_validation(model):
    with pytest.raises(ValueError, match="strictly positive vapor"):
        init(model, vapor=0, liquid=0, ice=0)
    s = init(model, liquid=0, ice=0)
    out = model.step(s, 1.0)
    assert torch.count_nonzero(out.water_content[..., 1:]) == 0
    assert torch.isfinite(model.diagnostics(out)["energy"])
    for dt in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            model.step(s, dt)
    other = MoistState(
        s.u_e, s.ps, s.entropy_content, s.water_content, object(), s.column
    )
    with pytest.raises(ValueError, match="mesh and column"):
        model.diagnostics(other)


def test_initialization_diagnostic_gradients(model):
    ref = model.mesh.cell_area
    T = ref.new_tensor(280.0, requires_grad=True)
    ps = ref.new_tensor(100000.0, requires_grad=True)
    q = ref.new_tensor(0.01, requires_grad=True)

    def energy(t, p, v):
        return model.diagnostics(
            init(model, temperature=t, surface_pressure=p, vapor=v)
        )["energy"]

    E = energy(T, ps, q)
    grads = torch.autograd.grad(E, (T, ps, q))
    for i, (x, h) in enumerate(((T, 0.01), (ps, 1.0), (q, 1e-6))):
        plus = [T.detach(), ps.detach(), q.detach()]
        minus = list(plus)
        plus[i] = plus[i] + h
        minus[i] = minus[i] - h
        fd = (energy(*plus) - energy(*minus)) / (2 * h)
        torch.testing.assert_close(grads[i], fd, rtol=1e-6, atol=0)
