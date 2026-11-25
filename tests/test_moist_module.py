"""Ownership and delegation checks, not numerical qualification of new schemes."""

from dataclasses import FrozenInstanceError, is_dataclass, replace

import pytest
import torch

from test_model_api import model, init
from vfs import MoistAtmosphere, BoundedSSPRK3
from vfs.atmosphere import MoistConservativeDynamics, MoistState, BoundedGibbs
from vfs.atmosphere.bounded_transport import graph

KEYS = ("u_e", "ps", "entropy_content", "water_content")


def equal(a, b):
    for name in KEYS:
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)


def test_registration_and_state_dict(model):
    modules = list(model.named_modules(remove_duplicate=False))
    assert [n for n, _ in modules] == [
        "",
        "mesh",
        "column",
        "dynamics",
        "dynamics.kernel",
    ]
    assert len({id(m) for _, m in modules}) == len(modules)
    assert not isinstance(model.transport, torch.nn.Module)
    assert model.transport.base is model.dynamics
    buffers = list(model.named_buffers(remove_duplicate=False))
    assert buffers and all(n.startswith(("mesh.", "column.")) for n, _ in buffers)
    assert len({id(b) for _, b in buffers}) == len(buffers)
    assert not list(model.parameters())
    s = init(model)
    assert is_dataclass(s) and not isinstance(s, torch.nn.Module)
    before = set(model.state_dict())
    model(s, 1.0)
    assert set(model.state_dict()) == before == {n for n, _ in buffers}


def test_delegation_and_hooks(model):
    m = MoistAtmosphere(model.mesh, model.column, stepper=BoundedSSPRK3())
    s = init(m, normal_wind=1.0)
    expected = BoundedGibbs(m.dynamics).step(s, 1.0)
    seen = []
    handle = m.register_forward_hook(lambda module, args, out: seen.append(out))
    equal(m(s, 1.0), expected)
    equal(m.step(s, 1.0), expected)
    assert len(seen) == 2
    handle.remove()


@pytest.mark.parametrize("name", ["mesh", "column", "dynamics", "stepper", "transport"])
def test_fixed_components(model, name):
    with pytest.raises(AttributeError, match="cannot be replaced"):
        setattr(model, name, getattr(model, name))
    with pytest.raises(AttributeError, match="cannot be deleted"):
        delattr(model, name)


def test_borrowed_references_and_reuse(model):
    with pytest.raises(ValueError, match="already bound"):
        MoistAtmosphere(model.mesh, model.column, stepper=model.stepper)
    with pytest.raises(FrozenInstanceError):
        model.stepper._dynamics = None
    s = init(model)
    model.transport.base = MoistConservativeDynamics()
    with pytest.raises(ValueError, match="transport.base"):
        model(s, 1.0)


@pytest.mark.parametrize(
    "bad",
    ["ssprk3", object(), lambda s, dt: s, BoundedGibbs(MoistConservativeDynamics())],
)
def test_reject_unsupported_stepper(model, bad):
    with pytest.raises(TypeError, match="stepper must be"):
        MoistAtmosphere(model.mesh, model.column, stepper=bad)


def test_reject_configuration(model):
    with pytest.raises(TypeError, match="release_as_heat"):
        MoistAtmosphere(model.mesh, model.column, release_as_heat=False)


@pytest.mark.parametrize("field", KEYS)
@pytest.mark.parametrize("entry", ["step", "physical_fields", "diagnostics"])
def test_dtype_mismatch(model, field, entry):
    s = init(model)
    s = replace(s, **{field: getattr(s, field).float()})
    with pytest.raises(ValueError, match="state_to_model"):
        getattr(model, entry)(s, *([1.0] if entry == "step" else []))


def test_device_and_geometry_mismatch(model):
    s = init(model)
    with pytest.raises(ValueError, match="dtype/device"):
        model(replace(s, u_e=s.u_e.to("meta")), 1.0)
    with pytest.raises(ValueError, match="mesh and column"):
        model.state_to_model(replace(s, column=object()))
    with pytest.raises(ValueError, match="mesh and column"):
        model.state_to_model(replace(s, mesh=object()))


def test_conversion_and_gradients(model):
    s = init(model)
    leaves = [getattr(s, k).clone().requires_grad_() for k in KEYS]
    s = MoistState(*leaves, s.mesh, s.column)
    same = model.state_to_model(s)
    assert all(getattr(same, k) is x for k, x in zip(KEYS, leaves))
    integers = {
        n: b.clone() for n, b in model.named_buffers() if not b.is_floating_point()
    }
    model.float()
    with pytest.raises(ValueError, match="dtype/device"):
        model(s, 1.0)
    moved = model.state_to_model(s)
    assert moved.mesh is model.mesh and moved.column is model.column
    assert all(getattr(moved, k).dtype == torch.float32 for k in KEYS)
    grads = torch.autograd.grad(sum(getattr(moved, k).sum() for k in KEYS), leaves)
    assert all(torch.equal(g, torch.ones_like(x)) for g, x in zip(grads, leaves))
    for n, b in model.named_buffers():
        if n in integers:
            assert b.dtype == integers[n].dtype and torch.equal(b, integers[n])
    fresh = init(model)
    assert fresh.water_content.dtype == torch.float32
    _, i, j, _ = graph(model.dynamics, fresh)
    assert i.dtype == j.dtype == torch.int64


def test_checkpoint_roundtrip(model):
    # Same topology/configuration; default load copies buffers without replacing owners.
    from vfs import VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients

    mesh = VoronoiMesh(n_cells=32, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(4, p_top=5000.0)
    col = CharneyPhillipsColumn(
        4, A, B, phi_s=100 * mesh.cell_xyz[:, 0], phi_full_mode="direct_u_gl2_adjoint"
    )
    restored = MoistAtmosphere(mesh, col)
    restored.load_state_dict(model.state_dict())
    assert restored.mesh is mesh and restored.column is col
    assert restored.transport.base is restored.dynamics
    equal(model(init(model), 1.0), restored(init(restored), 1.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_forward(model):
    model.to("cuda")
    s = init(model)
    out = model(s, 1.0)
    assert all(torch.isfinite(getattr(out, k)).all() for k in KEYS)
