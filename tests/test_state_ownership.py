"""Tensor ownership and explicit conversion; no new numerical qualification."""

import pytest
import torch

from vfs import DryAtmosphere, State
from vfs.core.state import PrognosticFields
from vfs.field import Field
from test_conservative_imex import setup


@pytest.fixture
def dry():
    _, state, _ = setup()
    return DryAtmosphere(mesh=state.mesh, column=state.column), state


def test_tensor_payload_and_legacy_field_assignment(dry):
    model, state = dry
    wind = state.u_e.requires_grad_()
    field = Field(wind, location="edge", vertical_location="full")
    payload = PrognosticFields(field, state.mtheta_i, state.ps)
    assert payload.u_e is wind
    payload.u_e = field
    state.u_e = field
    assert payload.u_e is state.u_e is wind
    for fields in (payload, payload.zeros_like(), payload + payload):
        assert all(isinstance(getattr(fields, k), torch.Tensor) for k in ("u_e", "mtheta_i", "ps"))
    torch.testing.assert_close(torch.autograd.grad((payload + payload).u_e.sum(), wind)[0], torch.full_like(wind, 2))
    with pytest.raises(ValueError, match="edge/full"):
        payload.u_e = Field(wind, location="cell", vertical_location="full")
    with pytest.raises(TypeError, match="torch.Tensor"):
        payload.ps = [1.0]
    # Standalone Field still supplies placement-aware mesh operations.
    wrapped = model.mesh.grad(Field(state.ps, location="cell", vertical_location="surface"))
    assert isinstance(wrapped, Field)
    torch.testing.assert_close(wrapped.values, model.mesh.grad(state.ps))


@pytest.mark.parametrize("geometry", ["mesh", "column"])
def test_unrelated_geometry_rejected_and_unbound_payload_shared(dry, geometry):
    model, state = dry
    _, other, _ = setup()
    mismatched = State(fields=state.fields, mesh=state.mesh, column=state.column)
    setattr(mismatched, geometry, getattr(other, geometry))
    for operation in (model.bind_state, model.state_to_model, model.physical_fields):
        with pytest.raises(ValueError):
            operation(mismatched)
    unbound = State(fields=state.fields)
    bound = model.bind_state(unbound)
    assert bound.fields is unbound.fields and bound.mesh is model.mesh and bound.column is model.column
    assert unbound.mesh is unbound.column is None
    with pytest.raises(ValueError, match="borrow"):
        model.state_to_model(unbound)


def test_explicit_conversion_preserves_gradients_and_geometry(dry):
    model, state = dry
    inputs = tuple(getattr(state, k).float().requires_grad_() for k in ("u_e", "mtheta_i", "ps"))
    tracer = torch.ones(state.mesh.nCells, state.column.n_levels, 2, dtype=torch.float32, requires_grad=True)
    original = State(*inputs, tracer_m=tracer, mesh=state.mesh, column=state.column)
    with pytest.raises(ValueError, match="state_to_model"):
        model.step(original, 1.0)
    with pytest.raises(ValueError, match="dtype/device"):
        model.physical_fields(state.to(device="meta"))
    converted = model.state_to_model(original)
    assert converted.mesh is original.mesh and converted.column is original.column
    assert original.u_e is inputs[0] and original.u_e.dtype == torch.float32
    values = tuple(getattr(converted, k) for k in ("u_e", "mtheta_i", "ps", "tracer_m"))
    assert all(x.dtype == torch.float64 for x in values)
    for gradient in torch.autograd.grad(sum(x.sum() for x in values), (*inputs, tracer)):
        assert torch.equal(gradient, torch.ones_like(gradient))
    with pytest.raises(ValueError, match="convert explicitly"):
        State.from_theta(state.u_e, state.theta(), state.ps, mesh=state.mesh, column=state.column, q_tracers=tracer)
    assert model.state_to_model(state).u_e is state.u_e


def test_diagnostics_and_step_do_not_mutate_inputs_or_cache_fields(dry):
    model, state = dry
    q = state.ps.new_full((state.mesh.nCells, state.column.n_levels, 2), 0.01).requires_grad_()
    state = State.from_theta(state.u_e, state.theta(), state.ps, mesh=state.mesh, column=state.column, q_tracers=q)
    names = ("u_e", "mtheta_i", "ps", "tracer_m")
    refs = [getattr(state, k) for k in names]
    before = [x.clone() for x in refs]
    # Tensor containers/I/O retain tracer layout, but the retained conservative
    # dycore intentionally rejects passive tracers rather than changing physics.
    torch.testing.assert_close(state.tracers(), q)
    torch.testing.assert_close(torch.autograd.grad(state.tracers().sum(), q)[0], torch.ones_like(q))
    with pytest.raises(ValueError, match="dry states only"):
        model(state, 1.0)
    dry_state = State(state.u_e, state.mtheta_i, state.ps, mesh=state.mesh, column=state.column)
    fields = model.physical_fields(dry_state)
    torch.testing.assert_close(fields["temperature"], dry_state.temperature())
    changed = State(state.u_e, state.mtheta_i * 1.01, state.ps, mesh=state.mesh, column=state.column)
    assert not torch.equal(model.physical_fields(changed)["temperature"], fields["temperature"])
    model.step(dry_state, 1.0)
    for name, ref, saved in zip(names, refs, before):
        assert getattr(state, name) is ref
        torch.testing.assert_close(ref, saved, rtol=0, atol=0)
    assert not isinstance(state, torch.nn.Module)
    assert not any(id(x) in {id(t) for t in refs} for x in model.buffers())


@pytest.mark.parametrize('implicit', [False, True])
def test_retained_dry_model_delegation_and_module_ownership(dry, implicit):
    from vfs import DryAtmosphere, DrySSPRK3, DryIMEX
    from vfs.core.integrators import ssprk3
    from vfs.imex import DryPEIMEXARK, IMEXSolverConfig

    _, state = dry
    cfg = IMEXSolverConfig(implicit_operator='gravity_wave', preconditioner='hydrostatic_schur')
    model = DryAtmosphere(state.mesh, state.column, stepper=DryIMEX(cfg) if implicit else DrySSPRK3())
    expected = (DryPEIMEXARK(model.dynamics, cfg).step(state, 1.) if implicit
                else ssprk3(model.dynamics, state, 1.))
    modules = list(model.named_modules(remove_duplicate=False))
    assert [n for n, _ in modules] == ['', 'mesh', 'column', 'dynamics']
    assert len({id(m) for _, m in modules}) == len(modules)
    seen = []
    hook = model.register_forward_hook(lambda module, args, out: seen.append(out))
    for actual in (model(state, 1.), model.step(state, 1.)):
        for name in ('u_e', 'mtheta_i', 'ps'):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    hook.remove()
    assert len(seen) == 2
    assert all(n.startswith(('mesh.', 'column.')) for n in model.state_dict())
    assert not list(model.parameters())
    with pytest.raises(AttributeError, match='replaced'):
        model.dynamics = model.dynamics
    if implicit:
        solver = model.stepper.solver
        model(state, 1.)
        assert model.stepper.solver is solver
        solver.dynamics = object()
        with pytest.raises(ValueError, match='binding'):
            model(state, 1.)


def test_explicit_entropy_transport_view_shares_tensors_and_gradients():
    from test_moist_imex import setup as moist_setup
    from vfs.atmosphere.scalar_transport import ScalarTransportState
    dynamics, state, _ = moist_setup()
    state.entropy_content.requires_grad_()
    view = dynamics.transport_state(state)
    assert isinstance(view, ScalarTransportState)
    assert not isinstance(view, State)
    assert not hasattr(view, 'mtheta_i') and not hasattr(view, 'theta')
    assert view.scalar_content is state.entropy_content
    assert view.u_e is state.u_e and view.ps is state.ps
    assert view.mesh is state.mesh and view.column is state.column
    derivative = torch.autograd.grad(view.specific_scalar().sum(), state.entropy_content)[0]
    from vfs.core.state import interface_mass
    torch.testing.assert_close(derivative, 1 / interface_mass(view.layer_dp()), rtol=0, atol=0)
