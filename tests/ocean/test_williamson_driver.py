"""Benchmark/reference integrity; no scientific accuracy acceptance thresholds."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('williamson',
    Path(__file__).resolve().parents[2] / 'tools/verify_ocean_williamson.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_continuum_initial_state_satisfies_rotating_momentum_balance():
    # Independent differentiation of h and |v|^2, projected onto the sphere.
    torch.set_default_dtype(torch.float64)
    xyz = torch.tensor([[1., 0., 0.], [0., 0., 1.], [1., 2., 3.], [-2., 3., -1.]])
    xyz = (xyz / xyz.norm(dim=-1, keepdim=True)).requires_grad_()
    velocity, h = driver.analytic_fields(xyz)
    assert h[0] == driver.H0
    assert torch.count_nonzero(velocity[1]) == 0
    bernoulli = driver.GRAVITY * h[:, 0] + .5 * velocity.square().sum(-1)
    gradient, = torch.autograd.grad(bernoulli.sum(), xyz)
    surface_gradient = (gradient - (gradient * xyz).sum(-1, keepdim=True) * xyz) / driver.RADIUS
    vorticity = 2 * (driver.OMEGA + driver.U0 / driver.RADIUS) * xyz[:, 2:3]
    rotation = -vorticity * torch.cross(xyz, velocity, dim=-1)
    residual = rotation - surface_gradient
    assert residual.abs().max() <= 64 * torch.finfo(xyz.dtype).eps * rotation.abs().max()


def test_frozen_mesh_replay_and_changed_initial_data_rejection(tmp_path):
    torch.set_default_dtype(torch.float64)
    model, state = driver.case_inputs(16, 0)
    driver.save_inputs(tmp_path / 'inputs_16.npz', model, state, 0)
    repeated, values = driver.case_inputs(16, 0, tmp_path)
    for key, tensor in model.mesh.named_buffers():
        assert torch.equal(tensor, dict(repeated.mesh.named_buffers())[key])
    for key in driver.capture.FIELDS:
        assert torch.equal(getattr(state, key), getattr(values, key))
    with pytest.raises(ValueError, match='parameters differ'):
        driver.case_inputs(16, 1, tmp_path)
    state.h[0] += 1
    driver.save_inputs(tmp_path / 'inputs_16.npz', model, state, 0)
    with pytest.raises(ValueError, match='initial field differs'):
        driver.case_inputs(16, 0, tmp_path)


def test_reported_endpoint_metrics_and_units():
    torch.set_default_dtype(torch.float64)
    model, initial = driver.case_inputs(16, 0)
    endpoint, series = driver.integrate(model, initial, 60, 120, sample_interval=60)
    assert [row['time_s'] for row in series] == [0, 60, 120]
    assert series[0]['h_error_m']['l2'] == 0
    assert series[0]['u_error_m_s']['linf'] == 0
    expected = driver.diagnostics(model, endpoint, initial, model.budgets(initial))
    assert {key: value for key, value in series[-1].items() if key != 'time_s'} == expected
    weights = torch.tensor([[1.], [3.]])
    result = driver.norms(torch.tensor([[2.], [-4.]]), weights)
    assert result == dict(l1=3.5, l2=13**.5, linf=4.)
    with pytest.raises(ValueError, match='divide'):
        driver.integrate(model, initial, 70, 120, sample_interval=60)
