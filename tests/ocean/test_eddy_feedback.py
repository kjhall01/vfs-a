"""Energy feedback, exact decay and full-H heating, without a second EOS."""

import sys
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import math
import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate"),
    str(Path(__file__).resolve().parents[2] / "validation/ocean_column"),
]
from eddy import GMRedi
from eddy_energy import EddyMixingLength, EddyDissipation, EddyState
from test_gmredi import setup
from test_jmd95_hamiltonian import mesh


def law():
    return EddyMixingLength(length=10000.0, efficiency=0.1, kinetic_fraction=0.5)


@pytest.mark.parametrize("coast", [False, True])
def test_feedback_scaling_zero_and_closed_circulation(mesh, coast):
    m, state = setup(mesh, coast)
    eddy = GMRedi(kappa_redi=100.0, energy_closure=law())
    i, j = eddy._metrics(m, state)[3:5]
    k = law().diffusivity(state, i, j)
    torch.testing.assert_close(
        law().diffusivity(replace(state, eddy_h=4 * state.eddy_h), i, j), 2 * k
    )
    empty = replace(state, eddy_h=torch.zeros_like(state.eddy_h))
    end, r = GMRedi(energy_closure=law()).apply(m, empty, 60.0)
    assert torch.count_nonzero(r.gm_diffusivity) == 0
    assert torch.equal(end.ocean.temp_h, empty.ocean.temp_h)
    assert torch.equal(end.eddy_h, empty.eddy_h)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward autodiff")):
        end, r = eddy.apply(m, state, 60.0)
    scale = r.horizontal_transport.abs().max() + r.vertical_transport.abs().max()
    assert r.volume_residual.abs().max() <= 128 * torch.finfo(k.dtype).eps * scale
    mass = m.hamiltonian.mass_area(state.ocean) * state.ocean.h
    wet = mass > 0
    for residual in (r.gm_local_energy_residual, r.redi_local_energy_residual):
        assert (residual[wet].abs() / mass[wet]).max() < 1e-12
    if coast:
        assert torch.count_nonzero(end.eddy_h[~wet]) == 0


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("ratio", [1e-6, 0.1, 10.0, 1000.0])
def test_decay_heats_full_hamiltonian_and_preserves_other_fields(mesh, coast, ratio):
    m, state = setup(mesh, coast)
    decay = EddyDissipation(1000.0)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward autodiff")):
        end, r = decay.apply(m, state, ratio * 1000.0)
    torch.testing.assert_close(
        end.eddy_h, state.eddy_h * math.exp(-ratio), rtol=2e-12, atol=1e-15
    )
    for name in ("u_e", "h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(end.ocean, name), getattr(state.ocean, name))
    assert end.tke_h is state.tke_h
    assert (end.eddy_h >= 0).all()
    assert (end.ocean.temperature() >= state.ocean.temperature()).all()
    # Stored CT rounds at its existing magnitude, so the budget tolerance is
    # based on that representable thermal-energy scale, not the small decay.
    s = state.ocean
    scale = m.hamiltonian.mass_area(s) * (m.config.cp * s.temp_h.abs() + state.eddy_h)
    assert (
        r.local_energy_residual.abs() <= 32 * torch.finfo(s.h.dtype).eps * scale
    ).all()
    reduction_error = abs(
        r.thermal_energy_gain - r.dissipated_energy - r.local_energy_residual.sum()
    )
    assert reduction_error <= 32 * torch.finfo(s.h.dtype).eps * (
        r.thermal_energy_gain.abs() + r.dissipated_energy.abs()
    )
    # Independent polynomial H difference includes the temperature-dependent PE.
    a, b, _, _ = m.hamiltonian.base.coefficients(s)
    dct = end.ocean.temperature() - s.temperature()
    heat = (m.config.cp + a * (end.ocean.temperature() + s.temperature()) + b) * dct
    budget = m.hamiltonian.mass_area(s) * (s.h * heat + end.eddy_h - state.eddy_h)
    assert (budget.abs() <= 64 * torch.finfo(s.h.dtype).eps * scale).all()
    entropy = lambda x: m.ocean.reference.evaluate(x.salinity(), x.temperature())[
        "entropy"
    ]
    assert (entropy(end.ocean) >= entropy(s)).all()


def test_decay_zero_identity_and_split_semigroup(mesh):
    m, state = setup(mesh)
    empty = replace(state, eddy_h=torch.zeros_like(state.eddy_h))
    out, r = EddyDissipation(100.0).apply(m, empty, 10.0)
    assert torch.equal(out.ocean.temp_h, empty.ocean.temp_h)
    assert r.dissipated_energy == 0
    decay = EddyDissipation(100.0)
    once, _ = decay.apply(m, state, 60.0)
    half, _ = decay.apply(m, state, 30.0)
    twice, _ = decay.apply(m, half, 30.0)
    torch.testing.assert_close(once.eddy_h, twice.eddy_h, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(
        once.ocean.temp_h, twice.ocean.temp_h, rtol=1e-14, atol=1e-12
    )


def test_feedback_and_decay_backward(mesh):
    m, state = setup(mesh)
    amplitude = torch.tensor(1.0, dtype=state.eddy_h.dtype, requires_grad=True)

    def objective(alpha):
        initial = replace(state, eddy_h=state.eddy_h * alpha)
        end, _ = GMRedi(kappa_redi=100.0, energy_closure=law()).apply(m, initial, 60.0)
        end, _ = EddyDissipation(86400.0).apply(m, end, 60.0)
        return end.specific_eddy_energy().mean()

    grad = torch.autograd.grad(objective(amplitude), amplitude)[0]
    delta = 1e-4
    finite = (
        objective(amplitude.detach() + delta) - objective(amplitude.detach() - delta)
    ) / (2 * delta)
    torch.testing.assert_close(grad, finite, rtol=1e-7, atol=1e-10)
    empty = replace(state, eddy_h=torch.zeros_like(state.eddy_h, requires_grad=True))
    i, j = GMRedi._metrics(m, state)[3:5]
    grad = torch.autograd.grad(law().diffusivity(empty, i, j).sum(), empty.eddy_h)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


def test_invalid_feedback_parameters_and_state(mesh):
    for args in (
        (0.0, 0.1, 0.5),
        (100.0, 0.0, 0.5),
        (100.0, 0.1, 0.0),
        (100.0, 0.1, 1.1),
        (math.inf, 0.1, 0.5),
    ):
        with pytest.raises(ValueError):
            EddyMixingLength(*args)
    for tau in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            EddyDissipation(tau)
    with pytest.raises(ValueError):
        GMRedi(100.0, energy_closure=law())
    with pytest.raises(TypeError):
        GMRedi(energy_closure=object())
    m, state = setup(mesh)
    for dt in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            EddyDissipation(100.0).apply(m, state, dt)
    with pytest.raises(ValueError):
        EddyDissipation(100.0).apply(m, replace(state, eddy_h=-state.eddy_h), 1.0)
    huge = replace(state, eddy_h=state.ocean.h * 1e9)
    with pytest.raises(ValueError):
        EddyDissipation(1.0).apply(m, huge, 1.0)
