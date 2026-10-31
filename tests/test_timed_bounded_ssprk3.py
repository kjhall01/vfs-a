"""The extracted stage map retains atmospheric arithmetic and timed Euler budgets."""
from dataclasses import replace

import torch

from test_bounded_gradients import smooth_case, KEYS
from vfs.atmosphere.bounded_moist import bounded_ssprk3_with_info


def test_timed_stages_and_prescribed_state_handling():
    model, initial = smooth_case()
    time, dt = 7., .25
    calls = []

    def velocity(t):
        return torch.full_like(initial.u_e, 2. + t * t)

    def advance(state, t, budget):
        # Verification replaces wind before production transport. Deliberately
        # poisoned Euler wind must never enter the next production evaluation.
        prescribed = replace(state, u_e=velocity(t))
        updated, info = model.transport.euler_with_info(prescribed, budget)
        expected = info['amounts'][:, 1:].reshape_as(updated.water_content)
        expected = expected / state.mesh.cell_area[:, None, None]
        assert torch.equal(updated.water_content, expected)
        calls.append((t, budget, prescribed, updated, info))
        return replace(updated, u_e=torch.full_like(updated.u_e, 1e9)), info

    saved = {key: getattr(initial, key).clone() for key in KEYS}
    result, infos = bounded_ssprk3_with_info(advance, initial, time, dt)
    result = replace(result, u_e=velocity(time + dt))
    assert [(t, budget) for t, budget, *_ in calls] == [
        (time, dt), (time + dt, dt), (time + dt / 2, dt)
    ]
    assert all(info is call[-1] for info, call in zip(infos, calls))
    assert torch.equal(result.u_e, velocity(time + dt))
    assert not torch.equal(result.u_e, velocity(time + dt / 2))
    for key in KEYS:
        assert torch.equal(getattr(initial, key), saved[key])
    for key in KEYS[1:]:
        assert torch.equal(getattr(calls[1][2], key), getattr(calls[0][3], key))
        expected = .75 * getattr(initial, key) + .25 * getattr(calls[1][3], key)
        assert torch.equal(getattr(calls[2][2], key), expected)
    assert result.mesh is initial.mesh and result.column is initial.column


def test_atmospheric_delegation_exact():
    model, initial = smooth_case()
    expected, expected_info = model.transport.step_with_info(initial, 10.)
    actual, actual_info = bounded_ssprk3_with_info(
        lambda state, time, dt: model.transport.euler_with_info(state, dt),
        initial, 123., 10.,
    )
    for key in KEYS:
        assert torch.equal(getattr(actual, key), getattr(expected, key))
    for left, right in zip(actual_info, expected_info):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                assert torch.equal(left[key], right[key]), key
            else:
                assert left[key] == right[key], key
