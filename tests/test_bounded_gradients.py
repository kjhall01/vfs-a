"""Regression checks for validation and one-step, branch-local derivatives."""

from dataclasses import replace
import json

import pytest
import torch

from test_extraction import ROOT, load
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)
from vfs.atmosphere.bounded_transport import bounded_transfer

KEYS = ("u_e", "ps", "entropy_content", "water_content")
SCALES = dict(zip(KEYS, (10.0, 1e5, 1e7, 100.0)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("carrier", [-0.125, 0.0, 0.125])
@pytest.mark.parametrize("offset", [0.0, 0.5, 1.0, 2.0])
def test_flux_sum_inequality(dtype, carrier, offset):
    m = torch.ones(2, dtype=dtype)
    amounts = torch.tensor([[0.8, 0.1, 0.05, 0.05]] * 2, dtype=dtype)
    G = torch.tensor([carrier], dtype=dtype)
    eps = torch.finfo(dtype).eps
    scale = G.abs().max().clamp_min(torch.finfo(dtype).tiny)
    # Reference allclose receives a nongradient scalar tolerance. The call under
    # test receives a gradient-carrying carrier and species flux, never detached.
    delta = offset * (128 * eps * scale + 128 * eps * G.abs())
    reference_flux = torch.stack([G + delta, G * 0, G * 0, G * 0], -1)
    expected = torch.allclose(
        reference_flux.sum(-1), G, rtol=128 * eps, atol=float(128 * eps * scale)
    )
    G.requires_grad_()
    J = torch.stack([G + delta, G * 0, G * 0, G * 0], -1)
    args = (m, amounts, torch.tensor([0]), torch.tensor([1]), G, J)
    if not expected:
        with pytest.raises(ValueError, match="species fluxes must sum"):
            bounded_transfer(*args)
    else:
        result, flux, _ = bounded_transfer(*args)
        assert flux is J  # Safe high-order ledger retains tensor ownership.
        result[1, 0].backward()
        torch.testing.assert_close(G.grad, torch.ones_like(G), rtol=0, atol=0)


def test_zero_carrier_inconsistent_sum_and_overflow_rejected():
    for dtype, carrier, species in [
        (torch.float64, 0.0, [1e-10, 0.0, 0.0, 0.0]),
        # bfloat16's 128*eps=1 makes the tolerance overflow, so a bare
        # inf<=inf would incorrectly pass. Preserve allclose's finite-error rule.
        (
            torch.bfloat16,
            torch.finfo(torch.bfloat16).max,
            [torch.finfo(torch.bfloat16).max] * 2 + [0.0, 0.0],
        ),
    ]:
        with pytest.raises(ValueError, match="species fluxes must sum"):
            bounded_transfer(
                torch.ones(2, dtype=dtype),
                torch.full((2, 4), 0.25, dtype=dtype),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([carrier], dtype=dtype, requires_grad=True),
                torch.tensor([species], dtype=dtype, requires_grad=True),
            )


def smooth_case():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(4, p_top=5000.0)
    model = MoistAtmosphere(
        mesh, CharneyPhillipsColumn(4, A, B, phi_full_mode="direct_u_gl2_adjoint")
    )
    p = model.interface_pressure(100000.0)
    x = mesh.cell_xyz[:, :1]
    state = model.state_from_fields(
        temperature=280 * (p / 100000) ** 0.08 * (1 + 0.01 * x),
        surface_pressure=100000.0,
        normal_wind=model.normal_wind(eastward=10.0, northward=2.0),
        vapor=0.01 * (1 + 0.1 * x),
        liquid=0.001 * (1 + 0.1 * x),
        ice=0.0001 * (1 + 0.1 * x),
    )
    return model, state


def objective(state):
    # Nonuniform weights avoid a conserved global sum hiding derivative errors.
    total = 0
    for k in KEYS:
        value = getattr(state, k)
        weights = torch.sin(
            torch.arange(value.numel(), dtype=value.dtype, device=value.device).reshape(
                value.shape
            )
            * 0.37
            + 0.2
        )
        total = total + (weights * value / SCALES[k]).mean()
    return total


def directional_probe(model, state, key, dt, active):
    leaf = getattr(state, key).clone().requires_grad_()
    differentiated = replace(state, **{key: leaf})
    # Exercise the public module through one complete SSPRK3 step.
    objective(model(differentiated, dt)).backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    if active:
        # Preserve exactly absent condensates and cloud support under +/- probes.
        direction = 0.01 * getattr(state, key)
    else:
        direction = (
            torch.cos(
                torch.arange(leaf.numel(), dtype=leaf.dtype).reshape(leaf.shape) * 0.29
                + 0.7
            )
            * SCALES[key]
            * 0.01
        )
    ad = (leaf.grad * direction).sum()
    _, stages = model.transport.step_with_info(state, dt)
    assert all(
        s["limited_faces"] > 0 if active else s["limited_faces"] == 0 for s in stages
    )
    rows = []
    for h in (0.1, 0.03, 0.01):
        values = []
        for sign in (-1, 1):
            perturbed = replace(
                state, **{key: getattr(state, key) + sign * h * direction}
            )
            out, info = model.transport.step_with_info(perturbed, dt)
            # No switch in the limiter-active mask or discrete subcycle count.
            for current, anchor in zip(info, stages):
                assert current["substeps"] == anchor["substeps"]
                assert torch.equal(current["beta"] < 1, anchor["beta"] < 1)
                if not active:
                    floors = leaf.new_tensor(
                        [
                            model.transport.dry_floor,
                            model.transport.vapor_floor,
                            0.0,
                            0.0,
                        ]
                    )
                    margin = (
                        current["amounts"] / current["target_mass"][:, None] - floors
                    )
                    assert margin.min() > 128 * torch.finfo(leaf.dtype).eps
            values.append(objective(out))
        fd = (values[1] - values[0]) / (2 * h)
        # Roundoff floor for subtracting O(objective) values and dividing by h.
        rounding = (
            64 * torch.finfo(leaf.dtype).eps * (1 + abs(values[0]) + abs(values[1])) / h
        )
        rows.append((fd, rounding))
    # Central differences have O(h^2) truncation error. The .03/.01 pair
    # estimates the last truncation error as |D(.03)-D(.01)|/(3^2-1).
    # Four times that estimate plus the roundoff floor is a numerical test
    # allowance, not a physical acceptance threshold or universal error bound.
    allowance = 4 * abs(rows[-2][0] - rows[-1][0]) / 8 + rows[-1][1]
    assert abs(ad - rows[-1][0]) <= allowance
    print(
        json.dumps(
            dict(
                case="active" if active else "smooth",
                field=key,
                ad=float(ad),
                fd=[float(r[0]) for r in rows],
                absolute_errors=[float(abs(ad - r[0])) for r in rows],
                numerical_allowance=float(allowance),
                limited_faces=[s["limited_faces"] for s in stages],
                substeps=[s["substeps"] for s in stages],
            )
        )
    )


@pytest.mark.parametrize("key", KEYS)
def test_smooth_complete_step_directional_derivative(key):
    model, state = smooth_case()
    directional_probe(model, state, key, 10.0, active=False)


@pytest.mark.parametrize("key", KEYS)
def test_active_cloud_complete_step_directional_derivative(key):
    torch.set_num_threads(1)
    _, state = load(ROOT / "examples/bounded_moist.py").initialize()
    model = MoistAtmosphere(state.mesh, state.column)
    directional_probe(model, state, key, 30.0, active=True)


def test_limiter_switch_has_distinct_one_sided_derivatives():
    torch.set_default_dtype(torch.float64)
    m = torch.ones(2)
    amounts = torch.tensor([[0.8, 0.1, 0.05, 0.05]] * 2)
    G = torch.tensor([0.1])

    def transfer(z):
        J = torch.stack([G - z, G * 0, z, G * 0], -1)
        result, _, info = bounded_transfer(
            m, amounts, torch.tensor([0]), torch.tensor([1]), G, J
        )
        return result[0, 2], info["limited_faces"]

    z = torch.tensor([0.05], requires_grad=True)
    at, count = transfer(z)
    at.backward()
    assert count == 0  # Equality uses the safe-high-order branch.
    assert z.grad.item() == -1.0
    h = 1e-5
    left, lc = transfer(torch.tensor([0.05 - h]))
    right, rc = transfer(torch.tensor([0.05 + h]))
    assert lc == 0 and rc == 1
    # The existing 64*eps limiter margin perturbs the right endpoint by O(eps).
    atol = 128 * torch.finfo(z.dtype).eps / h
    torch.testing.assert_close((at - left) / h, at.new_tensor(-1.0), rtol=0, atol=atol)
    torch.testing.assert_close((right - at) / h, at.new_tensor(0.0), rtol=0, atol=atol)
