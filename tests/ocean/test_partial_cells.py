from pathlib import Path

import pytest
import torch

from vfs.ocean.partial_cells import (
    WetFaces,
    reference_thickness,
    remap_active_contents,
    remap_roquet,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def shelf():
    depth = tensor([22, 30, 0])
    h = reference_thickness(depth, tensor([10, 10, 10]))
    cells = torch.tensor([[0, 1], [1, 2], [2, 0]])
    return depth, h, cells


def test_bottom_merge_and_extra_depth():
    h = reference_thickness(tensor([0, 1, 10, 12, 12.5, 20, 39]), tensor([10, 10, 10]))
    torch.testing.assert_close(
        h,
        tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [10, 0, 0],
                [12, 0, 0],
                [10, 2.5, 0],
                [10, 10, 0],
                [10, 10, 19],
            ]
        ),
    )


def test_split_face_work_and_flux():
    depth, h, cells = shelf()
    f = WetFaces.from_columns(depth, h, cells)
    torch.testing.assert_close(f.opening, tensor([10, 10, 2]))
    assert f.left.tolist() == [0, 1, 1]
    assert f.right.tolist() == [3, 4, 5]
    p = tensor([[1, 2, 0], [6, 0, -4], [0, 0, 0]])
    u = tensor([-0.3, 0.2, 1.0])
    length = tensor([4, 5, 6])
    distance = tensor([7, 8, 9])
    flux = f.volume_flux(u, length)
    force = -f.jump(p) / distance[f.edge]
    work = (p * f.exchange(flux)).sum() + (distance[f.edge] * flux * force).sum()
    torch.testing.assert_close(work, tensor(0.0))
    torch.testing.assert_close(f.exchange(flux).sum(), tensor(0.0))
    moved = h.clone()
    moved[0, 0] += 2
    moved[1, 0] -= 1
    torch.testing.assert_close(
        WetFaces.from_columns(depth, moved, cells).opening.sum(), tensor(21.0)
    )


def test_budgeted_ale_merge_and_shear():
    root = Path(__file__).resolve().parents[2] / "validation"
    from pytest import MonkeyPatch

    with MonkeyPatch.context() as paths:
        for name in ("ocean_column", "temperature_coordinate"):
            paths.syspath_prepend(str(root / name))
        from roquet import RoquetEnergy, RoquetFreezingEOS
    depth, target, cells = shelf()
    h = tensor([[8, 12, 2], [11, 10, 9], [0, 0, 0]])
    t = tensor([[17, 9, 4], [14, 8, 5], [0, 0, 0]])
    s = tensor([[33, 34, 35], [34, 35, 36], [0, 0, 0]])
    old = WetFaces.from_columns(depth, h, cells)
    u = torch.linspace(-1, 1, len(old.edge), dtype=h.dtype)
    energy = RoquetEnergy(RoquetFreezingEOS(), gravity=9.81, rho0=1025)
    result = remap_roquet(
        depth,
        h,
        target,
        h * t,
        h * s,
        u,
        cells,
        tensor([4, 5, 6]),
        tensor([9, 10, 11]),
        energy,
    )
    torch.testing.assert_close(result.salt_h.sum(1), (h * s).sum(1))
    torch.testing.assert_close(
        (result.faces.opening * result.velocity).sum(), (old.opening * u).sum()
    )
    assert result.edge_kinetic_change[0] < 0
    assert result.column_energy_residual.abs().max() < 1e-5
    assert result.column_entropy_change.min() >= 0
    assert not result.temp_h[target == 0].any()
    tke = remap_active_contents(h, h * t.square(), target)
    torch.testing.assert_close(tke.sum(1), (h * t.square()).sum(1))
    assert tke.min() >= 0


def test_ale_rejects_lost_inventory():
    _, h, _ = shelf()
    with pytest.raises(ValueError, match="inactive inventory"):
        remap_active_contents(h, torch.ones_like(h), h)
    with pytest.raises(ValueError, match="total thickness"):
        remap_active_contents(h, h, 2 * h)


def test_shelf_eddy_operators():
    from vfs.ocean.partial_cell_eddy import ShelfRedi, gm_volume_carriers

    depth, h, cells = shelf()
    faces = WetFaces.from_columns(depth, h, cells)
    length = tensor([8, 9, 7])
    z = h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - h / 2
    operator = ShelfRedi.build(
        faces, depth, h, 1.0e-4 * z, tensor([1000, 1050, 960]), length, 100.0
    )
    torch.testing.assert_close(operator.inventory_tendency(z), torch.zeros_like(h))
    q = tensor([[1, 7, 0], [3, 0, 5], [0, 0, 0]])
    dq = operator.inventory_tendency(q)
    torch.testing.assert_close(dq.sum(), tensor(0.0))
    assert (q * dq).sum() < 0 and not dq[h == 0].any()
    assert 0 < operator.step_limit(1e5 * h) < torch.inf
    flux, down = gm_volume_carriers(faces, h, length, tensor([0.1, 0.3, 0.0]))
    up = torch.cat((torch.zeros_like(down[:, :1]), down[:, :-1]), 1)
    torch.testing.assert_close(faces.exchange(flux) + up - down, torch.zeros_like(h))


def test_kinetic_derivative_and_pressure_work():
    from vfs.ocean.partial_cell_dynamics import (
        kinetic_thickness_conjugate,
        pressure_transport,
    )

    depth, _, cells = shelf()
    # Different free surfaces and interface depths avoid topology switches.
    h = tensor([[9.5, 13, 0], [10, 10, 10], [0, 0, 0]]).requires_grad_()
    f = WetFaces.from_columns(depth, h, cells)
    u = torch.linspace(-0.2, 0.4, len(f.edge), dtype=h.dtype)
    distance, length, A = tensor([20, 21, 22]), tensor([11, 12, 13]), tensor([5, 9, 7])
    ke = f.kinetic_energy(u, distance * length, 1025.0)
    (derivative,) = torch.autograd.grad(ke.sum(), h)
    actual = kinetic_thickness_conjugate(f, depth, h, u, distance * length, 1025.0)
    torch.testing.assert_close(actual[h > 0], derivative[h > 0])
    ct = h * 0.3
    sa = h * 0.2 + 34
    gh, gt, gs = actual + h.square(), h + ct, sa.square()
    rhs = pressure_transport(f, u, length, distance, A, 1025.0, ct, sa, gh, gt, gs)
    work = (rhs.h * gh + rhs.temp_h * gt + rhs.salt_h * gs).sum()
    work += (1025.0 * (distance * length)[f.edge] * f.opening * u * rhs.velocity).sum()
    assert work.abs() < 1e-8


def test_energy_settlement_has_independent_endpoint_budget(monkeypatch):
    from vfs.ocean.partial_cell_eddy import ShelfRedi, advance_gm_roquet

    root = Path(__file__).resolve().parents[2] / "validation"
    for name in ("ocean_column", "temperature_coordinate"):
        monkeypatch.syspath_prepend(str(root / name))
    from roquet import RoquetEnergy, RoquetFreezingEOS

    depth, h, cells = shelf()
    area, length, dc = (
        tensor([105, 125, 135]),
        tensor([10, 12, 14]),
        tensor([40, 41, 42]),
    )
    f = WetFaces.from_columns(depth, h, cells)
    z = h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - h / 2
    t = 17 + 0.3 * z + tensor([1, 0, 0])[:, None]
    s = 34 + 0.02 * z
    ht, hs, he = h * t, h * s, h * 2
    eos = RoquetFreezingEOS()
    energy = RoquetEnergy(eos, gravity=9.81, rho0=1025)
    op = ShelfRedi.build(f, depth, h, 1.0e-4 * z, dc, length, 1.0)

    def inventory(ht, hs, he):
        divisor = torch.where(h > 0, h, 1.0)
        t, s = ht / divisor, hs / divisor
        f0 = eos.polynomial(t, s, 0.0) - eos.polynomial(
            t.new_tensor(eos.CT0), eos.SA0, 0.0
        )
        psi = (
            energy.gravity
            / energy.rho0
            * (z * f0 - 0.5 * eos.tz * (t - eos.CT0) * (z.square() + h.square() / 12))
        )
        # CT has its own conservative flux. Check mechanical + eddy energy
        # separately, so large reference heat cannot hide a settlement error.
        return torch.stack([(area[:, None] * q).sum() for q in (ht, hs, h * psi + he)])

    start = inventory(ht, hs, he)
    args = (depth, h, ht, hs, he, area, energy)
    for end in (
        op.advance_roquet(*args, dt=0.1),
        advance_gm_roquet(f, *args, length, tensor([0.02, 0.01, -0.01]), 0.1),
    ):
        torch.testing.assert_close(inventory(*end), start, rtol=2e-15, atol=1e-10)
        assert (end[2] >= 0).all()


def test_one_layer_and_missing_neighbors():
    h = tensor([[2], [4], [0]])
    depth = h[:, 0]
    cells = torch.tensor([[-1, 0], [0, 1], [1, 2], [1, -1]])
    f = WetFaces.from_columns(depth, h, cells)
    assert f.edge.tolist() == [1]
    torch.testing.assert_close(f.opening, tensor([2]))
    closed = WetFaces.from_columns(depth, h, cells[:1])
    u, dke = closed.remap_velocity(
        torch.empty_like(closed.opening), closed, tensor([2]), 1025.0
    )
    assert u.numel() == 0 and dke.item() == 0
