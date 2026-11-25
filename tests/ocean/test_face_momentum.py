import pytest
import torch

from vfs.ocean.face_momentum import transfer_face_momentum
from vfs.ocean.partial_cells import WetFaces


def f(x):
    return torch.tensor(x, dtype=torch.float64)


def test_surface_motion_and_shear():
    depth, cells, area = f([10, 20]), torch.tensor([[0, 1]]), f([3])
    old, new = f([[4, 5], [15, 5]]), f([[8, 3], [17, 4]])
    u = f([2, -1])
    args = depth, old, new, cells, u, area, 2.0
    with pytest.raises(ValueError, match="donor velocity"):
        transfer_face_momentum(*args)
    result = transfer_face_momentum(*args, incoming_velocity=f([4, float("nan")]))
    torch.testing.assert_close(result.velocity, f([1.75, -1]))
    torch.testing.assert_close(result.incoming.mass, f([12, 0]))
    torch.testing.assert_close(result.incoming.momentum, f([48, 0]))
    torch.testing.assert_close(result.mixed_kinetic, f([76.5, 0]))
    reverse = transfer_face_momentum(depth, new, old, cells, result.velocity, area, 2.0)
    torch.testing.assert_close(reverse.outgoing.mass, f([12, 0]))
    torch.testing.assert_close(reverse.outgoing.momentum, f([21, 0]))
    torch.testing.assert_close(reverse.outgoing.kinetic, f([18.375, 0]))


def test_birth_and_death_at_shelf():
    depth, cells, area = f([10, 20, 0]), torch.tensor([[0, 1], [1, 2]]), f([3, 4])
    disconnected, connected = f([[10], [5], [0]]), f([[10], [12], [0]])
    start = transfer_face_momentum(
        depth,
        disconnected,
        connected,
        cells,
        f([]),
        area,
        2.0,
        incoming_velocity=f([-3]),
    )
    torch.testing.assert_close(start.velocity, f([-3]))
    torch.testing.assert_close(start.incoming.mass, f([12]))
    assert start.faces.edge.tolist() == [0]
    end = transfer_face_momentum(
        depth, connected, disconnected, cells, start.velocity, area, 2.0
    )
    torch.testing.assert_close(end.outgoing.kinetic, start.incoming.kinetic)
    torch.testing.assert_close(end.outgoing.momentum, start.incoming.momentum)
    assert not end.velocity.numel()


def test_same_extent_matches_ale_and_constant_flow():
    depth, cells, area = f([10, 16]), torch.tensor([[0, 1]]), f([7])
    h, target = f([[3, 7], [8, 8]]), f([[6, 4], [4, 12]])
    old, new = [WetFaces.from_columns(depth, x, cells) for x in (h, target)]
    u = f([2, -0.5, 1.5])
    v, change = old.remap_velocity(u, new, area, 2.0)
    result = transfer_face_momentum(depth, h, target, cells, u, area, 2.0)
    torch.testing.assert_close(result.velocity, v)
    torch.testing.assert_close(result.mixed_kinetic.sum(), -change.sum())
    shifted = target.clone()
    shifted[:, 0] += 1.0
    dst = WetFaces.from_columns(depth, shifted, cells)
    constant = transfer_face_momentum(
        depth,
        h,
        shifted,
        cells,
        torch.full_like(u, -2.0),
        area,
        2.0,
        incoming_velocity=torch.full_like(dst.opening, -2.0),
    )
    torch.testing.assert_close(constant.velocity, torch.full_like(dst.opening, -2.0))
    assert not constant.mixed_kinetic.any()


def test_independent_per_edge_budgets():
    generator = torch.Generator().manual_seed(412)

    def random(shape):
        return torch.rand(shape, generator=generator, dtype=torch.float64)

    n = 32
    h = 0.1 + random((n, 8))
    target = 0.1 + random((n, 7))
    depth = h.sum(-1)
    cells = torch.stack((torch.arange(n), torch.arange(n).roll(-1)), 1)
    old, new = [WetFaces.from_columns(depth, x, cells) for x in (h, target)]
    u, donor = random(old.opening.shape) - 0.5, random(new.opening.shape) - 0.5
    area = 20 + random((n,))
    result = transfer_face_momentum(
        depth, h, target, cells, u, area, 1025.0, incoming_velocity=donor
    )

    def total(faces, values):
        return torch.zeros_like(depth).index_add(0, faces.edge, values)

    m0 = 1025.0 * area[old.edge] * old.opening
    m1 = 1025.0 * area[new.edge] * new.opening
    for before, after, arrives, leaves in [
        (m0, m1, result.incoming.mass, result.outgoing.mass),
        (
            m0 * u,
            m1 * result.velocity,
            result.incoming.momentum,
            result.outgoing.momentum,
        ),
        (
            0.5 * m0 * u.square(),
            0.5 * m1 * result.velocity.square() + result.mixed_kinetic,
            result.incoming.kinetic,
            result.outgoing.kinetic,
        ),
    ]:
        torch.testing.assert_close(
            total(old, before - leaves),
            total(new, after - arrives),
            rtol=5e-14,
            atol=5e-11,
        )
