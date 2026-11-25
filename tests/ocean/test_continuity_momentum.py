from types import SimpleNamespace
import pytest
import torch

from vfs.ocean.continuity_momentum import (
    advance_continuity_momentum,
    reconstruct_cell_velocity,
)
from vfs.ocean.partial_cells import WetFaces


def f(x):
    return torch.tensor(x, dtype=torch.float64)


def channel(n, cells):
    # Great-circle channel, with known analytic parallel transport of eastward flow.
    angle = torch.arange(n, dtype=torch.float64) * 0.31

    def radial(a):
        return torch.stack((a.cos(), a.sin(), torch.zeros_like(a)), 1)

    def east(a):
        return torch.stack((-a.sin(), a.cos(), torch.zeros_like(a)), 1)

    cells = torch.tensor(cells)
    a, b = angle[cells].T
    mesh = SimpleNamespace(
        cellsOnEdge=cells,
        cell_xyz=radial(angle),
        edge_normal=east((a + b) / 2) * (b - a).sign()[:, None],
        dc_edge=torch.ones_like(a),
        dv_edge=torch.ones_like(a),
        cell_area=torch.ones_like(angle),
        edge_side_area=torch.ones_like(cells, dtype=angle.dtype),
    )
    return mesh, east(angle)[:, None, :]


@pytest.mark.parametrize("reverse", [False, True])
def test_open_face_uses_water_from_the_rising_side(reverse):
    mesh, east = channel(3, [[0, 1], [2, 1] if reverse else [1, 2]])
    depth, h = f([5, 5, 2]), f([[5], [2], [2]])
    flow = f([1])
    u = east * f([1, 0, 9])[:, None, None]
    step = advance_continuity_momentum(mesh, depth, h, flow, u, 1.0, 1.0)
    torch.testing.assert_close(step.h, f([[3], [4], [2]]))
    torch.testing.assert_close(step.cell_velocity, east * f([1, 0.5, 9])[:, None, None])
    idx = step.face.faces.edge == 1
    torch.testing.assert_close(step.face.incoming.mass[idx], f([1]))
    torch.testing.assert_close(step.face.velocity[idx], f([-0.5 if reverse else 0.5]))
    torch.testing.assert_close(step.cell_mixing_energy[:, 0], f([0, 0.5, 0]))
    # The receiving side was accelerated by the actual upwind donor.
    # Halving the water flux changes the mixed speed even with the same initial vectors.
    shorter = advance_continuity_momentum(mesh, depth, h, flow, u, 0.75, 1.0)
    torch.testing.assert_close(
        shorter.donor_velocity[idx], f([(-1 if reverse else 1) * 1.5 / 3.5])
    )


def test_both_sides_wet_the_opening():
    mesh, east = channel(4, [[0, 1], [1, 2], [3, 2]])
    mesh.edge_side_area[1] = f([1, 3])
    h = f([[5], [2], [2], [5]])
    step = advance_continuity_momentum(
        mesh,
        f([5] * 4),
        h,
        f([0.5, 0, 0.5]),
        east * f([3, 0, 0, -6])[:, None, None],
        1.0,
        2.0,
    )
    # Both recipient layers gain one unit, with post-arrival velocities +1 and -2.
    index = step.face.faces.edge == 1
    torch.testing.assert_close(step.donor_velocity[index], f([-1.25]))
    torch.testing.assert_close(step.face.velocity[index], f([-1.25 / 3]))
    torch.testing.assert_close(step.donor_mixing_energy[index], f([1.6875]))
    start = f([3, 0, 0, -6])[:, None]
    k0 = (h * start.square()).sum()
    k1 = (step.h * step.cell_velocity.square().sum(-1)).sum()
    torch.testing.assert_close(k0, k1 + step.cell_mixing_energy.sum())


def test_initial_fit_and_continuity_cfl():
    mesh, east = channel(3, [[0, 1], [1, 2]])
    h = f([[2], [2], [2]])
    depth = h[:, 0]
    faces = WetFaces.from_columns(depth, h, mesh.cellsOnEdge)
    u = reconstruct_cell_velocity(mesh, h, faces, f([2, 2]))
    torch.testing.assert_close(u, 2 * east)
    with pytest.raises(ValueError, match="outgoing volume"):
        advance_continuity_momentum(mesh, depth, h, f([2, 2]), u, 1.0, 1.0)


def test_face_hamiltonian_local_settlement_and_second_step():
    from vfs.ocean.continuity_momentum import settle_transport_energy

    mesh, east = channel(3, [[0, 1], [1, 2]])
    depth, h = f([5, 5, 2]), f([[5], [2], [2]])
    v = east * f([1, 0, 9])[:, None, None]
    u = f([1])
    tke_h = h.clone()
    # A second update must use the vectors carried through the first birth.
    for dt in (1.0, 0.05):
        step = advance_continuity_momentum(mesh, depth, h, u, v, dt, 1.0)
        closed = settle_transport_energy(mesh, depth, h, u, v, tke_h, step, 1.0)

        def local_energy(h, u, htke):
            face = WetFaces.from_columns(depth, h, mesh.cellsOnEdge)
            k = 0.25 * face.opening * u.square()
            return htke.flatten().index_add(0, face.left, k).index_add(0, face.right, k)

        before = local_energy(h, u, tke_h)
        after = local_energy(step.h, step.face.velocity, closed.tke_h)
        torch.testing.assert_close(after - before, closed.transported_energy.flatten())
        torch.testing.assert_close(after.sum(), before.sum())
        h, v, u, tke_h = step.h, step.cell_velocity, step.face.velocity, closed.tke_h


def test_rejects_unfunded_face_work_without_mutating_reservoir():
    from vfs.ocean.continuity_momentum import settle_transport_energy

    mesh, east = channel(3, [[0, 1], [1, 2]])
    h, depth = f([[5], [2], [2]]), f([5, 5, 2])
    v = east * f([1, 0, 0])[:, None, None]
    u = f([1])
    step = advance_continuity_momentum(mesh, depth, h, u, v, 1.0, 1.0)
    reserve = torch.zeros_like(h)
    with pytest.raises(ValueError, match="local TKE"):
        settle_transport_energy(mesh, depth, h, u, v, reserve, step, 1.0)
    assert not reserve.any()


def test_deep_inflow_lifts_resident_surface_water():
    # A supplies only B's lower layer. Its mass lifts B's existing upper
    # layer into a new opening against C. The exposed water keeps B's upper
    # velocity, even though the incoming deep parcel moves much faster.
    mesh, east = channel(3, [[0, 1], [1, 2]])
    depth = f([10, 8, 3])
    h = f([[5, 0], [2, 4], [2, 0]])
    faces = WetFaces.from_columns(depth, h, mesh.cellsOnEdge)
    flow = torch.where(faces.edge == 0, 0.5, 0.0).to(h)
    u = east * f([[9, 0], [2, 1], [6, 0]])[:, :, None]
    step = advance_continuity_momentum(mesh, depth, h, flow, u, 1.0, 1.0)
    receiving = (step.face.incoming.mass > 0) & (step.face.faces.edge == 1)
    assert receiving.sum() == 1
    torch.testing.assert_close(step.face.incoming.mass[receiving], f([1]))
    torch.testing.assert_close(step.donor_velocity[receiving], f([2]))
    torch.testing.assert_close(step.h[1], f([2, 5.5]))
