"""Material velocity of water exposed by a split-face continuity update.

Vectors live in the tangent plane of their own cell. The cell transport
ledger uses primal water masses. Swept edge masses in FaceMomentum are
changes of the dual kinetic metric and are accounted for separately.
"""

from dataclasses import dataclass

import torch

from .face_momentum import FaceMomentum, _transfer_between
from .partial_cells import WetFaces, layer_interfaces


def _parallel(v, a, b):
    """Transport tangent vectors from unit a to unit b along the short arc."""
    return v - ((v * b).sum(-1) / (1 + (a * b).sum(-1)))[..., None] * (a + b)


def _geometry(mesh, faces, levels):
    left = mesh.cell_xyz[faces.left // levels]
    right = mesh.cell_xyz[faces.right // levels]
    midpoint = torch.nn.functional.normalize(left + right, dim=-1)
    return left, right, midpoint


def reconstruct_cell_velocity(mesh, h, faces, velocity):
    """Initialize cell vectors by an opening-area-weighted fit of normal samples.

    Apply once when starting from a C-grid state. A rank-one fit sets the
    unobserved transverse component to zero. A layer without samples starts
    at rest; callers with known cell velocities can pass them directly.
    """
    if velocity.shape != faces.opening.shape or not torch.isfinite(velocity).all():
        raise ValueError("invalid normal velocity")
    left, right, midpoint = _geometry(mesh, faces, h.shape[1])
    normal = mesh.edge_normal[faces.edge]
    axis = torch.eye(3, dtype=h.dtype, device=h.device)[mesh.cell_xyz.abs().argmin(1)]
    first = torch.nn.functional.normalize(
        torch.linalg.cross(mesh.cell_xyz, axis), dim=1
    )
    basis = torch.stack((first, torch.linalg.cross(mesh.cell_xyz, first)), dim=1)
    matrix = h.new_zeros((h.numel(), 2, 2))
    rhs = h.new_zeros((h.numel(), 2))
    weight = faces.opening * mesh.dv_edge[faces.edge]
    for owner, center in ((faces.left, left), (faces.right, right)):
        n = (
            basis[owner // h.shape[1]] * _parallel(normal, midpoint, center)[:, None, :]
        ).sum(-1)
        matrix = matrix.index_add(
            0, owner, weight[:, None, None] * n[:, :, None] * n[:, None, :]
        )
        rhs = rhs.index_add(0, owner, (weight * velocity)[:, None] * n)
    # Rank changes at coasts and isolated bottom layers.
    inverse = torch.linalg.pinv(
        matrix, hermitian=True, rtol=64 * torch.finfo(h.dtype).eps
    )
    coefficients = (inverse @ rhs[:, :, None]).reshape(*h.shape, 2)
    return (coefficients[..., None] * basis[:, None, :, :]).sum(-2)


@dataclass(frozen=True)
class ContinuityMomentum:
    h: torch.Tensor
    cell_velocity: torch.Tensor
    cell_mixing_energy: torch.Tensor  # J per receiving cell-layer
    transferred_volume: torch.Tensor  # signed m³ per OLD opening
    face: FaceMomentum
    donor_velocity: torch.Tensor  # normal component on newly exposed intervals
    donor_mixing_energy: torch.Tensor  # J of normal variance before face projection


def advance_continuity_momentum(mesh, depth, h, velocity, cell_velocity, dt, rho0):
    """Transport water and its velocity with one upwind forward continuity step.

    The same signed volume debits the donor and credits the receiver. At a new
    opening, take the normal component from whichever side filled the interval.
    Where both sides filled it, use their dual-side area fractions.
    """
    if not 0 < dt < float("inf") or not 0 < rho0 < float("inf"):
        raise ValueError("positive finite step and density required")
    cells = mesh.cellsOnEdge
    old = WetFaces.from_columns(depth, h, cells)
    if velocity.shape != old.opening.shape or cell_velocity.shape != (*h.shape, 3):
        raise ValueError("velocities must match wet faces and cell layers")
    if any(
        x.dtype != h.dtype or x.device != h.device or not torch.isfinite(x).all()
        for x in (velocity, cell_velocity)
    ):
        raise ValueError("velocities must be finite and match layer dtype/device")
    radial = (cell_velocity * mesh.cell_xyz[:, None, :]).sum(-1)
    if (
        radial.abs() > 256 * torch.finfo(h.dtype).eps * (1 + cell_velocity.norm(dim=-1))
    ).any():
        raise ValueError("cell velocity must lie in the local tangent plane")
    area = mesh.cell_area[:, None]
    signed = dt * old.volume_flux(velocity, mesh.dv_edge)
    donor = torch.where(signed >= 0, old.left, old.right)
    receiver = torch.where(signed >= 0, old.right, old.left)
    volume = signed.abs()
    initial = (area * h).flatten()
    remaining = initial.index_add(0, donor, -volume)
    if (remaining < 0).any():
        raise ValueError("outgoing volume exceeds available donor water")
    final = remaining.index_add(0, receiver, volume)
    # Retain h exactly in layers with no net volume flux.
    new_h = h + old.exchange(signed) / area
    new = WetFaces.from_columns(depth, new_h, cells)
    v = cell_velocity.reshape(-1, 3)
    xyz = mesh.cell_xyz
    arriving = _parallel(
        v[donor], xyz[donor // h.shape[1]], xyz[receiver // h.shape[1]]
    )
    momentum = (remaining[:, None] * v).index_add(
        0, receiver, volume[:, None] * arriving
    )
    mixed = momentum / torch.where(final > 0, final, 1.0)[:, None]
    loss = (
        0.5
        * rho0
        * (remaining * (v - mixed).square().sum(-1)).index_add(
            0, receiver, volume * (arriving - mixed[receiver]).square().sum(-1)
        )
    )

    # Positive-down physical depths: water above an old surface was absent.
    zi = layer_interfaces(depth, new_h)
    top = torch.maximum(zi[:, :-1].flatten()[new.left], zi[:, :-1].flatten()[new.right])
    surface = layer_interfaces(depth, h)[:, 0]
    formerly_empty = (surface[cells[new.edge]] - top[:, None]).clamp_min(0)
    formerly_empty = torch.minimum(formerly_empty, new.opening[:, None])
    both = formerly_empty.amin(1, keepdim=True)
    sides = mesh.edge_side_area[new.edge]
    weights = formerly_empty - both + both * (sides / sides.sum(1, keepdim=True))
    gained = weights.sum(1)
    fraction = weights / torch.where(gained > 0, gained, 1.0)[:, None]
    left, right, midpoint = _geometry(mesh, new, h.shape[1])
    normal = mesh.edge_normal[new.edge]
    components = torch.stack(
        (
            (mixed[new.left] * _parallel(normal, midpoint, left)).sum(-1),
            (mixed[new.right] * _parallel(normal, midpoint, right)).sum(-1),
        ),
        1,
    )
    incoming = (fraction * components).sum(1)
    dual = mesh.dc_edge * mesh.dv_edge
    face = _transfer_between(depth, h, new_h, old, new, velocity, dual, rho0, incoming)
    variance = (
        0.5
        * face.incoming.mass
        * (fraction * (components - incoming[:, None]).square()).sum(1)
    )
    return ContinuityMomentum(
        new_h,
        mixed.view_as(cell_velocity),
        loss.view_as(h),
        signed,
        face,
        incoming,
        variance,
    )


@dataclass(frozen=True)
class MomentumSettlement:
    tke_h: torch.Tensor
    transported_energy: torch.Tensor  # net influx of resolved KE + TKE in J
    conversion: torch.Tensor  # J into TKE (negative means a resolved KE cost)


def settle_transport_energy(
    mesh, depth, old_h, old_velocity, old_cell_velocity, old_tke_h, step, rho0
):
    """Local face-KE/TKE budget for the isolated, unforced transport step.

    Count resolved energy with the split-face Hamiltonian. Move kinetic energy
    and TKE with the same upwind water as momentum. Settle the local difference
    into h*TKE, including the cost of changing the face metric. This is a
    finite transport closure, separate from pressure work. The cell-vector
    KE and the mixing diagnostics are never counted as additional reservoirs.
    """
    if (
        old_tke_h.shape != old_h.shape
        or old_tke_h.dtype != old_h.dtype
        or old_tke_h.device != old_h.device
        or not torch.isfinite(old_tke_h).all()
        or (old_tke_h < 0).any()
        or (old_tke_h[old_h == 0] != 0).any()
    ):
        raise ValueError("invalid initial h*TKE")
    old = WetFaces.from_columns(depth, old_h, mesh.cellsOnEdge)
    dual = mesh.dc_edge * mesh.dv_edge

    def cell_ke(faces, v):
        # Same equal split between neighboring columns as the Roquet ALE ledger.
        half = 0.5 * faces.kinetic_energy(v, dual, rho0)
        return (
            half.new_zeros(old_h.numel())
            .index_add(0, faces.left, half)
            .index_add(0, faces.right, half)
            .reshape_as(old_h)
        )

    signed = step.transferred_volume
    donor = torch.where(signed >= 0, old.left, old.right)
    tke = old_tke_h / torch.where(old_h > 0, old_h, 1.0)
    resolved = 0.5 * old_cell_velocity.square().sum(-1)
    ke_flux = old.exchange(rho0 * signed * resolved.flatten()[donor])
    tke_flux = old.exchange(rho0 * signed * tke.flatten()[donor])
    conversion = ke_flux - (
        cell_ke(step.face.faces, step.face.velocity) - cell_ke(old, old_velocity)
    )
    mass_area = rho0 * mesh.cell_area[:, None]
    updated = old_tke_h + (tke_flux + conversion) / mass_area
    if (
        not torch.isfinite(updated).all()
        or (updated < 0).any()
        or (updated[step.h == 0] != 0).any()
    ):
        raise ValueError("local TKE cannot fund the projected face momentum step")
    return MomentumSettlement(updated, ke_flux + tke_flux, conversion)


def transport_with_budget(mesh, depth, h, velocity, cell_velocity, tke_h, dt, rho0):
    """Return the accepted continuity step and local face-energy settlement together.

    Carry step.cell_velocity along with step.h, step.face.velocity and
    settlement.tke_h to the next unforced transport step. Inputs are immutable
    if either the donor-volume or the energy-budget check fails.
    """
    step = advance_continuity_momentum(
        mesh, depth, h, velocity, cell_velocity, dt, rho0
    )
    settlement = settle_transport_energy(
        mesh, depth, h, velocity, cell_velocity, tke_h, step, rho0
    )
    return step, settlement
