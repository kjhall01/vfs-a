"""Physical-height transfer of edge-normal momentum across a moving wet opening.

The old and new cell thicknesses describe the same mesh and fixed seabed.
Exchange with the rest of the ocean is explicit wherever the opening changes.
"""

from dataclasses import dataclass

import torch

from .partial_cells import WetFaces, layer_interfaces


@dataclass(frozen=True)
class FaceExchange:
    """Positive mass and KE, signed normal momentum, all integrated over the step."""

    mass: torch.Tensor  # kg
    momentum: torch.Tensor  # kg m/s, in the mesh edge orientation
    kinetic: torch.Tensor  # J


@dataclass(frozen=True)
class FaceMomentum:
    faces: WetFaces
    velocity: torch.Tensor
    incoming: FaceExchange  # indexed by new segment
    outgoing: FaceExchange  # indexed by old segment
    mixed_kinetic: torch.Tensor  # J lost to subsegment shear, indexed by new segment


def _coordinates(depth, h, faces, n_edges):
    """Sorted segment bottoms per edge. Padding stays outside the physical domain."""
    z = layer_interfaces(depth, h)
    top = torch.maximum(
        z[:, :-1].flatten()[faces.left], z[:, :-1].flatten()[faces.right]
    )
    bottom = torch.minimum(
        z[:, 1:].flatten()[faces.left], z[:, 1:].flatten()[faces.right]
    )
    count = torch.bincount(faces.edge, minlength=n_edges)
    offset = count.cumsum(0) - count
    width = max(1, int(count.max())) if n_edges else 1
    padded = h.new_full((n_edges, width), torch.inf)
    slot = torch.arange(len(faces.edge), device=h.device) - offset[faces.edge]
    padded[faces.edge, slot] = bottom
    surface = h.new_full((n_edges, 1), torch.inf)
    surface[faces.edge[slot == 0], 0] = top[slot == 0]
    return surface, padded, count, offset


def transfer_face_momentum(
    depth, old_h, new_h, cells, velocity, dual_area, rho0, *, incoming_velocity=None
):
    """Project old and arriving water onto new segments using mass-weighted velocity.

    incoming_velocity (m/s) is supplied per NEW segment wherever it gains
    physically uncovered water. The caller owns its donor momentum and KE.
    Removed water returns its old velocity and energy to the caller's transport
    ledger. No external impulse or pressure work is included in this operation.
    Splits preserve velocity exactly in real arithmetic. Merges report the
    unresolved velocity variance as a nonnegative energy available to TKE.
    """
    old = WetFaces.from_columns(depth, old_h, cells)
    new = WetFaces.from_columns(depth, new_h, cells)
    return _transfer_between(
        depth, old_h, new_h, old, new, velocity, dual_area, rho0, incoming_velocity
    )


def _transfer_between(
    depth, old_h, new_h, old, new, velocity, dual_area, rho0, incoming_velocity
):
    if (
        dual_area.shape != (old.n_edges,)
        or velocity.shape != old.opening.shape
        or any(
            x.dtype != depth.dtype or x.device != depth.device
            for x in (dual_area, velocity)
        )
        or not torch.isfinite(dual_area).all()
        or (dual_area <= 0).any()
        or not torch.isfinite(velocity).all()
        or not 0 < rho0 < float("inf")
    ):
        raise ValueError("invalid edge velocity, dual area or density")
    before = _coordinates(depth, old_h, old, old.n_edges)
    after = _coordinates(depth, new_h, new, old.n_edges)
    # One sorted union per mesh edge, linear storage in vertical levels.
    bounds = torch.cat((before[0], before[1], after[0], after[1]), 1).sort(1).values
    start, end = bounds[:, :-1].contiguous(), bounds[:, 1:]
    edge, slot = torch.where((end > start) & torch.isfinite(end))

    def locate(coordinates):
        surface, bottom, count, offset = coordinates
        layer = torch.searchsorted(bottom, start, right=True)[edge, slot]
        present = (start[edge, slot] >= surface[edge, 0]) & (layer < count[edge])
        return offset[edge] + layer, present

    src, was_wet = locate(before)
    dst, now_wet = locate(after)
    shared = was_wet & now_wet
    gained = ~was_wet & now_wet
    lost = was_wet & ~now_wet
    mass = rho0 * dual_area[edge] * (end[edge, slot] - start[edge, slot])
    incoming_mass = torch.zeros_like(new.opening).index_add(
        0, dst[gained], mass[gained]
    )
    outgoing_mass = torch.zeros_like(old.opening).index_add(0, src[lost], mass[lost])
    if incoming_velocity is None:
        if gained.any():
            raise ValueError("newly opened water requires its donor velocity")
        incoming_velocity = torch.zeros_like(new.opening)
    if (
        incoming_velocity.shape != new.opening.shape
        or incoming_velocity.dtype != depth.dtype
        or incoming_velocity.device != depth.device
        or not torch.isfinite(incoming_velocity[incoming_mass > 0]).all()
    ):
        raise ValueError("expected finite donor velocity on each newly exposed segment")
    donor = torch.where(incoming_mass > 0, incoming_velocity, 0.0)
    incoming = FaceExchange(
        incoming_mass, incoming_mass * donor, 0.5 * incoming_mass * donor.square()
    )
    outgoing = FaceExchange(
        outgoing_mass, outgoing_mass * velocity, 0.5 * outgoing_mass * velocity.square()
    )
    dest = dst[shared]
    retained_mass, retained_u = mass[shared], velocity[src[shared]]
    momentum = incoming.momentum.index_add(0, dest, retained_mass * retained_u)
    mapped = momentum / (rho0 * dual_area[new.edge] * new.opening)
    mixing = (0.5 * incoming_mass * (donor - mapped).square()).index_add(
        0, dest, 0.5 * retained_mass * (retained_u - mapped[dest]).square()
    )
    return FaceMomentum(new, mapped, incoming, outgoing, mixing)
