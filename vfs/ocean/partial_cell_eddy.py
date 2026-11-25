"""Rotated diffusion and GM volume carriers on physical wet-face intersections."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShelfRedi:
    """Sparse gradient rows and their positive diffusion measures on wet openings."""

    owners: torch.Tensor
    weights: torch.Tensor
    measure: torch.Tensor
    slope: torch.Tensor
    shape: torch.Size

    @classmethod
    def build(
        cls,
        faces,
        depth,
        h,
        buoyancy,
        distance,
        length,
        diffusivity,
        *,
        max_slope=0.01,
        min_n2=1e-8,
    ):
        if diffusivity < 0 or max_slope <= 0 or min_n2 <= 0:
            raise ValueError("invalid Redi coefficients")
        z = (h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - 0.5 * h).flatten()
        dc = distance[faces.edge]
        left, right = faces.left, faces.right

        # One-sided endpoint secants, centered mean of the two secants elsewhere.
        # A single-layer column contributes no vertical derivative.
        def vertical(owner):
            above = owner.remainder(h.shape[1]) > 0
            following = (owner + 1).clamp_max(h.numel() - 1)
            below = ((owner + 1).remainder(h.shape[1]) != 0) & (
                h.flatten()[following] > 0
            )
            upper = torch.where(above, owner - 1, owner)
            lower = torch.where(below, following, owner)
            count = (above.long() + below.long()).clamp_min(1)
            a = above / (count * torch.where(above, z[upper] - z[owner], 1.0))
            c = below / (count * torch.where(below, z[lower] - z[owner], 1.0))
            return torch.stack((upper, owner, lower), 1), torch.stack((a, -a - c, c), 1)

        li, lw = vertical(left)
        ri, rw = vertical(right)
        b = buoyancy.flatten()
        bz = 0.5 * ((b[li] * lw).sum(1) + (b[ri] * rw).sum(1))
        stable = bz > min_n2
        zn = (z[right] - z[left]) / dc
        raw = zn - (b[right] - b[left]) / (dc * torch.where(stable, bz, 1.0))
        slope = torch.where(stable, raw.clamp(-max_slope, max_slope), 0.0)
        relative = (slope - zn)[:, None]
        owners = torch.cat((left[:, None], right[:, None], li, ri), 1)
        weights = torch.cat(
            (
                -1 / dc[:, None],
                1 / dc[:, None],
                0.5 * relative * lw,
                0.5 * relative * rw,
            ),
            1,
        )
        measure = diffusivity * length[faces.edge] * dc * faces.opening * stable
        return cls(owners, weights, measure, slope, h.shape)

    def _transpose(self, flux, weights):
        out = flux.new_zeros(self.shape.numel())
        return out.index_add(
            0, self.owners.flatten(), (flux[:, None] * weights).flatten()
        ).view(self.shape)

    def inventory_tendency(self, concentration):
        """Extensive tendency -R.T M R q. Divide by wet volume to obtain dq/dt."""
        grad = (concentration.flatten()[self.owners] * self.weights).sum(1)
        return -self._transpose(self.measure * grad, self.weights)

    def step_limit(self, volume):
        """Conservative inverse absolute-row-sum bound in seconds."""
        w = self.weights.abs()
        bound = self._transpose(self.measure * w.sum(1), w)
        return torch.where(bound > 0, volume / bound, torch.inf).amin()

    def advance_roquet(
        self, depth, h, temp_h, salt_h, eddy_h, area, energy, dt, *, carried_h=None
    ):
        """Fixed-volume Redi update with conservative local dynamic-enthalpy flux.

        cp*CT follows the CT flux exactly. Advect the layer-mean parcel
        potential through the same diffusion operator and settle the difference
        from the nonlinear endpoint in h*EKE. Reject unfunded exchanges.
        """
        if dt <= 0 or dt > self.step_limit(h * area[:, None]):
            raise ValueError(
                "Redi timestep exceeds the explicit bound or is nonpositive"
            )

        divisor = torch.where(h > 0, h, 1.0)
        measure = self.measure
        if carried_h is not None:
            # One factor per rotated diffusion row keeps all fields and the
            # transported thermodynamic potential on the same funded exchange.
            scale = torch.ones_like(measure)
            for content in (salt_h, eddy_h, *carried_h.unbind(-1)):
                q = (content / divisor).flatten()
                gradient = (q[self.owners] * self.weights).sum(1)
                changes = -dt * (measure * gradient)[:, None] * self.weights
                withdrawals = (-changes).clamp_min(0)
                demand = q.new_zeros(q.shape).index_add(
                    0, self.owners.flatten(), withdrawals.flatten()
                )
                amount = (area[:, None] * content).flatten()
                room = (amount / torch.where(demand > 0, demand, 1.0)).clamp(max=1.0)
                room = torch.where(
                    room < 1, room * (1 - 64 * torch.finfo(q.dtype).eps), room
                )
                scale = torch.minimum(
                    scale, torch.where(withdrawals > 0, room[self.owners], 1.0).amin(1)
                )
            measure = measure * scale

        def increment(q):
            grad = (q.flatten()[self.owners] * self.weights).sum(1)
            return -dt * self._transpose(measure * grad, self.weights) / area[:, None]

        return _transport_and_settle(
            depth, h, temp_h, salt_h, eddy_h, carried_h, energy, increment
        )


def _transport_and_settle(
    depth, h, temp_h, salt_h, eddy_h, carried_h, energy, increment
):
    divisor = torch.where(h > 0, h, 1.0)
    eddy = eddy_h + increment(eddy_h / divisor)
    result = _settle_exchange(depth, h, temp_h, salt_h, eddy, energy, increment)
    if carried_h is None:
        return result
    updated = torch.stack(
        [content + increment(content / divisor) for content in carried_h.unbind(-1)], -1
    )
    if not torch.isfinite(updated).all() or (updated < 0).any():
        raise ValueError("invalid transported passive or turbulent energy inventory")
    return (*result, updated)


def _settle_exchange(depth, h, temp_h, salt_h, eddy_h, energy, increment):
    divisor = torch.where(h > 0, h, 1.0)
    z = h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - h
    nodes, weights = energy._quadrature(h)
    heights = z[..., None] + h[..., None] * nodes

    def potential(t, s):
        psi, _, _, _ = energy.parcel_potential(t[..., None], s[..., None], heights)
        return (psi * weights).sum(-1)

    t, s = temp_h / divisor, salt_h / divisor
    before = potential(t, s)
    ht, hs = temp_h + increment(t), salt_h + increment(s)
    t, s = ht / divisor, hs / divisor
    if ((h > 0) & ((s < 0) | (s > 42) | (t <= -2.5) | (t >= 40))).any():
        raise ValueError("exchange outside caloric domain")
    exchange = increment(before) - h * (potential(t, s) - before)
    reservoir = eddy_h + exchange
    if (reservoir < 0).any() or not torch.isfinite(reservoir).all():
        raise ValueError("exchange exceeds local EKE budget")
    return ht, hs, reservoir


def gm_volume_carriers(faces, h, length, interface_psi):
    """Close the GM circulation using one streamfunction on each split face.

    interface_psi is the streamfunction (m²/s) at the bottom of each
    segment. Surface and seabed are impermeable. Vertical transport is
    positive down and stored at the bottom of each wet cell.
    """
    e = faces.edge
    if interface_psi.shape != e.shape:
        raise ValueError("expected one lower-interface streamfunction per segment")
    if not e.numel():
        return interface_psi, torch.zeros_like(h)
    interior = torch.cat((e[:-1] == e[1:], e.new_zeros(1, dtype=torch.bool)))
    bottom = torch.where(interior, interface_psi, 0.0)
    top = torch.cat((torch.zeros_like(bottom[:1]), bottom[:-1]))
    horizontal = length[e] * (bottom - top)
    vertical = faces.exchange(horizontal).cumsum(1)
    lower_wet = torch.cat(
        (h[:, 1:] > 0, torch.zeros_like(h[:, :1], dtype=torch.bool)), 1
    )
    return horizontal, torch.where(lower_wet, vertical, 0.0)


def advance_gm_roquet(
    faces,
    depth,
    h,
    temp_h,
    salt_h,
    eddy_h,
    area,
    energy,
    length,
    transport,
    dt,
    *,
    carried_h=None,
):
    """Upwind GM parcel exchange, including vertical return flow and carried EKE.

    transport = diffusivity times physical neutral slope on each opening.
    """
    if dt <= 0 or not torch.isfinite(transport).all():
        raise ValueError("invalid GM step or transport")
    following = torch.cat((transport[1:], transport[-1:]))
    horizontal, down = gm_volume_carriers(
        faces, h, length, 0.5 * (transport + following)
    )
    ids = torch.arange(h.numel(), device=h.device).reshape_as(h)
    interior = h[:, 1:] > 0
    left = torch.cat((faces.left, ids[:, :-1][interior]))
    right = torch.cat((faces.right, ids[:, 1:][interior]))
    amount = dt * torch.cat((horizontal, down[:, :-1][interior]))
    donor = torch.where(amount >= 0, left, right)
    outflow = h.new_zeros(h.numel()).index_add(0, donor, amount.abs())
    if (outflow.view_as(h) > h * area[:, None]).any():
        raise ValueError("GM outgoing volume exceeds available water")

    def increment(q):
        transfer = amount * q.flatten()[donor]
        delta = (
            h.new_zeros(h.numel())
            .index_add(0, left, -transfer)
            .index_add(0, right, transfer)
        )
        return delta.view_as(h) / area[:, None]

    return _transport_and_settle(
        depth, h, temp_h, salt_h, eddy_h, carried_h, energy, increment
    )
