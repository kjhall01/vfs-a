"""Conserve water while restoring positive vapor in a column physics endpoint."""

import torch
from pyam21.constants import CP_AIR, HLV, HLF


# kg vapor / kg mixture. Positive endpoints, however small, are left alone.
# This host-side regularization is part of the campaign source fingerprint.
VAPOR_EPSILON = 1e-12


def restore_positive_vapor(temperature, tracers, out, dt, *, dp=None):
    """Return corrected copies and a sparse ledger, or the original objects.

    The adapter first validates the upstream endpoint/tendency agreement and
    all fields except vapor positivity. Withdraw liquid/ice in proportion to
    their available stores. Use the actual representable withdrawals, rather
    than adding water that a rounded subtraction failed to remove.

    Cooling follows AM2's constant-cp latent-energy convention. The adapter's
    native energy solve retains the ORIGINAL upstream energy target, so this
    internal phase transfer supplies no external energy and hides no upstream
    defect. The small AM2 cooling roundoff is reported separately.
    """
    indices = (tracers[..., 0] <= 0).nonzero()
    if indices.numel() == 0:
        return temperature, tracers, out, None
    cell, level = indices.unbind(-1)
    old_q = tracers[cell, level]
    old_T = temperature[cell, level]
    cloud = old_q[:, 1:3]
    available = cloud.clamp_min(0).sum(-1)
    # In float32, the requested epsilon can be smaller than a cloud-store ulp.
    requested = torch.maximum(
        available.new_full(available.shape, VAPOR_EPSILON),
        8 * torch.finfo(available.dtype).eps * available,
    )
    if dp is not None and ((old_q[:, 0] < 0) | (available < requested)).any():
        return redistribute_column_vapor(temperature, tracers, out, dt, dp, indices)
    if (old_q[:, 0] < 0).any():
        raise ValueError("negative-vapor redistribution requires layer pressure thickness")
    if (available < requested).any():
        bad = int((available < requested).nonzero()[0, 0])
        raise ValueError(
            "zero-vapor correction lacks local condensate"
            f"; cell={int(cell[bad])} level={int(level[bad])}"
            f" available={float(available[bad]):.12g}"
            f" required={float(requested[bad]):.12g} kg/kg"
        )
    remaining = torch.where(
        cloud > 0, cloud * (1 - requested / available)[:, None], cloud
    )
    withdrawn = cloud - remaining
    vapor = withdrawn.sum(-1)
    latent = HLV * vapor + HLF * withdrawn[:, 1]
    new_T = old_T - latent / CP_AIR
    if (vapor <= 0).any() or (new_T <= 0).any() or not torch.isfinite(new_T).all():
        raise ValueError("zero-vapor phase correction cannot produce an admissible state")
    new_q = torch.cat((vapor[:, None], remaining, old_q[:, 3:]), -1)
    delta_T, delta_q = new_T - old_T, new_q - old_q
    T, q = temperature.clone(), tracers.clone()
    T[cell, level], q[cell, level] = new_T, new_q
    tdt, rdt = out["tdt"].clone(), out["rdt"].clone()
    tdt[cell, level] = tdt[cell, level] + delta_T / dt
    rdt[cell, level] = rdt[cell, level] + delta_q / dt
    corrected = dict(out, endpoint_temperature=T, endpoint_tracers=q, tdt=tdt, rdt=rdt)
    report = dict(
        indices=indices,
        requested_vapor=requested,
        vapor_added=vapor,
        liquid_removed=withdrawn[:, 0],
        ice_removed=withdrawn[:, 1],
        temperature_change=delta_T,
        water_residual=delta_q[:, :3].sum(-1),
        am2_energy_residual=CP_AIR * delta_T + latent,
        # Retain the raw operands for these rows; all other rows are unchanged.
        upstream_temperature=old_T,
        upstream_tracers=old_q,
        upstream_tdt=out["tdt"][cell, level],
        upstream_rdt=out["rdt"][cell, level],
    )
    return T, q, corrected, report


def redistribute_column_vapor(temperature, tracers, out, dt, dp, recipients):
    """Fund depleted layers from local condensate, then nearest column vapor.

    Transfers use pressure-weighted amounts. Only originally positive layers
    may donate vapor. Each donor retains a positive reserve. All edits are
    transactional and the native solver still receives the raw energy target.
    """
    q = tracers.clone()
    # The correction is rare. Visit only affected columns and preserve order
    # when several layers draw from the same finite donor.
    for cell, level in recipients.tolist():
        row = q[cell, level]
        cloud = row[1:3].clamp_min(0)
        available = cloud.sum()
        floor = torch.maximum(row.new_tensor(VAPOR_EPSILON),
                              8 * torch.finfo(row.dtype).eps * available)
        need = floor - row[0]
        if available > 0:
            fraction = (need / available).clamp(max=1)
            remaining = torch.where(row[1:3] > 0, cloud * (1 - fraction), row[1:3])
            removed = (row[1:3] - remaining).sum()
            q[cell, level, 1:3] = remaining
            q[cell, level, 0] = row[0] + removed
        if row[0] < floor:
            deficit = (floor - row[0]) * dp[cell, level]
            for donor in sorted(range(q.shape[1]), key=lambda k: abs(k - level)):
                if tracers[cell, donor, 0] <= VAPOR_EPSILON:
                    continue
                original = q[cell, donor, 0].clone()
                capacity = (original - VAPOR_EPSILON).clamp_min(0) * dp[cell, donor]
                if capacity <= 0:
                    continue
                take = torch.minimum(deficit, capacity)
                remaining = (original - take / dp[cell, donor]).clamp_min(VAPOR_EPSILON)
                # If a rounded debit underpays, transfer one more donor ulp.
                # Credit its actual mass to the recipient, including the excess.
                if (original - remaining) * dp[cell, donor] < take and remaining > VAPOR_EPSILON:
                    lower = torch.nextafter(remaining.detach(), torch.zeros_like(remaining))
                    remaining = (remaining + (lower - remaining.detach())).clamp_min(VAPOR_EPSILON)
                q[cell, donor, 0] = remaining
                deficit = deficit - (original - remaining) * dp[cell, donor]
                if deficit <= 0:
                    break
            # An unrepresentable remainder is acceptable only at the scale of
            # the recipient's positive reserve. Never fill it from nowhere.
            q[cell, level, 0] = floor - deficit / dp[cell, level]
        if row[0] <= 0:
            raise ValueError(f"insufficient column water to restore vapor; cell={cell} level={level}")

    indices = (q != tracers).any(-1).nonzero()
    c, k = indices.unbind(-1)
    change = q[c, k] - tracers[c, k]
    cells = recipients[:, 0].unique()
    water_change = (q[cells, :, :3] - tracers[cells, :, :3]).sum(-1) * dp[cells]
    residual = water_change.sum(-1)
    scale = (dp[cells, :, None] * tracers[cells, :, :3].abs()).sum((1, 2))
    if (residual.abs() > 32 * torch.finfo(q.dtype).eps * scale).any():
        raise ValueError("column vapor correction failed its water budget")
    latent = HLV * change[:, 0] - HLF * change[:, 2]
    T = temperature.clone()
    T[c, k] = temperature[c, k] - latent / CP_AIR
    if (T[c, k] <= 0).any() or not torch.isfinite(T[c, k]).all() or (q[c, k, :3].sum(-1) >= 1).any():
        raise ValueError("column vapor correction exhausted thermal or dry-air capacity")
    dT = T[c, k] - temperature[c, k]
    rdt, tdt = out["rdt"].clone(), out["tdt"].clone()
    rdt[c, k] += change / dt
    tdt[c, k] += dT / dt
    corrected = dict(out, endpoint_temperature=T, endpoint_tracers=q, tdt=tdt, rdt=rdt)
    return T, q, corrected, dict(
        indices=indices, recipient_count=len(recipients),
        column_indices=cells, column_water_residual_Pa=residual,
        vapor_added=change[:, 0].clamp_min(0), vapor_donated=(-change[:, 0]).clamp_min(0),
        liquid_removed=-change[:, 1], ice_removed=-change[:, 2], temperature_change=dT,
        layer_water_transfer=change[:, :3].sum(-1), am2_energy_residual=CP_AIR * dT + latent,
        upstream_temperature=temperature[c, k], upstream_tracers=tracers[c, k],
        upstream_tdt=out["tdt"][c, k], upstream_rdt=out["rdt"][c, k],
    )


def _withdraw(store, request, reserve):
    """Debit the representable amount and keep each donor at or above its reserve."""
    remaining = torch.maximum(store - request, reserve)
    # A trace request can round away against a much larger donor. Credit the
    # recipient with the actual debit, including the extra ulp in this case.
    short = (store - remaining < request) & (remaining > reserve)
    lower = torch.nextafter(remaining.detach(), torch.zeros_like(remaining))
    remaining = torch.where(short, torch.maximum(remaining + (lower - remaining.detach()), reserve), remaining)
    remaining = torch.where((request > 0) & (store > reserve), remaining, store)
    return remaining, store - remaining


def fund_interface_vapor(water, mass, floor, gravity):
    """Pay the transport vapor floor on native CP nodes after the physics map.

    Water contents and CP masses are pressure amounts (Pa). First evaporate
    liquid, then sublime ice at the recipient. If needed, move vapor from the
    nearest interfaces in the same column, keeping donors above the floor.
    The caller settles the original native layer energy targets for this mixture.
    """
    fraction = water[..., 0] / mass
    indices = (fraction < floor).nonzero()
    if not indices.numel():
        return water, None
    cell, interface = indices.unbind(-1)
    funded = water.clone()
    rows = water[cell, interface].clone()
    # A few relative ulps cover mass normalization and the transport graph's
    # multiply/divide by area. No absolute epsilon or external water source.
    bound = mass * floor
    reserve = bound * (1 + 8 * torch.finfo(mass.dtype).eps)
    target = reserve[cell, interface]
    for species in (1, 2):
        stock = water[cell, interface, species]
        remainder, debit = _withdraw(stock, (target - rows[:, 0]).clamp_min(0), torch.zeros_like(stock))
        rows[:, species] = remainder
        rows[:, 0] = rows[:, 0] + debit
    funded[cell, interface] = rows
    # Only nodes whose local condensate was insufficient enter this path.
    for c, k in indices[rows[:, 0] < target].tolist():
        for donor in sorted(range(mass.shape[1]), key=lambda j: abs(k - j)):
            if donor == k:
                continue
            stock = funded[c, donor, 0].clone()
            remaining, debit = _withdraw(stock, reserve[c, k] - funded[c, k, 0], reserve[c, donor])
            funded[c, donor, 0] = remaining
            funded[c, k, 0] = funded[c, k, 0] + debit
            if funded[c, k, 0] >= reserve[c, k]:
                break
        if funded[c, k, 0] / mass[c, k] < floor:
            raise ValueError(f"cannot fund native vapor floor from column water; cell={c} interface={k} "
                             f"vapor={float(fraction[c, k]):.12g} floor={floor:.12g} kg/kg")
    touched = (funded != water).any(-1).nonzero()
    c, k = touched.unbind(-1)
    return funded, dict(indices=indices, minimum_before=fraction[cell, interface].min().detach(),
                        floor=floor, vapor_added_kg_m2=(funded[cell, interface, 0] - water[cell, interface, 0]) / gravity,
                        changed_indices=touched, species_change_Pa=funded[c, k] - water[c, k])
