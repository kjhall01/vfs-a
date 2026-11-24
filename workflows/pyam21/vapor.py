"""Fund an exact-zero vapor endpoint from the same layer's cloud water."""

import torch
from pyam21.constants import CP_AIR, HLV, HLF


# kg vapor / kg mixture. Positive endpoints, however small, are left alone.
# This host-side regularization is part of the campaign source fingerprint.
VAPOR_EPSILON = 1e-12


def restore_positive_vapor(temperature, tracers, out, dt):
    """Return corrected copies and a sparse ledger, or the original objects.

    The adapter first validates the upstream endpoint/tendency agreement and
    all fields except exact-zero vapor. Withdraw liquid/ice in proportion to
    their available stores. Use the actual representable withdrawals, rather
    than adding water that a rounded subtraction failed to remove.

    Cooling follows AM2's constant-cp latent-energy convention. The adapter's
    native energy solve retains the ORIGINAL upstream energy target, so this
    internal phase transfer supplies no external energy and hides no upstream
    defect. The small AM2 cooling roundoff is reported separately.
    """
    indices = (tracers[..., 0] == 0).nonzero()
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
