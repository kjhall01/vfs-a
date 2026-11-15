"""Existing bounded VFS stages plus a passive cloudy/clear amount ledger."""

from dataclasses import replace
import math
import torch
from vfs.atmosphere.bounded_transport import graph, bounded_transfer
from vfs.atmosphere.moist_cp import interface_pressure_mass


def advance_dynamics(adapter, state, dt):
    """Advance dynamics without advancing the physics clock or its memory.

    The authoritative atmospheric Euler maps and their SSP combinations are
    unchanged. Cloud fraction uses the same carrier and stage endpoints, with
    cloudy/clear amounts bounded as complementary passive constituents.
    """
    model = adapter.model
    if model.transport is None:
        raise ValueError("cloud transport currently supports bounded SSPRK3 only")
    adapter.check(state)
    area = model.mesh.cell_area[:, None]

    def mass(s):
        return area * interface_pressure_mass(s.column.interface_pressure(s.ps).diff())

    def euler(s, cloud_amount):
        end, info = model.transport.euler_with_info(s, dt)
        m, i, j, F = graph(model.dynamics, s)
        q = (cloud_amount / m).flatten()
        fractions = torch.stack((1 - q, q), -1)
        face = 0.5 * (fractions[i] + fractions[j])
        amounts, _, _ = bounded_transfer(
            m.flatten(),
            m.flatten()[:, None] * fractions,
            i,
            j,
            dt * F,
            dt * F[:, None] * face,
            target_mass=mass(end).flatten(),
        )
        return end, amounts[:, 1].reshape_as(m)

    initial = state.atmosphere
    C = mass(initial) * state.cloud_fraction
    s1, c1 = euler(initial, C)
    s1e, c1e = euler(s1, c1)
    s2, c2 = initial.combine(s1e, a=0.75, b=0.25), 0.75 * C + 0.25 * c1e
    s2e, c2e = euler(s2, c2)
    end = initial.combine(s2e, a=1 / 3, b=2 / 3)
    cloud = ((1 / 3) * C + (2 / 3) * c2e) / mass(end)
    result = replace(state, atmosphere=end, cloud_fraction=cloud)
    adapter.check(result)
    return result


def pressure_velocity(adapter, state):
    """Physical Dp/Dt diagnostic for convection, distinct from grid-relative mu.

    Interpolate relative pressure flux to the physics level, then add coordinate
    motion and horizontal pressure advection. This is a diagnostic interpolation;
    it does not alter VFS continuity or its vertical transport.
    """
    s = state.atmosphere
    d = adapter.model.dynamics
    ctx = d.kernel._transport_primitives(d.transport_state(s))
    _, geometry = adapter.mapping.columns(s, state.cloud_fraction)
    p, pf = geometry["phalf"], geometry["pfull"]
    dp = p.diff()
    B = s.column.B_interface
    derivative = (
        pf
        / dp
        * (B[1:] * torch.log(p[:, 1:] / pf) - B[:-1] * torch.log(p[:, :-1] / pf))
    )
    x = (pf - p[:, :-1]) / dp
    relative = (1 - x) * ctx["mu"][:, :-1] + x * ctx["mu"][:, 1:]
    pressure_gradient = s.mesh.cell_wind_vector(s.mesh.grad(s.ps))
    advection = (adapter.mapping.vector(s.u_e) * pressure_gradient).sum(-1)
    return relative + derivative * (ctx["dps_dt"] + advection)


def step(adapter, state, dt, *, dynamics_dt):
    """Sequential first-order dynamics/physics composition, one physics call."""
    if (
        not math.isfinite(dt)
        or not math.isfinite(dynamics_dt)
        or dt <= 0
        or dynamics_dt <= 0
        or not float(dt).is_integer()
    ):
        raise ValueError("positive dt/dynamics_dt and integer physics seconds required")
    elapsed = 0.0
    result = state
    while elapsed < dt:
        h = min(dynamics_dt, dt - elapsed)
        result = advance_dynamics(adapter, result, h)
        elapsed += h
    return adapter.apply(result, dt, omega=pressure_velocity(adapter, result))
