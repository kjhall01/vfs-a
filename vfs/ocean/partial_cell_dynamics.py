"""Reversible pressure/transport block on split faces (instantaneous tendencies).

Velocity lives on current physical intersections. A trajectory also requires
a momentum transfer when this set changes, and a compatible rotation operator.
"""

from dataclasses import dataclass

import torch

from .partial_cells import layer_interfaces


@dataclass(frozen=True)
class PressureTransport:
    h: torch.Tensor
    temp_h: torch.Tensor
    salt_h: torch.Tensor
    velocity: torch.Tensor


def kinetic_thickness_conjugate(faces, depth, h, velocity, dual_area, rho0):
    """dK/dh for bottom-anchored layers at fixed segment velocities, in J/m.

    Each segment extends from the deeper top to the shallower bottom. An
    interface moves upwards when any layer below it thickens. At coincident
    interfaces split the derivative symmetrically between the two columns.
    """
    shape = h.shape
    interfaces = layer_interfaces(depth, h)
    tops, bottoms = interfaces[:, :-1], interfaces[:, 1:]
    l, r = faces.left, faces.right
    top_delta = tops.flatten()[l] - tops.flatten()[r]
    bottom_delta = bottoms.flatten()[l] - bottoms.flatten()[r]
    left_top = 0.5 * (1 + top_delta.sign())
    left_bottom = 0.5 * (1 - bottom_delta.sign())
    density = 0.5 * rho0 * dual_area[faces.edge] * velocity.square()
    # Accumulate at each interface, then propagate to the layers below it.
    interfaces = h.new_zeros((shape[0], shape[1] + 1))
    out = interfaces.flatten()
    for owner, top, bottom in (
        (l, left_top, left_bottom),
        (r, 1 - left_top, 1 - left_bottom),
    ):
        interface = owner + owner.div(shape[1], rounding_mode="floor")
        out = out.index_add(0, interface, density * top)
        out = out.index_add(0, interface + 1, -density * bottom)
    return out.view_as(interfaces)[:, :-1].cumsum(1)


def pressure_transport(
    faces,
    velocity,
    length,
    distance,
    area,
    rho0,
    temperature,
    salinity,
    conjugate_h,
    conjugate_t,
    conjugate_s,
):
    """Conjugates are full Hamiltonian derivatives per stored h, hCT, hSA.

    The same centered carriers multiply the thermodynamic force and inventories.
    This pairing cancels pressure work for arbitrary thermodynamic conjugates.
    """
    left, right = faces.left, faces.right
    t = 0.5 * (temperature.flatten()[left] + temperature.flatten()[right])
    s = 0.5 * (salinity.flatten()[left] + salinity.flatten()[right])
    flux = faces.volume_flux(velocity, length)
    A = area[:, None]
    jump = faces.jump
    force = -(
        jump(conjugate_h / A) + t * jump(conjugate_t / A) + s * jump(conjugate_s / A)
    )
    du = force / (rho0 * distance[faces.edge])
    return PressureTransport(
        faces.exchange(flux) / A,
        faces.exchange(flux * t) / A,
        faces.exchange(flux * s) / A,
        du,
    )
