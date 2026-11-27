"""Ocean momentum on the instantaneous shared opening between partial columns."""

import torch

from vfs.geometry.mesh.rotation import cell_to_dual
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.partial_cells import layer_interfaces


def shelf_geometry(s, domain, h=None):
    mesh = s.mesh
    z = layer_interfaces(domain.depth[:, 0], s.h if h is None else h)
    left, right = mesh.cellsOnEdge.T
    zl, zr = z[left, :1], z[right, :1]
    top = torch.maximum(zl, zr)
    bed = torch.minimum(domain.depth[left], domain.depth[right])
    middle = mesh.cell_to_edge(z)[:, 1:-1]
    interior = domain.active_edge_layers[:, 1:]
    inner = torch.where(interior, middle.clamp(min=top, max=bed), bed)
    bounds = torch.cat((top, inner, bed), 1)
    opening = torch.where(
        domain.active_edge_layers, bounds.diff(dim=1).clamp_min(0), 0.0
    )
    return opening, zl, zr, middle, top, bed, interior


def wet_ocean_peak(model, state):
    """Largest absolute native velocity with an actual water opening, and its location."""
    if state.column.layer_reference is not None and model.ocean.domain is not None:
        wet = shelf_geometry(state, model.ocean.domain)[0] > 0
    else:
        wet = model.energy.base.mechanical.metric(state) > 0
    speeds = torch.where(wet, state.u_e.abs(), -1.0).flatten()
    value, index = speeds.max(0)
    speed = float(value)
    if speed < 0:
        return 0.0, None, None
    edge, layer = divmod(int(index), state.u_e.shape[1])
    return speed, edge, layer


class OpeningEnergy(ConstantDensityEnergy):
    """KE = sum(rho * dc * dv * open_thickness * u**2 / 2) on native slots."""

    def geometry(self, s, h=None):
        return shelf_geometry(s, self.domain, h)

    def metric(self, s):
        return (
            self.rho0 * (s.mesh.dc_edge * s.mesh.dv_edge)[:, None] * self.geometry(s)[0]
        )

    def kinetic_thickness_gradient(self, s):
        opening, zl, zr, middle, top, bed, interior = self.geometry(s)
        mesh = s.mesh
        density = (
            0.5 * self.rho0 * (mesh.dc_edge * mesh.dv_edge)[:, None] * s.u_e.square()
        )
        density = torch.where(opening > 0, density, 0.0)
        # At an internal interface, moving its depth transfers thickness
        # from one velocity slot to the other. At the fixed bed it moves none.
        difference = density[:, :-1] - density[:, 1:]
        above_top = (1 + (middle - top).sign()) * 0.5
        above_bed = (1 + (bed - torch.maximum(middle, top)).sign()) * 0.5
        mid_derivative = torch.where(interior, difference * above_top * above_bed, 0.0)
        top_derivative = -density[:, :1] + torch.where(
            interior, difference * (1 - above_top) * above_bed, 0.0
        ).sum(-1, keepdim=True)
        cell_mid = self.interpolation_transpose(mesh, mid_derivative)
        left, right = mesh.cellsOnEdge.T
        left_share = (1 + (zl - zr).sign()) * 0.5
        cell_top = torch.zeros_like(s.h[:, :1]).index_add(
            0, left, top_derivative * left_share
        )
        cell_top = cell_top.index_add(0, right, top_derivative * (1 - left_share))
        # z_j = bed - sum_{k>=j} h_k. The padded entries are fixed at zero.
        result = -torch.cat((cell_top, cell_mid), 1).cumsum(1)
        return torch.where(s.column.active_layers, result, 0.0)

    def velocity(self, state):
        return torch.where(self.edge_thickness(state, state.h) > 0, state.u_e, 0.0)

    def edge_thickness(self, state, h):
        return self.geometry(state, h)[0]

    def column_kinetic_energy(self, state):
        edge_ke = 0.5 * self.metric(state) * state.u_e.square()
        return self.interpolation_transpose(state.mesh, edge_ke).sum(-1)


class OpeningRotation(ConstantDensityDynamics):
    def rotation_acceleration(self, s):
        """Use the same skew PV operator, with the open transport in each slot."""
        mesh = s.mesh
        opening = self.energy.geometry(s)[0]
        wet = opening > 0
        u = torch.where(wet, s.u_e, 0.0)
        dual = cell_to_dual(mesh, self.domain.apply_cell(s.h))
        pv = (
            mesh.curl(u) + self.rotation_rate * mesh.coriolis_unit_vertex[:, None]
        ) / torch.where(dual > 0, dual, 1.0)
        v0, v1 = mesh.verticesOnEdge.T
        qe = 0.5 * (pv[v0] + pv[v1])
        ids = mesh.safe_rotation_stencil
        flux = opening * u
        coeff = (
            -0.5 * (qe[:, None, :] + qe[ids]) * mesh.rotation_weights_masked[..., None]
        )
        return torch.where(wet, (coeff * flux[ids]).sum(1), 0.0)


def install_opening_momentum(model):
    """Use a common open thickness in shelf momentum, rotation and ALE."""
    from .shelf_transport import ShelfTemperatureOcean

    if not isinstance(model.dynamics, ShelfTemperatureOcean):
        raise ValueError("matched momentum requires shelf transport")
    native = model.ocean.base.dynamics
    old = native.mechanical
    rotation = OpeningRotation(
        domain=old.domain,
        gravity=old.gravity,
        rho0=old.rho0,
        omega=old.rotation_rate,
        min_thickness=old.min_thickness,
    )
    if model.ocean.column.layer_reference is None or model.ocean.domain is None:
        raise ValueError(
            "matched shelf momentum requires partial columns and a wet domain"
        )
    energy = OpeningEnergy(domain=old.domain, gravity=old.gravity, rho0=old.rho0)
    rotation.energy = energy
    native.mechanical = rotation
    model.ocean.energy.base.mechanical = energy
    model.dynamics.matched_openings = True
    return energy


def project_changed_faces(old, new, velocity):
    """Transfer normal velocity on edges whose set of open slots changed.

    Intersect bins in metres above their common fixed bed. The old surface
    velocity accompanies any extension above the old surface. A previously
    impermeable whole edge starts with zero normal flow. Return the velocity
    and the lost shear energy divided by rho times horizontal edge area.
    """

    def bounds(width):
        top = width.flip((1,)).cumsum(1).flip((1,))
        return torch.cat((top[:, 1:], torch.zeros_like(top[:, :1])), dim=1), top

    old_lo, old_hi = bounds(old)
    new_lo, new_hi = bounds(new)
    overlap = (
        torch.minimum(new_hi[:, :, None], old_hi[:, None, :])
        - torch.maximum(new_lo[:, :, None], old_lo[:, None, :])
    ).clamp_min(0)
    # A zero-width bin cannot donate momentum, even at roundoff-sized cuts.
    u = torch.where(old > 0, velocity, 0.0)
    surface_slot = (old > 0).long().argmax(1, keepdim=True)
    surface_u = u.gather(1, surface_slot)
    exposed = (new_hi - torch.maximum(new_lo, old_hi[:, :1])).clamp_min(0)
    width = overlap.sum(-1) + exposed
    momentum = (overlap * u[:, None, :]).sum(-1) + exposed * surface_u
    mapped = momentum / torch.where(width > 0, width, 1.0)
    variance = (overlap * (u[:, None, :] - mapped[:, :, None]).square()).sum(-1)
    heat = 0.5 * (variance + exposed * (surface_u - mapped).square())
    return mapped, heat
