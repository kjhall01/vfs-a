"""Candidate operators for a controlled ocean–ice replay, outside the campaign."""

import math
from dataclasses import replace

import torch

from vfs.geometry.mesh.rotation import cell_to_dual
from vfs.ice.ice_drag import DragReport
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.partial_cells import layer_interfaces


def edge_drag(coupled, state, dt, *, normal_only=False):
    """Exchange equal impulses on each wet edge and deposit lost KE in surface TKE.

    Use the native normal slip. A tangent reconstructed from the neighboring
    cells supplies the second component of the quadratic speed. Holding the
    initial direction fixed over one substep gives a rational decay factor.
    For purely normal flow the scalar quadratic-drag solution is exact.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("expected positive drag interval")
    coupled.check(state)
    s, ice = state.ocean.ocean, state.ice
    mesh = s.mesh
    mi = coupled.hamiltonian.metric(state)
    mo = coupled.ocean.energy.base.mechanical.metric(s)[:, :1]
    active = coupled.ridge.active_edges(state) & (mo > 0)
    mi_safe = torch.where(active, mi, 1.0)
    mo_safe = torch.where(active, mo, 1.0)
    slip = torch.where(active, s.u_e[:, :1] - ice.velocity, 0.0)
    speed = slip.abs()
    if not normal_only:
        relative = mesh.cell_wind_vector(slip)[:, 0, :]
        tangent = (mesh.cell_to_edge(relative) * mesh.edge_tangent).sum(
            -1, keepdim=True
        )
        speed = torch.linalg.vector_norm(
            torch.cat((slip, tangent), -1), dim=-1, keepdim=True
        )
    coverage = ice.area.sum(-1, keepdim=True)
    edge_area = (mesh.dc_edge * mesh.dv_edge)[:, None]
    drag = (
        s.column.rho0
        * coupled.drag_coefficient
        * edge_area
        * mesh.cell_to_edge(coverage)
    )
    strength = dt * speed * drag
    impulse = slip * (strength / (1.0 + strength * (1.0 / mi_safe + 1.0 / mo_safe)))
    ui = ice.velocity + impulse / mi_safe
    uo = s.u_e[:, :1] - impulse / mo_safe
    edge_heat = impulse * (slip + uo - ui) * 0.5
    # Share each edge's heat in proportion to its participating ice area.
    side = mesh.edge_side_area * coverage[mesh.cellsOnEdge, 0]
    denom = side.sum(-1, keepdim=True)
    share = side / torch.where(denom > 0, denom, 1.0)
    heat = s.h.new_zeros(mesh.nCells).index_add(
        0, mesh.cellsOnEdge.flatten(), (share * edge_heat).flatten()
    )
    mass = coupled.ocean.energy.mass_area(s)
    dk = heat[:, None] / torch.where(mass > 0, mass, 1.0)
    ocean = s.with_tensors(
        torch.cat((uo, s.u_e[:, 1:]), -1), s.h, s.temp_h, s.salt_h, s.tracer_h
    )
    tke = torch.cat((state.ocean.tke_h[:, :1] + dk, state.ocean.tke_h[:, 1:]), -1)
    end = replace(
        state,
        ice=replace(ice, velocity=ui),
        ocean=replace(state.ocean, ocean=ocean, tke_h=tke),
    )
    dpi = mi * (ui - ice.velocity)
    dpo = mo * (uo - s.u_e[:, :1])
    dke = (dpi * (ui + ice.velocity) * 0.5 + dpo * (uo + s.u_e[:, :1]) * 0.5).sum()
    coupled.check(end)
    return end, DragReport(heat, dpi + dpo, dke + heat.sum(), dke.new_zeros(()), 0)


class OpeningEnergy(ConstantDensityEnergy):
    """KE = sum(rho * dc * dv * open_thickness * u**2 / 2) on native slots."""

    def geometry(self, s, h=None):
        mesh, domain = s.mesh, self.domain
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


class MatchedOpeningEnergy(OpeningEnergy):
    def edge_thickness(self, state, h):
        return self.geometry(state, h)[0]

    def column_kinetic_energy(self, state):
        edge_ke = 0.5 * self.metric(state) * state.u_e.square()
        return self.interpolation_transpose(state.mesh, edge_ke).sum(-1)


class OpeningRotation(ConstantDensityDynamics):
    def rotation_acceleration(self, s):
        """Use the same skew PV operator, with the open transport in each slot."""
        mesh = s.mesh
        u = self.energy.velocity(s)
        dual = cell_to_dual(mesh, self.domain.apply_cell(s.h))
        pv = (
            mesh.curl(u) + self.rotation_rate * mesh.coriolis_unit_vertex[:, None]
        ) / torch.where(dual > 0, dual, 1.0)
        v0, v1 = mesh.verticesOnEdge.T
        qe = 0.5 * (pv[v0] + pv[v1])
        ids = mesh.safe_rotation_stencil
        flux = self.energy.geometry(s)[0] * u
        coeff = (
            -0.5 * (qe[:, None, :] + qe[ids]) * mesh.rotation_weights_masked[..., None]
        )
        return self.domain.apply_layer_edge((coeff * flux[ids]).sum(1))


def use_openings(model, *, match_ale=True):
    """Install the candidate in one new model instance used by the experiment."""
    native = model.ocean.base.dynamics
    old = native.mechanical
    rotation = OpeningRotation(
        domain=old.domain,
        gravity=old.gravity,
        rho0=old.rho0,
        omega=old.rotation_rate,
        min_thickness=old.min_thickness,
    )
    kind = MatchedOpeningEnergy if match_ale else OpeningEnergy
    energy = kind(domain=old.domain, gravity=old.gravity, rho0=old.rho0)
    rotation.energy = energy
    native.mechanical = rotation
    model.ocean.energy.base.mechanical = energy
    return energy
