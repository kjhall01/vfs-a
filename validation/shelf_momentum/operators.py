"""Candidate operators for a controlled ocean–ice replay, outside the campaign."""

import math
from dataclasses import replace

import torch

from vfs.ice.ice_drag import DragReport
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.temperature.shelf_momentum import OpeningEnergy, install_opening_momentum


def use_openings(model, *, match_ale=True):
    energy = install_opening_momentum(model)
    if not match_ale:
        # Deliberately inconsistent control for validation/shelf_momentum/ale_check.py.
        energy.edge_thickness = ConstantDensityEnergy.edge_thickness.__get__(energy)
        energy.column_kinetic_energy = ConstantDensityEnergy.column_kinetic_energy.__get__(energy)
        model.dynamics.matched_openings = False
    return energy


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
