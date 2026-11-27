"""Reciprocal ice–ocean drag with kinetic energy loss deposited in ocean TKE.

The midpoint solver reconstructs the relative velocity in cells. The native
edge scheme exchanges impulses directly between ice and ocean velocity slots.
"""

import math
from dataclasses import dataclass, replace
import torch
from vfs.ice.ocean_ice import vector_transpose, safe


@dataclass(frozen=True)
class DragReport:
    tke_gain: torch.Tensor  # J/cell
    momentum_residual: torch.Tensor  # edge generalized momentum
    energy_residual: torch.Tensor  # J, KE+TKE change
    relative_solver_residual: torch.Tensor
    iterations: int


def cg(action, b, tolerance, diagonal, max_iterations=500):
    x = torch.zeros_like(b)
    r = b
    z = r / diagonal
    p = z
    rr = (r * z).sum()
    initial = (r * r).sum().detach()
    if initial == 0:
        return x
    for _ in range(max_iterations):
        Ap = action(p)
        denominator = (p * Ap).sum()
        if denominator <= 0:
            raise FloatingPointError("drag Jacobian is not positive definite")
        alpha = rr / denominator
        x = x + alpha * p
        r = r - alpha * Ap
        if (r * r).sum().detach() <= tolerance**2 * initial:
            return x
        z = r / diagonal
        next_rr = (r * z).sum()
        p = z + (next_rr / rr) * p
        rr = next_rr
    raise FloatingPointError("drag CG did not converge")


def midpoint_drag(coupled, state, dt, tolerance=None):
    """Use the FP64 default or 32 ulps for the native-precision Newton residual.

    A fixed 1e-12 relative residual cannot be resolved in FP32. Explicit caller
    tolerances remain literal. Reciprocal impulses and TKE work are unchanged.
    """
    coupled.check(state)
    H = coupled.hamiltonian
    s = state.ocean.ocean
    if tolerance is None:
        tolerance = max(1e-12, 32 * torch.finfo(s.h.dtype).eps)
    mesh = s.mesh
    Mi = H.metric(state)
    Mo = coupled.ocean.energy.base.mechanical.metric(s)[:, :1]
    active = coupled.ridge.active_edges(state)
    B = torch.where(active, 1 / safe(Mi) + 1 / safe(Mo), 0.0)
    root = torch.sqrt(B)
    relative = s.u_e[:, :1] - state.ice.velocity
    b = torch.where(active, relative / safe(root), 0.0)
    coefficient = (
        mesh.cell_area[:, None]
        * state.ice.area.sum(-1, keepdim=True)
        * s.column.rho0
        * coupled.drag_coefficient
    )
    reconstruct = lambda x: mesh.cell_wind_vector(root * x)[:, 0, :]
    transpose = lambda x: root * vector_transpose(mesh, x)
    y = b
    norm_b = torch.linalg.vector_norm(b).detach()
    if norm_b == 0:
        norm_b = torch.ones_like(norm_b)
    for iteration in range(20):
        v = reconstruct(y)
        speed = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        force = vector_transpose(mesh, coefficient * speed * v)
        residual = y + 0.5 * dt * root * force - b
        relative_residual = torch.linalg.vector_norm(residual) / norm_b
        if relative_residual.detach() <= tolerance:
            break

        def jacobian(direction):
            d = reconstruct(direction)
            stress_derivative = coefficient * (
                speed * d + v * (v * d).sum(-1, keepdim=True) / safe(speed)
            )
            return direction + 0.5 * dt * transpose(stress_derivative)

        # Exact diagonal of I + dt/2 sqrt(B) R^T D R sqrt(B).
        weights = (
            mesh.cell_rec_w1[..., None] * mesh.cell_rec_basis1[:, None, :]
            + mesh.cell_rec_w2[..., None] * mesh.cell_rec_basis2[:, None, :]
        ) * mesh.edgeSignOnCell[..., None]
        entries = coefficient * (
            speed * weights.square().sum(-1)
            + (weights * v[:, None, :]).sum(-1).square() / safe(speed)
        )
        entries = torch.where(mesh.valid_edges_on_cell, entries, 0.0)
        diagonal = 1 + 0.5 * dt * B * B.new_zeros(B.shape).index_add(
            0, mesh.safe_edges_on_cell.flatten(), entries.reshape(-1, 1)
        )
        y = y - cg(jacobian, residual, tolerance * 0.1, diagonal)
    else:
        raise FloatingPointError("implicit drag midpoint solve did not converge")
    impulse = dt * torch.where(active, force, 0.0)
    ui = state.ice.velocity + impulse / safe(Mi)
    uo = s.u_e[:, :1] - impulse / safe(Mo)
    gain = (dt * coefficient * speed.pow(3))[:, 0]
    mass = coupled.ocean.energy.mass_area(s)
    dk = torch.cat((gain[:, None] / safe(mass), torch.zeros_like(s.h[:, 1:])), -1)
    physical = s.with_tensors(
        torch.cat((uo, s.u_e[:, 1:]), -1), s.h, s.temp_h, s.salt_h, s.tracer_h
    )
    end = replace(
        state,
        ocean=replace(state.ocean, ocean=physical, tke_h=state.ocean.tke_h + dk),
        ice=replace(state.ice, velocity=ui),
    )
    dki = (Mi * (ui - state.ice.velocity) * 0.5 * (ui + state.ice.velocity)).sum()
    dko = (Mo * (uo - s.u_e[:, :1]) * 0.5 * (uo + s.u_e[:, :1])).sum()
    momentum = Mi * (ui - state.ice.velocity) + Mo * (uo - s.u_e[:, :1])
    coupled.check(end)
    return end, DragReport(
        gain, momentum, dki + dko + gain.sum(), relative_residual, iteration
    )


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
