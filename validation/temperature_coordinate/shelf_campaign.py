"""Exercise native moving-face transport, GM/Redi and ALE on a coastal mesh."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocean_column"))
from vfs.ocean.temperature.roquet import RoquetEnergy, RoquetFreezingEOS
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean.continuity_momentum import (
    reconstruct_cell_velocity,
    transport_with_budget,
)
from vfs.ocean.partial_cells import (
    WetFaces,
    reference_thickness,
    remap_active_contents,
    remap_roquet,
)
from vfs.ocean.shelf_state import ShelfState
from vfs.ocean.partial_cell_eddy import ShelfRedi, advance_gm_roquet


def check_moving_momentum(mesh, depth, h, faces, velocity, rho0):
    """Carry donor momentum and the local TKE reservoir through four steps."""
    cell_velocity = reconstruct_cell_velocity(mesh, h, faces, velocity)
    tke_h = 0.1 * h
    for _ in range(4):
        step, settlement = transport_with_budget(
            mesh, depth, h, velocity, cell_velocity, tke_h, 120.0, rho0
        )
        audit_transport(
            mesh, h, faces, velocity, cell_velocity, tke_h, step, settlement, rho0
        )
        h, faces, velocity = step.h, step.face.faces, step.face.velocity
        cell_velocity, tke_h = step.cell_velocity, settlement.tke_h


def audit_transport(
    mesh, h, faces, velocity, cell_velocity, initial_tke_h, step, settlement, rho0
):
    area = mesh.cell_area
    dual = mesh.dc_edge * mesh.dv_edge
    result = step.face
    new = result.faces
    k0 = 0.5 * rho0 * area[:, None] * h * cell_velocity.square().sum(-1)
    k1 = 0.5 * rho0 * area[:, None] * step.h * step.cell_velocity.square().sum(-1)
    residual_cell = (k1 - k0 + step.cell_mixing_energy).sum().abs() / k0.sum()
    print(
        f"Continuity donors: {int((result.incoming.mass > 0).sum())} receiving segments; "
        f"relative cell kinetic-energy residual {residual_cell:.3g}"
    )
    if residual_cell > 256 * torch.finfo(h.dtype).eps:
        raise ArithmeticError("cell momentum energy budget did not close")

    before_ke = faces.kinetic_energy(velocity, dual, rho0).sum()
    after_ke = new.kinetic_energy(result.velocity, dual, rho0).sum()
    heat = rho0 * (area[:, None] * (settlement.tke_h - initial_tke_h)).sum()
    residual_dual = (after_ke - before_ke + heat).abs() / before_ke
    print(
        f"Relative face KE + TKE residual {residual_dual:.3g}; minimum wet "
        f"TKE {(settlement.tke_h[step.h > 0] / step.h[step.h > 0]).min():.6g} J/kg"
    )
    if residual_dual > 1e-11:
        raise ArithmeticError("split-face transport energy settlement did not close")

    def per_edge(faces, values):
        return dual.new_zeros(dual.shape).index_add(0, faces.edge, values)

    def residual(old_amount, new_amount):
        a, b = per_edge(faces, old_amount), per_edge(new, new_amount)
        # Normalize locally by gross (unsigned) transport to handle signed flow.
        scale = per_edge(faces, old_amount.abs()) + per_edge(new, new_amount.abs())
        return ((a - b).abs() / torch.where(scale > 0, scale, 1.0)).amax().item()

    old_mass = rho0 * dual[faces.edge] * faces.opening
    new_mass = rho0 * dual[new.edge] * new.opening
    mass = residual(old_mass - result.outgoing.mass, new_mass - result.incoming.mass)
    momentum = residual(
        old_mass * velocity - result.outgoing.momentum,
        new_mass * result.velocity - result.incoming.momentum,
    )
    kinetic = residual(
        0.5 * old_mass * velocity.square() - result.outgoing.kinetic,
        0.5 * new_mass * result.velocity.square()
        + result.mixed_kinetic
        - result.incoming.kinetic,
    )
    print(
        f"Moving openings {len(faces.edge)} -> {len(new.edge)}; maximum normalized "
        f"edge residuals: mass {mass:.3g}, momentum {momentum:.3g}, kinetic {kinetic:.3g}"
    )
    if max(mass, momentum, kinetic) > 256 * torch.finfo(h.dtype).eps:
        raise ArithmeticError("moving face momentum budget did not close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--geography", required=True)
    parser.add_argument("--levels", type=int, default=40)
    parser.add_argument("--reference-depth", type=float, default=5000.0)
    parser.add_argument("--surface-layer", type=float, default=10.0)
    parser.add_argument("--merge-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.levels < 2 or not 0 < args.surface_layer < args.reference_depth:
        parser.error("use at least two levels and 0 < surface layer < reference depth")
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    xyz = np.load(args.mesh, allow_pickle=False)
    mesh = VoronoiMesh(
        n_cells=len(xyz), generator_xyz=xyz, lloyd_iters=0, verbose=False
    )
    with np.load(args.geography, allow_pickle=False) as geo:
        depth = torch.from_numpy(np.where(geo["land"], 0.0, geo["depth_m"])).double()
    if depth.shape != (len(xyz),):
        raise ValueError("geography must use the same cell ordering as the mesh")
    tail = torch.logspace(0, 1, args.levels - 1)
    nominal = torch.cat(
        (
            depth.new_tensor([args.surface_layer]),
            tail * (args.reference_depth - args.surface_layer) / tail.sum(),
        )
    )
    h = reference_thickness(depth, nominal, merge_fraction=args.merge_fraction)
    edges = mesh.cellsOnEdge
    faces = WetFaces.from_columns(depth, h, edges)
    area = mesh.cell_area
    dual = mesh.dc_edge * mesh.dv_edge
    eos = RoquetFreezingEOS()
    energy = RoquetEnergy(eos, gravity=9.8, rho0=1025.0)
    z = h.cumsum(1) - 0.5 * h
    surface = (28 - 37 * mesh.cell_xyz[:, 2].square()).clamp_min(-0.054 * 34.0)
    deep = surface.clamp_max(2.0)
    t = deep[:, None] + (surface - deep)[:, None] * torch.exp(-(z - z[:, :1]) / 250.0)
    ht, hs, he = h * t, 34 * h, 0.1 * h
    denom = torch.where(h > 0, h, 1.0)

    def inventory(ht, hs, he, h):
        safe = torch.where(h > 0, h, 1.0)
        ct, sa = ht / safe, hs / safe
        height = h.flip([1]).cumsum(1).flip([1]) - depth[:, None] - h / 2
        # Independent exact integral of the Roquet polynomial in height.
        anomaly = eos.s * (sa - eos.SA0) + (ct - eos.CT0) * (
            eos.t + eos.tt * (ct + eos.CT0)
        )
        psi = (
            energy.gravity
            / energy.rho0
            * (
                height * anomaly
                - 0.5 * eos.tz * (ct - eos.CT0) * (height.square() + h.square() / 12)
            )
        )
        return torch.stack((hs, eos.cp * ht, h * psi, he), -1)

    # Compose moving-face transport, GM, Redi and ALE on the returned state.
    # Keep the independent operator audits below for comparison.
    velocity = 0.02 * torch.sin(faces.left.to(h.dtype))
    native = ShelfState(
        h,
        ht,
        hs,
        0.1 * h,
        he,
        velocity,
        reconstruct_cell_velocity(mesh, h, faces, velocity),
    )

    def totals(state):
        return torch.stack(
            [(area[:, None] * field).sum() for field in (state.h, state.salt_h)]
        )

    def specific_budget(state):
        inv = inventory(state.temp_h, state.salt_h, state.eddy_h, state.h)
        surface = state.h.sum(1) - depth
        current = WetFaces.from_columns(depth, state.h, edges)
        # Keep thermal and mechanical changes separate to retain precision.
        return (
            energy.rho0 * area[:, None, None] * inv[..., 1:],
            energy.rho0 * area[:, None] * state.tke_h,
            0.5 * energy.rho0 * energy.gravity * area * surface.square(),
            current.kinetic_energy(state.velocity, dual, energy.rho0).sum(),
        )

    before = specific_budget(native)
    starting = totals(native)
    for _ in range(4):
        native = native.transport(mesh, depth, 120.0, energy.rho0)
        native = native.mesoscale(mesh, depth, energy, 100.0, 100.0, 120.0)
        target = native.h.clone()
        transfer = 0.01 * target[:, :2].amin(1)
        target[:, 0] += transfer
        target[:, 1] -= transfer
        native = native.rezone(mesh, depth, target, energy)
    torch.testing.assert_close(totals(native), starting, rtol=1e-13, atol=0)
    drift = (
        sum((b - a).sum() for a, b in zip(before, specific_budget(native))) / area.sum()
    )
    print(f"Composed unforced transport/GM/Redi/ALE energy change {drift:.6g} J/m²")
    print(
        "Four composed transport/GM/Redi/ALE steps conserved water and salt.",
        flush=True,
    )
    initial = inventory(ht, hs, he, h)
    for step in range(4):
        b = -energy.gravity / energy.rho0 * eos.anomaly(ht / denom, hs / denom, 0.0)
        redi = ShelfRedi.build(faces, depth, h, b, mesh.dc_edge, mesh.dv_edge, 100.0)
        limit = redi.step_limit(h * area[:, None])
        if step == 0:
            print(
                f"{int((h > 0).sum())} wet layers, {len(faces.edge)} openings, "
                f"initial Redi bound {limit.item():.1f} s",
                flush=True,
            )
        ht, hs, he = advance_gm_roquet(
            faces,
            depth,
            h,
            ht,
            hs,
            he,
            area,
            energy,
            mesh.dv_edge,
            100.0 * redi.slope,
            120.0,
        )
        b = -energy.gravity / energy.rho0 * eos.anomaly(ht / denom, hs / denom, 0.0)
        redi = ShelfRedi.build(faces, depth, h, b, mesh.dc_edge, mesh.dv_edge, 100.0)
        limit = redi.step_limit(h * area[:, None])
        ht, hs, he = redi.advance_roquet(
            depth, h, ht, hs, he, area, energy, min(120.0, 0.8 * limit.item())
        )
    # Prescribe an interior rezone; both the top and the bed stay fixed.
    transfer = torch.where(h[:, 1] > 0, 0.03 * h[:, :2].amin(1), 0.0)
    target = h.clone()
    target[:, 0] += transfer
    target[:, 1] -= transfer
    u = 0.02 * torch.sin(faces.left.to(h.dtype))
    check_moving_momentum(mesh, depth, h, faces, u, energy.rho0)
    result = remap_roquet(depth, h, target, ht, hs, u, edges, area, dual, energy)
    remapped_eddy = remap_active_contents(h, he, target)
    change = inventory(result.temp_h, result.salt_h, remapped_eddy, target) - initial
    change = (area[:, None, None] * change).sum((0, 1))
    total_energy = (
        energy.rho0 * change[1:].sum() + result.edge_kinetic_change.sum()
    ) / area.sum()
    print(
        f"Combined drift per global area: energy {total_energy:.6g} J/m², "
        f"salt {energy.rho0 * 1e-3 * change[0] / area.sum():.6g} kg/m²"
    )
    wet = depth > 0
    residual = result.column_energy_residual / area
    print(
        f"Four GM/Redi steps and one ALE completed; max column ALE residual "
        f"{residual.abs().max():.6g} J/m²; minimum wet EKE "
        f"{(remapped_eddy[target > 0] / target[target > 0]).min():.6g} J/kg; "
        f"{int(wet.sum())} wet columns"
    )


if __name__ == "__main__":
    main()
