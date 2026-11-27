"""Check drag directionality on smooth solid rotation and exact no-ice behavior."""

import argparse
import json
from dataclasses import replace

import torch

from .operators import edge_drag
from .replay import load_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    with torch.no_grad():
        owner, initial = load_restart(args.restart)
        s, ice = initial.ocean.ocean, initial.ice
        mesh = s.mesh
        domain = owner.ocean.ocean.domain
        axis = torch.tensor([0.3, 0.4, 0.5])
        axis = axis / axis.norm()
        vector = torch.linalg.cross(axis.expand_as(mesh.edge_xyz), mesh.edge_xyz)
        native = (vector * mesh.edge_normal).sum(-1, keepdim=True)
        u = domain.apply_edge(native)
        oc = s.with_tensors(
            torch.cat((u, s.u_e[:, 1:]), -1), s.h, s.temp_h, s.salt_h, s.tracer_h
        )
        areas = torch.zeros_like(ice.area)
        areas[:, 0] = domain.wet_cell
        pack = replace(
            ice,
            area=areas,
            volume=areas.clone(),
            velocity=torch.zeros_like(ice.velocity),
        )
        st = replace(initial, ice=pack, ocean=replace(initial.ocean, ocean=oc))
        free = replace(st, ice=replace(pack, area=0.0 * areas, volume=0.0 * areas))
        same, report = edge_drag(owner, free, 600)
        assert torch.equal(same.ocean.ocean.u_e, oc.u_e) and torch.equal(
            same.ocean.tke_h, free.ocean.tke_h
        )
        assert (
            torch.equal(same.ice.velocity, free.ice.velocity)
            and report.tke_gain.sum() == 0
        )
        dt = 60.0
        mi = owner.hamiltonian.metric(st)
        mo = owner.ocean.energy.base.mechanical.metric(s)[:, :1]
        interior = (
            ~mesh.valid_edges_on_cell | domain.active_edge[mesh.safe_edges_on_cell]
        ).all(1)
        select = interior[mesh.cellsOnEdge].all(1)
        speed = vector.norm(dim=-1, keepdim=True)
        gamma = (
            dt
            * s.column.rho0
            * owner.drag_coefficient
            * (mesh.dc_edge * mesh.dv_edge)[:, None]
        )
        inverse = 1 / mi[select] + 1 / mo[select]
        target = u[select] / (1 + gamma[select] * inverse * speed[select])
        for normal in [True, False]:
            end, _ = edge_drag(owner, st, dt, normal_only=normal)
            after = (end.ocean.ocean.u_e[:, :1] - end.ice.velocity)[select]
            error = (after - target).norm() / (u[select] - target).norm()
            print(
                json.dumps(
                    {
                        "normal_only": normal,
                        "interior_edges": int(select.sum()),
                        "relative_rms_drag_impulse_error": error.item(),
                    }
                ),
                flush=True,
            )
            if not normal:
                assert error < 0.01


if __name__ == "__main__":
    main()
